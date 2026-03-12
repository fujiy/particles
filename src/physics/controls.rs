use bevy::log::tracing;
use bevy::prelude::*;

use crate::params::ActivePhysicsParams;
use crate::physics::gpu_mpm::buffers::GpuParticle;
use crate::physics::gpu_mpm::gpu_resources::{MpmGpuControl, MpmGpuUploadRequest};
use crate::physics::gpu_mpm::phase::{
    MPM_PHASE_ID_GRANULAR_SAND, MPM_PHASE_ID_GRANULAR_SOIL, MPM_PHASE_ID_WATER,
    mpm_phase_id_for_particle,
};
use crate::physics::gpu_mpm::sync::{
    MpmFullParticleReadbackCache, MpmFullParticleReadbackRequest, MpmParticleReadbackStatus,
    PendingGpuParticleAdds,
};
use crate::physics::material::{MaterialParams, ParticleMaterial, terrain_boundary_radius_m};
use crate::physics::save_load;
use crate::physics::scenario::{
    apply_scenario_spec_to_terrain, count_solid_cells, default_scenario_spec_by_name,
    write_scenario_artifacts_for_snapshot,
};
use crate::physics::state::{
    LoadDefaultWorldRequest, LoadMapRequest, ReplayLoadScenarioRequest, ReplaySaveArtifactRequest,
    ReplayState, ResetSimulationRequest, SaveMapRequest, SimulationState,
};
use crate::physics::world::terrain::TerrainWorld;

fn material_from_phase(phase_id: u32) -> Option<ParticleMaterial> {
    match phase_id as u8 {
        MPM_PHASE_ID_WATER => Some(ParticleMaterial::WaterLiquid),
        MPM_PHASE_ID_GRANULAR_SOIL => Some(ParticleMaterial::SoilGranular),
        MPM_PHASE_ID_GRANULAR_SAND => Some(ParticleMaterial::SandGranular),
        _ => None,
    }
}

fn snapshot_particles_from_gpu_readback(
    particles: &[GpuParticle],
) -> Result<Vec<save_load::SnapshotParticle>, String> {
    let snapshot_particles: Vec<save_load::SnapshotParticle> = particles
        .iter()
        .filter_map(|p| {
            Some(save_load::SnapshotParticle {
                position: Vec2::from_array(p.x),
                velocity: Vec2::from_array(p.v),
                material: material_from_phase(p.phase_id)?,
            })
        })
        .collect();
    if snapshot_particles.len() != particles.len() {
        return Err("GPU readback contains unknown phase ids".to_string());
    }
    Ok(snapshot_particles)
}

fn gpu_particles_from_snapshot(
    particles: &[save_load::SnapshotParticle],
    rho0: f32,
) -> Vec<GpuParticle> {
    particles
        .iter()
        .filter_map(|particle| {
            let phase_id = mpm_phase_id_for_particle(particle.material)?;
            let mass = crate::physics::material::particle_properties(particle.material).mass;
            Some(GpuParticle::from_cpu(
                particle.position,
                particle.velocity,
                mass,
                mass.max(0.0) / rho0,
                Mat2::IDENTITY,
                Mat2::ZERO,
                0.0,
                phase_id,
            ))
        })
        .collect()
}

#[derive(Resource, Debug, Default)]
pub(crate) struct PendingReplayArtifactSave {
    pub save_final: Option<bool>,
}

#[derive(Resource, Debug, Default)]
pub(crate) struct PendingMapSave {
    pub slot_name: Option<String>,
}

pub(crate) fn initialize_default_world(
    mut terrain_world: ResMut<TerrainWorld>,
    mut sim_state: ResMut<SimulationState>,
    material_params: Res<MaterialParams>,
) {
    let terrain_boundary_radius_m = terrain_boundary_radius_m(*material_params);
    terrain_world.set_generation_enabled(true);
    terrain_world.clear();
    terrain_world.rebuild_static_particles_if_dirty(terrain_boundary_radius_m);
    sim_state.running = false;
    sim_state.step_once = false;
}

pub(crate) fn handle_sim_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut sim_state: ResMut<SimulationState>,
    mut reset_writer: MessageWriter<ResetSimulationRequest>,
    mut save_writer: MessageWriter<SaveMapRequest>,
    mut load_writer: MessageWriter<LoadMapRequest>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.running = !sim_state.running;
    }
    if keyboard.just_pressed(KeyCode::Period) || keyboard.just_pressed(KeyCode::NumpadDecimal) {
        sim_state.running = false;
        sim_state.step_once = true;
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        reset_writer.write(ResetSimulationRequest);
    }

    if keyboard.just_pressed(KeyCode::F5) {
        save_writer.write(SaveMapRequest {
            slot_name: save_load::DEFAULT_QUICK_SAVE_SLOT.to_string(),
        });
    }

    if keyboard.just_pressed(KeyCode::F9) {
        load_writer.write(LoadMapRequest {
            slot_name: save_load::DEFAULT_QUICK_SAVE_SLOT.to_string(),
        });
    }
}

pub(crate) fn handle_replay_requests(
    mut load_reader: MessageReader<ReplayLoadScenarioRequest>,
    mut save_reader: MessageReader<ReplaySaveArtifactRequest>,
    mut replay_state: ResMut<ReplayState>,
    mut sim_state: ResMut<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut gpu_upload: ResMut<MpmGpuUploadRequest>,
    mut gpu_control: ResMut<MpmGpuControl>,
    mut readback_status: ResMut<MpmParticleReadbackStatus>,
    mut full_readback_request: ResMut<MpmFullParticleReadbackRequest>,
    mut full_readback_cache: ResMut<MpmFullParticleReadbackCache>,
    mut pending_gpu_adds: ResMut<PendingGpuParticleAdds>,
    mut pending_replay_save: ResMut<PendingReplayArtifactSave>,
    active_params: Res<ActivePhysicsParams>,
) {
    for request in load_reader.read() {
        let Some(spec) = default_scenario_spec_by_name(&request.scenario_name) else {
            replay_state.status_message = format!("Unknown scenario: {}", request.scenario_name);
            continue;
        };
        sim_state.running = false;
        sim_state.step_once = false;
        match apply_scenario_spec_to_terrain(&spec, &mut terrain_world) {
            Ok(snapshot_particles) => {
                let rho0 = active_params.0.water.rho0.max(1.0e-6);
                gpu_upload.particles = gpu_particles_from_snapshot(&snapshot_particles, rho0);
                gpu_upload.particle_revision = gpu_upload
                    .particle_revision
                    .max(readback_status.particle_revision)
                    .max(pending_gpu_adds.latest_revision())
                    .saturating_add(1);
                gpu_upload.upload_particles = true;
                gpu_upload.mover_results.clear();
                gpu_upload.upload_mover_results = true;
                readback_status.particle_count = gpu_upload.particles.len() as u32;
                readback_status.particle_revision = gpu_upload.particle_revision;
                full_readback_cache.payload = Some(
                    crate::physics::gpu_mpm::readback::GpuParticleReadbackPayload {
                        particles: gpu_upload.particles.clone(),
                        particle_revision: gpu_upload.particle_revision,
                    },
                );
                full_readback_request.requested = false;
                full_readback_request.min_particle_revision = 0;
                pending_gpu_adds.clear();
                gpu_control.readback_enabled = true;
                gpu_control.readback_interval_frames = 1;
                replay_state.enabled = true;
                replay_state.scenario_name = Some(spec.name.clone());
                replay_state.scenario_total_steps = spec.step_count;
                replay_state.current_step = 0;
                replay_state.baseline_particle_count = snapshot_particles.len();
                replay_state.baseline_solid_cell_count = count_solid_cells(&terrain_world);
                replay_state.status_message = if spec.name == "water_drop" {
                    format!("Loaded replay scenario: {} (MPM water enabled)", spec.name)
                } else {
                    format!("Loaded replay scenario: {}", spec.name)
                };
                sim_state.running = false;
                sim_state.step_once = false;
            }
            Err(error) => {
                replay_state.status_message =
                    format!("Failed to load replay scenario {}: {error}", spec.name);
            }
        }
    }

    for request in save_reader.read() {
        pending_replay_save.save_final = Some(request.save_final);
    }
    let Some(save_final) = pending_replay_save.save_final else {
        return;
    };
    let Some(name) = replay_state.scenario_name.clone() else {
        replay_state.status_message = "Replay scenario is not loaded".to_string();
        pending_replay_save.save_final = None;
        return;
    };
    let Some(spec) = default_scenario_spec_by_name(&name) else {
        replay_state.status_message = format!("Unknown replay scenario: {name}");
        pending_replay_save.save_final = None;
        return;
    };

    if save_final {
        sim_state.running = false;
        sim_state.step_once = false;
        replay_state.current_step = spec.step_count;
    }

    let min_particle_revision = readback_status.particle_revision;
    let Some(payload) = full_readback_cache
        .payload
        .as_ref()
        .filter(|payload| payload.particle_revision >= min_particle_revision)
    else {
        full_readback_request.request(min_particle_revision);
        return;
    };
    let snapshot_particles = match snapshot_particles_from_gpu_readback(&payload.particles) {
        Ok(snapshot_particles) => snapshot_particles,
        Err(error) => {
            replay_state.status_message = format!("Failed to capture replay particles: {error}");
            pending_replay_save.save_final = None;
            return;
        }
    };
    match write_scenario_artifacts_for_snapshot(
        &spec,
        replay_state.current_step,
        &terrain_world,
        &snapshot_particles,
    ) {
        Ok(path) => {
            replay_state.status_message = format!("Saved replay artifact: {}", path.display());
            pending_replay_save.save_final = None;
        }
        Err(error) => {
            replay_state.status_message = format!("Failed to save replay artifact: {error}");
            pending_replay_save.save_final = None;
        }
    }
}

pub(crate) fn apply_sim_reset(
    mut reset_reader: MessageReader<ResetSimulationRequest>,
    mut default_world_reader: MessageReader<LoadDefaultWorldRequest>,
    mut sim_state: ResMut<SimulationState>,
    mut replay_state: ResMut<ReplayState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut gpu_upload: ResMut<MpmGpuUploadRequest>,
    mut gpu_control: ResMut<MpmGpuControl>,
    mut readback_status: ResMut<MpmParticleReadbackStatus>,
    mut full_readback_request: ResMut<MpmFullParticleReadbackRequest>,
    mut full_readback_cache: ResMut<MpmFullParticleReadbackCache>,
    mut pending_gpu_adds: ResMut<PendingGpuParticleAdds>,
    material_params: Res<MaterialParams>,
    active_params: Res<ActivePhysicsParams>,
) {
    let has_default_world_request = default_world_reader.read().next().is_some();
    let has_reset_request = reset_reader.read().next().is_some();
    if !has_default_world_request && !has_reset_request {
        return;
    }

    if !has_default_world_request && replay_state.enabled {
        if let Some(name) = replay_state.scenario_name.clone() {
            if let Some(spec) = default_scenario_spec_by_name(&name) {
                let snapshot_particles =
                    match apply_scenario_spec_to_terrain(&spec, &mut terrain_world) {
                        Ok(snapshot_particles) => snapshot_particles,
                        Err(error) => {
                            replay_state.status_message =
                                format!("Replay reset failed for {}: {error}", spec.name);
                            return;
                        }
                    };
                let rho0 = active_params.0.water.rho0.max(1.0e-6);
                gpu_upload.particles = gpu_particles_from_snapshot(&snapshot_particles, rho0);
                gpu_upload.particle_revision = gpu_upload
                    .particle_revision
                    .max(readback_status.particle_revision)
                    .max(pending_gpu_adds.latest_revision())
                    .saturating_add(1);
                gpu_upload.upload_particles = true;
                gpu_upload.mover_results.clear();
                gpu_upload.upload_mover_results = true;
                readback_status.particle_count = gpu_upload.particles.len() as u32;
                readback_status.particle_revision = gpu_upload.particle_revision;
                full_readback_cache.payload = Some(
                    crate::physics::gpu_mpm::readback::GpuParticleReadbackPayload {
                        particles: gpu_upload.particles.clone(),
                        particle_revision: gpu_upload.particle_revision,
                    },
                );
                full_readback_request.requested = false;
                full_readback_request.min_particle_revision = 0;
                pending_gpu_adds.clear();
                gpu_control.readback_enabled = true;
                gpu_control.readback_interval_frames = 1;
                replay_state.scenario_total_steps = spec.step_count;
                replay_state.current_step = 0;
                replay_state.baseline_particle_count = snapshot_particles.len();
                replay_state.baseline_solid_cell_count = count_solid_cells(&terrain_world);
                sim_state.running = false;
                sim_state.step_once = false;
                replay_state.status_message = if spec.name == "water_drop" {
                    format!(
                        "Reloaded replay scenario: {} (MPM water enabled)",
                        spec.name
                    )
                } else {
                    format!("Reloaded replay scenario: {}", spec.name)
                };
                return;
            }
        }
    }

    terrain_world.set_generation_enabled(true);
    terrain_world.clear();
    terrain_world.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(*material_params));
    gpu_upload.particles.clear();
    gpu_upload.particle_revision = gpu_upload
        .particle_revision
        .max(readback_status.particle_revision)
        .max(pending_gpu_adds.latest_revision())
        .saturating_add(1);
    gpu_upload.upload_particles = true;
    gpu_upload.mover_results.clear();
    gpu_upload.upload_mover_results = true;
    readback_status.particle_count = 0;
    readback_status.particle_revision = gpu_upload.particle_revision;
    full_readback_cache.payload = Some(
        crate::physics::gpu_mpm::readback::GpuParticleReadbackPayload {
            particles: Vec::new(),
            particle_revision: gpu_upload.particle_revision,
        },
    );
    full_readback_request.requested = false;
    full_readback_request.min_particle_revision = 0;
    pending_gpu_adds.clear();
    gpu_control.readback_enabled = true;
    gpu_control.readback_interval_frames = 1;
    sim_state.running = false;
    sim_state.step_once = false;
    replay_state.enabled = false;
    replay_state.scenario_name = None;
    replay_state.current_step = 0;
    replay_state.scenario_total_steps = 0;
    replay_state.loop_enabled = false;
    replay_state.baseline_particle_count = 0;
    replay_state.baseline_solid_cell_count = 0;
    replay_state.status_message.clear();
}

pub(crate) fn apply_save_load_requests(
    mut save_reader: MessageReader<SaveMapRequest>,
    mut load_reader: MessageReader<LoadMapRequest>,
    mut sim_state: ResMut<SimulationState>,
    mut replay_state: ResMut<ReplayState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut gpu_upload: ResMut<MpmGpuUploadRequest>,
    mut gpu_control: ResMut<MpmGpuControl>,
    mut readback_status: ResMut<MpmParticleReadbackStatus>,
    mut full_readback_request: ResMut<MpmFullParticleReadbackRequest>,
    mut full_readback_cache: ResMut<MpmFullParticleReadbackCache>,
    mut pending_gpu_adds: ResMut<PendingGpuParticleAdds>,
    mut pending_map_save: ResMut<PendingMapSave>,
    active_params: Res<ActivePhysicsParams>,
) {
    for request in save_reader.read() {
        pending_map_save.slot_name = Some(request.slot_name.clone());
    }

    if let Some(slot_name) = pending_map_save.slot_name.clone() {
        gpu_control.readback_enabled = true;
        gpu_control.readback_interval_frames = 1;
        let min_particle_revision = readback_status.particle_revision;
        let Some(payload) = full_readback_cache
            .payload
            .as_ref()
            .filter(|payload| payload.particle_revision >= min_particle_revision)
        else {
            full_readback_request.request(min_particle_revision);
            return;
        };
        match snapshot_particles_from_gpu_readback(&payload.particles) {
            Ok(snapshot_particles) => match save_load::save_to_slot_with_particles(
                &slot_name,
                &terrain_world,
                &snapshot_particles,
                &sim_state,
            ) {
                Ok(path) => {
                    tracing::info!("saved map to {}", path.display());
                    pending_map_save.slot_name = None;
                }
                Err(error) => tracing::error!("failed to save map: {error}"),
            },
            Err(error) => tracing::error!("failed to save map: {error}"),
        }
    }

    for request in load_reader.read() {
        match save_load::load_from_slot_with_particles(
            &request.slot_name,
            &mut terrain_world,
            &mut sim_state,
        ) {
            Ok((path, snapshot_particles)) => {
                // Force a full clear/upload on GPU path even when particle count is zero.
                let rho0 = active_params.0.water.rho0.max(1.0e-6);
                gpu_upload.particles = gpu_particles_from_snapshot(&snapshot_particles, rho0);
                gpu_upload.particle_revision = gpu_upload
                    .particle_revision
                    .max(readback_status.particle_revision)
                    .max(pending_gpu_adds.latest_revision())
                    .saturating_add(1);
                gpu_upload.upload_particles = true;
                gpu_upload.mover_results.clear();
                gpu_upload.upload_mover_results = true;
                readback_status.particle_count = gpu_upload.particles.len() as u32;
                readback_status.particle_revision = gpu_upload.particle_revision;
                full_readback_cache.payload = Some(
                    crate::physics::gpu_mpm::readback::GpuParticleReadbackPayload {
                        particles: gpu_upload.particles.clone(),
                        particle_revision: gpu_upload.particle_revision,
                    },
                );
                full_readback_request.requested = false;
                full_readback_request.min_particle_revision = 0;
                pending_gpu_adds.clear();
                gpu_control.readback_enabled = true;
                gpu_control.readback_interval_frames = 1;

                replay_state.enabled = false;
                replay_state.scenario_name = None;
                replay_state.current_step = 0;
                replay_state.scenario_total_steps = 0;
                replay_state.loop_enabled = false;
                replay_state.baseline_particle_count = 0;
                replay_state.baseline_solid_cell_count = 0;
                replay_state.status_message.clear();
                tracing::info!("loaded map from {}", path.display());
            }
            Err(error) => tracing::error!("failed to load map: {error}"),
        }
    }
}
