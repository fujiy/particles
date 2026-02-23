use bevy::log::tracing;
use bevy::prelude::*;

use super::step::step_simulation_once;
use crate::physics::material::{MaterialParams, terrain_boundary_radius_m};
use crate::physics::save_load;
use crate::physics::scenario::{
    apply_scenario_spec, count_solid_cells, default_scenario_spec_by_name,
    write_scenario_artifacts_for_state,
};
use crate::physics::state::{
    LoadDefaultWorldRequest, LoadMapRequest, ReplayLoadScenarioRequest, ReplaySaveArtifactRequest,
    ReplayState, ResetSimulationRequest, SaveMapRequest, SimulationParallelSettings,
    SimulationState, TerrainStreamingSettings,
};
use crate::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::ParticleWorld;
use crate::physics::world::terrain::{CHUNK_SIZE_I32, TerrainWorld, world_to_cell};

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

pub(crate) fn stream_terrain_around_camera(
    mut terrain_world: ResMut<TerrainWorld>,
    replay_state: Res<ReplayState>,
    streaming_settings: Res<TerrainStreamingSettings>,
    camera_transforms: Query<&Transform, With<Camera2d>>,
) {
    if replay_state.enabled || !streaming_settings.enabled {
        return;
    }
    let Some(camera_transform) = camera_transforms.iter().next() else {
        return;
    };
    let center_cell = world_to_cell(camera_transform.translation.truncate());
    let center_chunk = IVec2::new(
        center_cell.x.div_euclid(CHUNK_SIZE_I32),
        center_cell.y.div_euclid(CHUNK_SIZE_I32),
    );
    let radius = streaming_settings.load_radius_chunks.max(0);
    for y in (center_chunk.y - radius)..=(center_chunk.y + radius) {
        for x in (center_chunk.x - radius)..=(center_chunk.x + radius) {
            terrain_world.ensure_chunk_loaded(IVec2::new(x, y));
        }
    }
    terrain_world.unload_pristine_chunks_outside_radius(center_chunk, radius + 1);
}

pub(crate) fn handle_replay_requests(
    mut load_reader: MessageReader<ReplayLoadScenarioRequest>,
    mut save_reader: MessageReader<ReplaySaveArtifactRequest>,
    mut replay_state: ResMut<ReplayState>,
    mut sim_state: ResMut<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
    parallel_settings: Res<SimulationParallelSettings>,
    material_params: Res<MaterialParams>,
) {
    for request in load_reader.read() {
        let Some(spec) = default_scenario_spec_by_name(&request.scenario_name) else {
            replay_state.status_message = format!("Unknown scenario: {}", request.scenario_name);
            continue;
        };
        *particle_world = ParticleWorld::default();
        object_world.clear();
        object_field.clear();
        sim_state.running = false;
        sim_state.step_once = false;
        match apply_scenario_spec(
            &spec,
            &mut terrain_world,
            &mut particle_world,
            &mut object_world,
            &mut object_field,
        ) {
            Ok(()) => {
                replay_state.enabled = true;
                replay_state.scenario_name = Some(spec.name.clone());
                replay_state.scenario_total_steps = spec.step_count;
                replay_state.current_step = 0;
                replay_state.baseline_particle_count = particle_world.particle_count();
                replay_state.baseline_solid_cell_count = count_solid_cells(&terrain_world);
                replay_state.status_message = format!("Loaded replay scenario: {}", spec.name);
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
        let Some(name) = replay_state.scenario_name.clone() else {
            replay_state.status_message = "Replay scenario is not loaded".to_string();
            continue;
        };
        let Some(spec) = default_scenario_spec_by_name(&name) else {
            replay_state.status_message = format!("Unknown replay scenario: {name}");
            continue;
        };

        if request.save_final {
            sim_state.running = false;
            sim_state.step_once = false;
            while replay_state.current_step < spec.step_count {
                let _ = step_simulation_once(
                    &mut terrain_world,
                    &mut particle_world,
                    &mut object_world,
                    &mut object_field,
                    parallel_settings.enabled,
                    terrain_boundary_radius_m(*material_params),
                );
                replay_state.current_step = replay_state.current_step.saturating_add(1);
            }
        }

        match write_scenario_artifacts_for_state(
            &spec,
            replay_state.current_step,
            &terrain_world,
            &particle_world,
            &object_world,
            &object_field,
        ) {
            Ok(path) => {
                replay_state.status_message = format!("Saved replay artifact: {}", path.display());
            }
            Err(error) => {
                replay_state.status_message = format!("Failed to save replay artifact: {error}");
            }
        }
    }
}

pub(crate) fn apply_sim_reset(
    mut reset_reader: MessageReader<ResetSimulationRequest>,
    mut default_world_reader: MessageReader<LoadDefaultWorldRequest>,
    mut sim_state: ResMut<SimulationState>,
    mut replay_state: ResMut<ReplayState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
    material_params: Res<MaterialParams>,
) {
    let has_default_world_request = default_world_reader.read().next().is_some();
    let has_reset_request = reset_reader.read().next().is_some();
    if !has_default_world_request && !has_reset_request {
        return;
    }

    if !has_default_world_request && replay_state.enabled {
        if let Some(name) = replay_state.scenario_name.clone() {
            if let Some(spec) = default_scenario_spec_by_name(&name) {
                if let Err(error) = apply_scenario_spec(
                    &spec,
                    &mut terrain_world,
                    &mut particle_world,
                    &mut object_world,
                    &mut object_field,
                ) {
                    replay_state.status_message =
                        format!("Replay reset failed for {}: {error}", spec.name);
                    return;
                }
                replay_state.scenario_total_steps = spec.step_count;
                replay_state.current_step = 0;
                replay_state.baseline_particle_count = particle_world.particle_count();
                replay_state.baseline_solid_cell_count = count_solid_cells(&terrain_world);
                sim_state.running = false;
                sim_state.step_once = false;
                replay_state.status_message = format!("Reloaded replay scenario: {}", spec.name);
                return;
            }
        }
    }

    terrain_world.reset_fixed_world();
    terrain_world.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(*material_params));
    *particle_world = ParticleWorld::default();
    object_world.clear();
    object_field.clear();
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
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
) {
    for request in save_reader.read() {
        match save_load::save_to_slot(
            &request.slot_name,
            &terrain_world,
            &particle_world,
            &object_world,
            &sim_state,
        ) {
            Ok(path) => tracing::info!("saved map to {}", path.display()),
            Err(error) => tracing::error!("failed to save map: {error}"),
        }
    }

    for request in load_reader.read() {
        match save_load::load_from_slot(
            &request.slot_name,
            &mut terrain_world,
            &mut particle_world,
            &mut object_world,
            &mut sim_state,
        ) {
            Ok(path) => {
                object_field.clear();
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
