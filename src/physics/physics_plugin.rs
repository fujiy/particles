use std::time::Instant;

use bevy::log::tracing;
use bevy::prelude::*;

use super::object::{ObjectPhysicsField, ObjectWorld};
use super::particle::{ParticleWorld, TERRAIN_BOUNDARY_RADIUS_M};
use super::save_load;
use super::scenario::{
    apply_scenario_spec, count_solid_cells, default_scenario_spec_by_name,
    write_scenario_artifacts_for_state,
};
use super::state::{
    LoadDefaultWorldRequest, LoadMapRequest, ReplayLoadScenarioRequest, ReplaySaveArtifactRequest,
    ReplayState, ResetSimulationRequest, SaveMapRequest, SimFixedSet, SimUpdateSet,
    SimulationPerfMetrics, SimulationState,
};
use super::terrain::TerrainWorld;

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(FixedUpdate, SimFixedSet::Physics)
            .configure_sets(
                Update,
                (
                    SimUpdateSet::Controls,
                    SimUpdateSet::Interaction,
                    SimUpdateSet::Rendering,
                    SimUpdateSet::Ui,
                    SimUpdateSet::Overlay,
                    SimUpdateSet::Finalize,
                )
                    .chain(),
            )
            .init_resource::<TerrainWorld>()
            .init_resource::<ParticleWorld>()
            .init_resource::<ObjectWorld>()
            .init_resource::<ObjectPhysicsField>()
            .init_resource::<SimulationState>()
            .init_resource::<ReplayState>()
            .init_resource::<SimulationPerfMetrics>()
            .add_message::<ResetSimulationRequest>()
            .add_message::<LoadDefaultWorldRequest>()
            .add_message::<SaveMapRequest>()
            .add_message::<LoadMapRequest>()
            .add_message::<ReplayLoadScenarioRequest>()
            .add_message::<ReplaySaveArtifactRequest>()
            .add_systems(Startup, initialize_default_world)
            .add_systems(
                FixedUpdate,
                step_water_particles.in_set(SimFixedSet::Physics),
            )
            .add_systems(
                Update,
                (
                    handle_sim_controls,
                    apply_sim_reset,
                    handle_replay_requests,
                    apply_save_load_requests,
                )
                    .chain()
                    .in_set(SimUpdateSet::Controls),
            )
            .add_systems(
                Update,
                finalize_frame_metrics.in_set(SimUpdateSet::Finalize),
            );
    }
}

fn initialize_default_world(
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
    mut sim_state: ResMut<SimulationState>,
) {
    terrain_world.reset_fixed_world();
    terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
    *particle_world = ParticleWorld::default();
    object_world.clear();
    object_field.clear();
    sim_state.running = false;
    sim_state.step_once = false;
}

fn step_water_particles(
    mut sim_state: ResMut<SimulationState>,
    mut replay_state: ResMut<ReplayState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
    mut perf_metrics: ResMut<SimulationPerfMetrics>,
) {
    let _step_span = tracing::info_span!("physics::fixed_step").entered();
    let should_step = sim_state.running || sim_state.step_once;
    {
        let _span = tracing::info_span!("physics::object_field_update").entered();
        object_world.update_physics_field(
            particle_world.positions(),
            particle_world.masses(),
            &mut object_field,
        );
    }
    if should_step {
        {
            let _span = tracing::info_span!("physics::terrain_rebuild_if_dirty").entered();
            terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        }
        let start = Instant::now();
        step_simulation_once(
            &mut terrain_world,
            &mut particle_world,
            &mut object_world,
            &mut object_field,
        );
        perf_metrics.physics_time_this_frame_secs += start.elapsed().as_secs_f64();
        if replay_state.enabled {
            replay_state.current_step = replay_state.current_step.saturating_add(1);
        }
    } else {
        let _span = tracing::info_span!("physics::particle_step").entered();
        particle_world.step_if_running(&terrain_world, &object_field, &mut object_world, false);
    }
    sim_state.step_once = false;
}

fn handle_sim_controls(
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

fn handle_replay_requests(
    mut load_reader: MessageReader<ReplayLoadScenarioRequest>,
    mut save_reader: MessageReader<ReplaySaveArtifactRequest>,
    mut replay_state: ResMut<ReplayState>,
    mut sim_state: ResMut<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
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
                step_simulation_once(
                    &mut terrain_world,
                    &mut particle_world,
                    &mut object_world,
                    &mut object_field,
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

fn apply_sim_reset(
    mut reset_reader: MessageReader<ResetSimulationRequest>,
    mut default_world_reader: MessageReader<LoadDefaultWorldRequest>,
    mut sim_state: ResMut<SimulationState>,
    mut replay_state: ResMut<ReplayState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
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
    terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
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

fn apply_save_load_requests(
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

fn finalize_frame_metrics(mut perf_metrics: ResMut<SimulationPerfMetrics>) {
    perf_metrics.physics_time_this_frame_secs = 0.0;
}

fn step_simulation_once(
    terrain_world: &mut TerrainWorld,
    particle_world: &mut ParticleWorld,
    object_world: &mut ObjectWorld,
    object_field: &mut ObjectPhysicsField,
) {
    {
        let _span = tracing::info_span!("physics::particle_step").entered();
        particle_world.step_if_running(terrain_world, object_field, object_world, true);
    }
    {
        let _span = tracing::info_span!("physics::terrain_fracture_commit").entered();
        if particle_world.apply_pending_terrain_fractures(terrain_world, object_world) {
            terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        }
    }
}
