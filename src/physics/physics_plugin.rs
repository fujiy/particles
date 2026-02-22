use std::time::Instant;

use bevy::log::tracing;
use bevy::prelude::*;

use super::object::{ObjectPhysicsField, ObjectWorld};
use super::particle::{ParticleStepBreakdown, ParticleWorld, TERRAIN_BOUNDARY_RADIUS_M};
use super::profiler::process_cpu_time_seconds;
use super::save_load;
use super::scenario::{
    apply_scenario_spec, count_solid_cells, default_scenario_spec_by_name,
    write_scenario_artifacts_for_state,
};
use super::state::{
    LoadDefaultWorldRequest, LoadMapRequest, PhysicsStepProfileSegment, PhysicsStepProfiler,
    ReplayLoadScenarioRequest, ReplaySaveArtifactRequest, ReplayState, ResetSimulationRequest,
    SaveMapRequest, SimFixedSet, SimUpdateSet, SimulationParallelSettings, SimulationPerfMetrics,
    SimulationState,
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
            .init_resource::<SimulationParallelSettings>()
            .init_resource::<ReplayState>()
            .init_resource::<SimulationPerfMetrics>()
            .init_resource::<PhysicsStepProfiler>()
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
    parallel_settings: Res<SimulationParallelSettings>,
    mut replay_state: ResMut<ReplayState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
    mut perf_metrics: ResMut<SimulationPerfMetrics>,
    mut step_profiler: ResMut<PhysicsStepProfiler>,
) {
    let _step_span = tracing::info_span!("physics::fixed_step").entered();
    let should_step = sim_state.running || sim_state.step_once;
    let object_update_start = Instant::now();
    let object_update_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
    {
        let _span = tracing::info_span!("physics::object_field_update").entered();
        object_world.update_physics_field(
            particle_world.positions(),
            particle_world.masses(),
            &mut object_field,
        );
    }
    let object_update_secs = object_update_start.elapsed().as_secs_f64();
    let object_update_cpu_secs =
        (process_cpu_time_seconds().unwrap_or(object_update_cpu_start) - object_update_cpu_start)
            .max(0.0);
    particle_world.set_parallel_enabled(parallel_settings.enabled);
    if should_step {
        let terrain_rebuild_start = Instant::now();
        let terrain_rebuild_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
        {
            let _span = tracing::info_span!("physics::terrain_rebuild_if_dirty").entered();
            terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        }
        let terrain_rebuild_secs = terrain_rebuild_start.elapsed().as_secs_f64();
        let terrain_rebuild_cpu_secs = (process_cpu_time_seconds()
            .unwrap_or(terrain_rebuild_cpu_start)
            - terrain_rebuild_cpu_start)
            .max(0.0);
        let start = Instant::now();
        let cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
        let sim_step = step_simulation_once(
            &mut terrain_world,
            &mut particle_world,
            &mut object_world,
            &mut object_field,
            parallel_settings.enabled,
        );
        let total_secs = start.elapsed().as_secs_f64();
        let total_cpu_secs =
            (process_cpu_time_seconds().unwrap_or(cpu_start) - cpu_start).max(0.0);
        perf_metrics.physics_time_this_frame_secs += total_secs;
        if sim_state.running {
            step_profiler.total_duration_ms = total_secs * 1000.0;
            step_profiler.segments = vec![
                PhysicsStepProfileSegment {
                    name: "object_field_update".to_string(),
                    wall_duration_ms: object_update_secs * 1000.0,
                    cpu_duration_ms: object_update_cpu_secs * 1000.0,
                },
                PhysicsStepProfileSegment {
                    name: "terrain_rebuild_if_dirty".to_string(),
                    wall_duration_ms: terrain_rebuild_secs * 1000.0,
                    cpu_duration_ms: terrain_rebuild_cpu_secs * 1000.0,
                },
            ];
            for phase in sim_step.particle_breakdown.phases() {
                if phase.wall_duration_secs <= 0.0 {
                    continue;
                }
                step_profiler.segments.push(PhysicsStepProfileSegment {
                    name: format!("particle_step::{}", phase.name),
                    wall_duration_ms: phase.wall_duration_secs * 1000.0,
                    cpu_duration_ms: phase.cpu_duration_secs * 1000.0,
                });
            }
            let particle_other_ms = ((sim_step.particle_step_secs
                - sim_step.particle_breakdown.total_wall_secs())
            .max(0.0))
                * 1000.0;
            let particle_other_cpu_ms = ((sim_step.particle_step_cpu_secs
                - sim_step.particle_breakdown.total_cpu_secs())
            .max(0.0))
                * 1000.0;
            if particle_other_ms > 0.001 {
                step_profiler.segments.push(PhysicsStepProfileSegment {
                    name: "particle_step::other".to_string(),
                    wall_duration_ms: particle_other_ms,
                    cpu_duration_ms: particle_other_cpu_ms,
                });
            }
            step_profiler.segments.push(PhysicsStepProfileSegment {
                name: "terrain_fracture_commit".to_string(),
                wall_duration_ms: sim_step.terrain_fracture_commit_secs * 1000.0,
                cpu_duration_ms: sim_step.terrain_fracture_commit_cpu_secs * 1000.0,
            });
            let known_wall_ms: f64 = step_profiler
                .segments
                .iter()
                .map(|segment| segment.wall_duration_ms)
                .sum();
            let known_cpu_ms: f64 = step_profiler
                .segments
                .iter()
                .map(|segment| segment.cpu_duration_ms)
                .sum();
            let unaccounted_wall_ms = (total_secs * 1000.0 - known_wall_ms).max(0.0);
            let unaccounted_cpu_ms = (total_cpu_secs * 1000.0 - known_cpu_ms).max(0.0);
            if unaccounted_wall_ms > 0.001 {
                step_profiler.segments.push(PhysicsStepProfileSegment {
                    name: "step_overhead".to_string(),
                    wall_duration_ms: unaccounted_wall_ms,
                    cpu_duration_ms: unaccounted_cpu_ms,
                });
            }
        }
        if replay_state.enabled {
            replay_state.current_step = replay_state.current_step.saturating_add(1);
        }
    } else {
        let _span = tracing::info_span!("physics::particle_step").entered();
        let _ = particle_world.step_if_running(&terrain_world, &object_field, &mut object_world, false);
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
    parallel_settings: Res<SimulationParallelSettings>,
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
    parallel_enabled: bool,
) -> StepSimulationTiming {
    particle_world.set_parallel_enabled(parallel_enabled);
    let particle_step_start = Instant::now();
    let particle_step_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
    let particle_breakdown = {
        let _span = tracing::info_span!("physics::particle_step").entered();
        particle_world.step_if_running(terrain_world, object_field, object_world, true)
    };
    let particle_step_secs = particle_step_start.elapsed().as_secs_f64();
    let particle_step_cpu_secs =
        (process_cpu_time_seconds().unwrap_or(particle_step_cpu_start) - particle_step_cpu_start)
            .max(0.0);
    let terrain_fracture_start = Instant::now();
    let terrain_fracture_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
    {
        let _span = tracing::info_span!("physics::terrain_fracture_commit").entered();
        if particle_world.apply_pending_terrain_fractures(terrain_world, object_world) {
            terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        }
    }
    StepSimulationTiming {
        particle_step_secs,
        particle_step_cpu_secs,
        particle_breakdown,
        terrain_fracture_commit_secs: terrain_fracture_start.elapsed().as_secs_f64(),
        terrain_fracture_commit_cpu_secs: (process_cpu_time_seconds()
            .unwrap_or(terrain_fracture_cpu_start)
            - terrain_fracture_cpu_start)
            .max(0.0),
    }
}

#[derive(Clone, Debug, Default)]
struct StepSimulationTiming {
    particle_step_secs: f64,
    particle_step_cpu_secs: f64,
    particle_breakdown: ParticleStepBreakdown,
    terrain_fracture_commit_secs: f64,
    terrain_fracture_commit_cpu_secs: f64,
}
