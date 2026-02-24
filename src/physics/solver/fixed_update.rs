use std::time::Instant;

use bevy::log::tracing;
use bevy::prelude::*;

use super::params_types::SolverParams;
use super::step::step_simulation_once;
use super::terrain_boundary::TerrainBoundarySampler;
use crate::physics::material::{MaterialParams, terrain_boundary_radius_m};
use crate::physics::profiler::process_cpu_time_seconds;
use crate::physics::state::{
    PhysicsActiveRegion, PhysicsRegionSettings, PhysicsStepProfileSegment, PhysicsStepProfiler,
    ReplayState, SimulationParallelSettings, SimulationPerfMetrics, SimulationState,
};
use crate::physics::world::constants::{
    CELL_SIZE_M, CHUNK_SIZE_I32, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X,
    WORLD_MIN_CHUNK_Y,
};
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::grid::{GridBlock, GridHierarchy};
use crate::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::{ParticleActivityState, ParticleWorld};
use crate::physics::world::terrain::{TerrainWorld, world_to_cell};

pub(crate) fn initialize_default_world(
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut continuum_world: ResMut<ContinuumParticleWorld>,
    mut grid_hierarchy: ResMut<GridHierarchy>,
    mut terrain_boundary_sampler: ResMut<TerrainBoundarySampler>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
    mut sim_state: ResMut<SimulationState>,
    solver_params: Res<SolverParams>,
    material_params: Res<MaterialParams>,
) {
    particle_world.set_solver_params(*solver_params);
    particle_world.set_material_params(*material_params);
    let terrain_boundary_radius_m = terrain_boundary_radius_m(*material_params);
    terrain_world.reset_fixed_world();
    terrain_world.rebuild_static_particles_if_dirty(terrain_boundary_radius_m);
    *particle_world = ParticleWorld::default();
    continuum_world.clear();
    reset_mpm_grid_hierarchy(&mut grid_hierarchy, solver_params.fixed_dt);
    terrain_boundary_sampler.clear();
    object_world.clear();
    object_field.clear();
    sim_state.running = false;
    sim_state.step_once = false;
}

pub(crate) fn step_physics(
    mut sim_state: ResMut<SimulationState>,
    solver_params: Res<SolverParams>,
    material_params: Res<MaterialParams>,
    parallel_settings: Res<SimulationParallelSettings>,
    region_settings: Res<PhysicsRegionSettings>,
    mut active_region: ResMut<PhysicsActiveRegion>,
    mut replay_state: ResMut<ReplayState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut continuum_world: ResMut<ContinuumParticleWorld>,
    mut grid_hierarchy: ResMut<GridHierarchy>,
    object_resources: (ResMut<ObjectWorld>, ResMut<ObjectPhysicsField>),
    mut terrain_boundary_sampler: ResMut<TerrainBoundarySampler>,
    mut perf_metrics: ResMut<SimulationPerfMetrics>,
    mut step_profiler: ResMut<PhysicsStepProfiler>,
    camera_transforms: Query<&Transform, With<Camera2d>>,
) {
    let (mut object_world, mut object_field) = object_resources;
    particle_world.set_solver_params(*solver_params);
    particle_world.set_material_params(*material_params);
    let terrain_boundary_radius_m = terrain_boundary_radius_m(*material_params);
    let _step_span = tracing::info_span!("physics::fixed_step").entered();
    let should_step = sim_state.running || sim_state.step_once;
    if replay_state.enabled || !region_settings.enabled {
        particle_world.set_active_chunk_region_bounds(None, None);
        particle_world.set_active_halo_chunks(0);
        particle_world.configure_far_field_queue(None, 0, 0, 0, 0.0, 0);
        active_region.active_chunks.clear();
        active_region.chunk_min = None;
        active_region.chunk_max = None;
    } else if let Some(camera_transform) = camera_transforms.iter().next() {
        let center_cell = world_to_cell(camera_transform.translation.truncate());
        let center_chunk = IVec2::new(
            center_cell.x.div_euclid(CHUNK_SIZE_I32),
            center_cell.y.div_euclid(CHUNK_SIZE_I32),
        );
        let release_radius = region_settings.active_radius_chunks.max(0);
        let halo_chunks = region_settings.active_halo_chunks.max(0);
        particle_world.set_active_halo_chunks(halo_chunks);
        let live_radius = release_radius + halo_chunks;
        particle_world.promote_particles_in_chunk_radius(center_chunk, live_radius);
        let freeze_radius = (live_radius + region_settings.far_field_freeze_margin_chunks.max(0))
            .max(live_radius + 1);
        particle_world.configure_far_field_queue(
            Some(center_chunk),
            freeze_radius,
            release_radius,
            region_settings.far_field_release_particles_per_frame,
            region_settings.far_field_release_clearance_radius_m,
            region_settings.far_field_release_clearance_max_wait_frames,
        );
        let mut active_chunks = Vec::new();
        let mut min_chunk = IVec2::new(i32::MAX, i32::MAX);
        let mut max_chunk = IVec2::new(i32::MIN, i32::MIN);
        for (&pos, &activity) in particle_world
            .positions()
            .iter()
            .zip(particle_world.activity_states().iter())
        {
            if activity != ParticleActivityState::Active {
                continue;
            }
            let cell = world_to_cell(pos);
            let chunk = IVec2::new(
                cell.x.div_euclid(CHUNK_SIZE_I32),
                cell.y.div_euclid(CHUNK_SIZE_I32),
            );
            if (chunk.x - center_chunk.x).abs() > live_radius
                || (chunk.y - center_chunk.y).abs() > live_radius
            {
                continue;
            }
            active_chunks.push(chunk);
            min_chunk.x = min_chunk.x.min(chunk.x);
            min_chunk.y = min_chunk.y.min(chunk.y);
            max_chunk.x = max_chunk.x.max(chunk.x);
            max_chunk.y = max_chunk.y.max(chunk.y);
        }
        active_chunks.sort_by_key(|chunk| (chunk.y, chunk.x));
        active_chunks.dedup();
        if active_chunks.is_empty() {
            min_chunk = center_chunk;
            max_chunk = center_chunk;
            active_chunks.push(center_chunk);
        }
        let halo_chunks = region_settings.active_halo_chunks.max(0);
        let load_min_chunk = min_chunk - IVec2::splat(halo_chunks);
        let load_max_chunk = max_chunk + IVec2::splat(halo_chunks);
        ensure_chunks_loaded_in_rect(&mut terrain_world, load_min_chunk, load_max_chunk);
        particle_world.set_active_chunk_region_bounds(Some(min_chunk), Some(max_chunk));
        active_region.active_chunks = active_chunks;
        active_region.chunk_min = Some(min_chunk);
        active_region.chunk_max = Some(max_chunk);
    } else {
        particle_world.set_active_chunk_region_bounds(None, None);
        particle_world.set_active_halo_chunks(0);
        particle_world.configure_far_field_queue(None, 0, 0, 0, 0.0, 0);
        active_region.active_chunks.clear();
        active_region.chunk_min = None;
        active_region.chunk_max = None;
    }
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
    let object_update_cpu_secs = (process_cpu_time_seconds().unwrap_or(object_update_cpu_start)
        - object_update_cpu_start)
        .max(0.0);
    particle_world.set_parallel_enabled(parallel_settings.enabled);
    if should_step {
        let terrain_rebuild_start = Instant::now();
        let terrain_rebuild_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
        {
            let _span = tracing::info_span!("physics::terrain_rebuild_if_dirty").entered();
            terrain_world.rebuild_static_particles_if_dirty(terrain_boundary_radius_m);
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
            &mut continuum_world,
            &mut grid_hierarchy,
            &mut object_world,
            &mut object_field,
            &mut terrain_boundary_sampler,
            parallel_settings.enabled,
            terrain_boundary_radius_m,
        );
        let total_secs = start.elapsed().as_secs_f64();
        let total_cpu_secs = (process_cpu_time_seconds().unwrap_or(cpu_start) - cpu_start).max(0.0);
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
        let _ =
            particle_world.step_if_running(&terrain_world, &object_field, &mut object_world, false);
    }
    sim_state.step_once = false;
}

pub(crate) fn finalize_frame_metrics(mut perf_metrics: ResMut<SimulationPerfMetrics>) {
    perf_metrics.physics_time_this_frame_secs = 0.0;
}

fn ensure_chunks_loaded_in_rect(
    terrain_world: &mut TerrainWorld,
    min_chunk: IVec2,
    max_chunk: IVec2,
) {
    for y in min_chunk.y..=max_chunk.y {
        for x in min_chunk.x..=max_chunk.x {
            terrain_world.ensure_chunk_loaded(IVec2::new(x, y));
        }
    }
}

pub(crate) fn reset_mpm_grid_hierarchy(grid_hierarchy: &mut GridHierarchy, dt: f32) {
    grid_hierarchy.clear();

    let padding_cells = 8;
    let min_cell = IVec2::new(
        WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32 - padding_cells,
        WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32 - padding_cells,
    );
    let max_cell_exclusive = IVec2::new(
        (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 + padding_cells,
        (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 + padding_cells,
    );
    let node_dims = (max_cell_exclusive - min_cell + IVec2::ONE).max(IVec2::ONE);
    grid_hierarchy.add_block(GridBlock::new(
        0,
        CELL_SIZE_M,
        dt.max(1e-6),
        min_cell,
        UVec2::new(node_dims.x as u32, node_dims.y as u32),
    ));
}
