use std::collections::HashSet;
use std::time::Instant;

use bevy::log::tracing;
use bevy::prelude::*;

use super::params_types::SolverParams;
use super::step::step_simulation_once;
use super::terrain_boundary::TerrainBoundarySampler;
use super::types::StepSimulationTiming;
use crate::physics::material::{MaterialParams, terrain_boundary_radius_m};
use crate::physics::profiler::process_cpu_time_seconds;
use crate::physics::state::{
    BlockColoringExperimentState, BlockColoringLayoutBlock, PhysicsActiveRegion,
    PhysicsRegionSettings, PhysicsStepProfileSegment, PhysicsStepProfiler, ReplayState,
    SimulationParallelSettings, SimulationPerfMetrics, SimulationState,
};
use crate::physics::world::constants::{
    CELL_SIZE_M, CHUNK_SIZE_I32, DEFAULT_MPM_BLOCK_NODE_SPAN, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y,
    WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y,
};
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::grid::{GridBlock, GridHierarchy, MpmBlockIndexTable};
use crate::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::{ParticleActivityState, ParticleWorld};
use crate::physics::world::terrain::{TerrainWorld, world_to_cell};

const BLOCK_COLORING_EXPERIMENT_SCENARIO: &str = "block_coloring_experiment";
const BLOCK_COLORING_EXPERIMENT_INTERVAL_SECS: f32 = 1.0;
const BLOCK_COLORING_EXPERIMENT_OPS_MIN: usize = 2;
const BLOCK_COLORING_EXPERIMENT_OPS_MAX: usize = 5;

pub(crate) fn initialize_default_world(
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut continuum_world: ResMut<ContinuumParticleWorld>,
    mut grid_hierarchy: ResMut<GridHierarchy>,
    mut mpm_block_index_table: ResMut<MpmBlockIndexTable>,
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
    reset_mpm_grid_hierarchy(
        &mut grid_hierarchy,
        solver_params.fixed_dt,
        DEFAULT_MPM_BLOCK_NODE_SPAN,
    );
    mpm_block_index_table.clear();
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
    mpm_resources: (ResMut<GridHierarchy>, ResMut<MpmBlockIndexTable>),
    object_resources: (ResMut<ObjectWorld>, ResMut<ObjectPhysicsField>),
    mut terrain_boundary_sampler: ResMut<TerrainBoundarySampler>,
    profiling_resources: (
        ResMut<SimulationPerfMetrics>,
        ResMut<PhysicsStepProfiler>,
        ResMut<BlockColoringExperimentState>,
    ),
    camera_transforms: Query<&Transform, With<Camera2d>>,
) {
    if !sim_state.mpm_enabled {
        // Debug/overlay mode: skip all fixed-step physics bookkeeping.
        // This avoids spending CPU on active-region scanning and object field rebuild
        // while MLS-MPM is globally disabled.
        particle_world.set_active_chunk_region_bounds(None, None);
        particle_world.set_active_halo_chunks(0);
        particle_world.configure_far_field_queue(None, 0, 0, 0, 0.0, 0);
        active_region.active_chunks.clear();
        active_region.chunk_min = None;
        active_region.chunk_max = None;
        sim_state.step_once = false;
        return;
    }

    if sim_state.gpu_mpm_active {
        // GPU MPM path: skip CPU-side fixed-step bookkeeping and solver work.
        // Terrain/object interactions are handled in GPU compute passes.
        particle_world.set_active_chunk_region_bounds(None, None);
        particle_world.set_active_halo_chunks(0);
        particle_world.configure_far_field_queue(None, 0, 0, 0, 0.0, 0);
        active_region.active_chunks.clear();
        active_region.chunk_min = None;
        active_region.chunk_max = None;
        sim_state.step_once = false;
        return;
    }

    let (mut grid_hierarchy, mut mpm_block_index_table) = mpm_resources;
    let (mut object_world, mut object_field) = object_resources;
    let (mut perf_metrics, mut step_profiler, mut block_coloring_experiment) = profiling_resources;
    particle_world.set_solver_params(*solver_params);
    particle_world.set_material_params(*material_params);
    let terrain_boundary_radius_m = terrain_boundary_radius_m(*material_params);
    let _step_span = tracing::info_span!("physics::fixed_step").entered();
    let should_step = sim_state.running || sim_state.step_once;
    if !should_step {
        // Paused: skip active-region scans and object/terrain bookkeeping.
        // They are recomputed on the first stepping frame after resume.
        sim_state.step_once = false;
        return;
    }
    update_block_coloring_experiment(
        &replay_state,
        should_step,
        solver_params.fixed_dt,
        &mut grid_hierarchy,
        &mut mpm_block_index_table,
        &mut block_coloring_experiment,
    );
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
        let sim_step = if !sim_state.mpm_enabled {
            // Overlay/debug mode: disable all MLS-MPM stepping.
            StepSimulationTiming::default()
        } else if sim_state.gpu_mpm_active {
            // GPU compute path owns ContinuumParticleWorld updates; skip CPU MPM.
            StepSimulationTiming::default()
        } else {
            step_simulation_once(
                &mut terrain_world,
                &mut particle_world,
                &mut continuum_world,
                &mut grid_hierarchy,
                &mut mpm_block_index_table,
                &mut object_world,
                &mut object_field,
                &mut terrain_boundary_sampler,
                parallel_settings.enabled,
                terrain_boundary_radius_m,
            )
        };
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
            if !sim_step.mpm_phases.is_empty() {
                // MPM water frame: show per-phase breakdown instead of particle_step::other.
                let mut mpm_accounted_wall_ms = 0.0_f64;
                let mut mpm_accounted_cpu_ms = 0.0_f64;
                for phase in &sim_step.mpm_phases {
                    if phase.wall_secs <= 0.0 {
                        continue;
                    }
                    let wall_ms = phase.wall_secs * 1000.0;
                    let cpu_ms = phase.cpu_secs * 1000.0;
                    mpm_accounted_wall_ms += wall_ms;
                    mpm_accounted_cpu_ms += cpu_ms;
                    step_profiler.segments.push(PhysicsStepProfileSegment {
                        name: phase.name.to_string(),
                        wall_duration_ms: wall_ms,
                        cpu_duration_ms: cpu_ms,
                    });
                }
                let mpm_other_ms =
                    (sim_step.particle_step_secs * 1000.0 - mpm_accounted_wall_ms).max(0.0);
                let mpm_other_cpu_ms =
                    (sim_step.particle_step_cpu_secs * 1000.0 - mpm_accounted_cpu_ms).max(0.0);
                if mpm_other_ms > 0.001 {
                    step_profiler.segments.push(PhysicsStepProfileSegment {
                        name: "mpm::other".to_string(),
                        wall_duration_ms: mpm_other_ms,
                        cpu_duration_ms: mpm_other_cpu_ms,
                    });
                }
            } else {
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

#[derive(Clone, Copy, Debug)]
struct MergeCandidate {
    child_level: u8,
    parent_origin: IVec2,
}

fn parent_origin_from_child_origin(child_origin: IVec2, block_span: IVec2) -> IVec2 {
    let span_x = block_span.x.max(1);
    let span_y = block_span.y.max(1);
    IVec2::new(
        child_origin.x.div_euclid(span_x * 2) * span_x,
        child_origin.y.div_euclid(span_y * 2) * span_y,
    )
}

fn update_block_coloring_experiment(
    replay_state: &ReplayState,
    should_step: bool,
    fixed_dt: f32,
    grid_hierarchy: &mut GridHierarchy,
    mpm_block_index_table: &mut MpmBlockIndexTable,
    experiment_state: &mut BlockColoringExperimentState,
) {
    let is_active = replay_state.enabled
        && replay_state.scenario_name.as_deref() == Some(BLOCK_COLORING_EXPERIMENT_SCENARIO);
    if !is_active {
        reset_block_coloring_experiment_state(experiment_state);
        return;
    }

    if !experiment_state.initialized {
        initialize_block_coloring_experiment_state(experiment_state, grid_hierarchy);
    } else if !block_coloring_layout_matches_grid(experiment_state, grid_hierarchy) {
        initialize_block_coloring_experiment_state(experiment_state, grid_hierarchy);
    }
    if !should_step || !experiment_state.enabled {
        return;
    }

    experiment_state.elapsed_secs += fixed_dt.max(0.0);
    let mut layout_changed = false;
    while experiment_state.elapsed_secs >= BLOCK_COLORING_EXPERIMENT_INTERVAL_SECS {
        experiment_state.elapsed_secs -= BLOCK_COLORING_EXPERIMENT_INTERVAL_SECS;
        layout_changed |= mutate_block_coloring_layout(experiment_state);
    }
    if !layout_changed {
        return;
    }

    rebuild_grid_from_block_coloring_layout(
        grid_hierarchy,
        mpm_block_index_table,
        experiment_state,
        fixed_dt,
    );
}

fn reset_block_coloring_experiment_state(experiment_state: &mut BlockColoringExperimentState) {
    experiment_state.enabled = false;
    experiment_state.initialized = false;
    experiment_state.elapsed_secs = 0.0;
    experiment_state.blocks.clear();
}

fn initialize_block_coloring_experiment_state(
    experiment_state: &mut BlockColoringExperimentState,
    grid_hierarchy: &GridHierarchy,
) {
    let mut blocks = Vec::with_capacity(grid_hierarchy.block_count());
    let mut max_level = 0_u8;
    for block in grid_hierarchy.blocks() {
        blocks.push(BlockColoringLayoutBlock {
            origin_node: block.origin_node,
            level: block.level,
        });
        max_level = max_level.max(block.level);
    }
    blocks.sort_by_key(|block| (block.level, block.origin_node.y, block.origin_node.x));
    blocks.dedup();
    experiment_state.block_cell_dims = grid_hierarchy
        .blocks()
        .first()
        .map(GridBlock::cell_dims)
        .unwrap_or(UVec2::new(16, 16));
    experiment_state.min_level = 0;
    experiment_state.max_level = max_level;
    experiment_state.blocks = blocks;
    experiment_state.elapsed_secs = 0.0;
    experiment_state.enabled = true;
    experiment_state.initialized = true;
}

fn block_coloring_layout_matches_grid(
    experiment_state: &BlockColoringExperimentState,
    grid_hierarchy: &GridHierarchy,
) -> bool {
    if experiment_state.blocks.len() != grid_hierarchy.block_count() {
        return false;
    }
    experiment_state
        .blocks
        .iter()
        .zip(grid_hierarchy.blocks().iter())
        .all(|(layout_block, grid_block)| {
            layout_block.level == grid_block.level
                && layout_block.origin_node == grid_block.origin_node
        })
}

fn mutate_block_coloring_layout(experiment_state: &mut BlockColoringExperimentState) -> bool {
    if experiment_state.blocks.is_empty() {
        return false;
    }
    let op_span = BLOCK_COLORING_EXPERIMENT_OPS_MAX
        .saturating_sub(BLOCK_COLORING_EXPERIMENT_OPS_MIN)
        .saturating_add(1);
    let op_count = BLOCK_COLORING_EXPERIMENT_OPS_MIN
        + (next_random_u64(&mut experiment_state.rng_state) as usize % op_span.max(1));
    let mut changed = false;
    for _ in 0..op_count {
        let split_candidates = collect_split_candidates(experiment_state);
        let merge_candidates = collect_merge_candidates(experiment_state);
        let can_split = !split_candidates.is_empty();
        let can_merge = !merge_candidates.is_empty();
        if !can_split && !can_merge {
            break;
        }

        let pick_split = if can_split && can_merge {
            (next_random_u64(&mut experiment_state.rng_state) & 1) == 0
        } else {
            can_split
        };
        if pick_split {
            changed |= apply_random_split(experiment_state, &split_candidates);
        } else {
            changed |= apply_random_merge(experiment_state, &merge_candidates);
        }
    }
    if changed {
        experiment_state
            .blocks
            .sort_by_key(|block| (block.level, block.origin_node.y, block.origin_node.x));
        experiment_state.blocks.dedup();
    }
    changed
}

fn collect_split_candidates(experiment_state: &BlockColoringExperimentState) -> Vec<usize> {
    experiment_state
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(index, block)| {
            if block.level > experiment_state.min_level {
                Some(index)
            } else {
                None
            }
        })
        .collect()
}

fn collect_merge_candidates(
    experiment_state: &BlockColoringExperimentState,
) -> Vec<MergeCandidate> {
    let span_x = experiment_state.block_cell_dims.x.max(1) as i32;
    let span_y = experiment_state.block_cell_dims.y.max(1) as i32;
    let block_span = IVec2::new(span_x, span_y);
    let block_set: HashSet<(u8, IVec2)> = experiment_state
        .blocks
        .iter()
        .map(|block| (block.level, block.origin_node))
        .collect();
    let mut visited = HashSet::<(u8, IVec2)>::new();
    let mut candidates = Vec::new();
    for block in &experiment_state.blocks {
        if block.level.saturating_add(1) > experiment_state.max_level {
            continue;
        }
        let parent_origin = parent_origin_from_child_origin(block.origin_node, block_span);
        if !visited.insert((block.level, parent_origin)) {
            continue;
        }
        let child_base = parent_origin * 2;
        let child_origins = [
            child_base,
            child_base + IVec2::new(span_x, 0),
            child_base + IVec2::new(0, span_y),
            child_base + IVec2::new(span_x, span_y),
        ];
        if child_origins
            .iter()
            .all(|origin| block_set.contains(&(block.level, *origin)))
        {
            candidates.push(MergeCandidate {
                child_level: block.level,
                parent_origin,
            });
        }
    }
    candidates
}

fn apply_random_split(
    experiment_state: &mut BlockColoringExperimentState,
    split_candidates: &[usize],
) -> bool {
    if split_candidates.is_empty() {
        return false;
    }
    let pick = split_candidates
        [next_random_u64(&mut experiment_state.rng_state) as usize % split_candidates.len()];
    if pick >= experiment_state.blocks.len() {
        return false;
    }
    let block = experiment_state.blocks.swap_remove(pick);
    if block.level <= experiment_state.min_level {
        experiment_state.blocks.push(block);
        return false;
    }

    let span_x = experiment_state.block_cell_dims.x.max(1) as i32;
    let span_y = experiment_state.block_cell_dims.y.max(1) as i32;
    let child_level = block.level.saturating_sub(1);
    let child_base = block.origin_node * 2;
    let child_origins = [
        child_base,
        child_base + IVec2::new(span_x, 0),
        child_base + IVec2::new(0, span_y),
        child_base + IVec2::new(span_x, span_y),
    ];
    for origin in child_origins {
        experiment_state.blocks.push(BlockColoringLayoutBlock {
            origin_node: origin,
            level: child_level,
        });
    }
    true
}

fn apply_random_merge(
    experiment_state: &mut BlockColoringExperimentState,
    merge_candidates: &[MergeCandidate],
) -> bool {
    if merge_candidates.is_empty() {
        return false;
    }
    let candidate = merge_candidates
        [next_random_u64(&mut experiment_state.rng_state) as usize % merge_candidates.len()];
    if candidate.child_level.saturating_add(1) > experiment_state.max_level {
        return false;
    }

    let span_x = experiment_state.block_cell_dims.x.max(1) as i32;
    let span_y = experiment_state.block_cell_dims.y.max(1) as i32;
    let child_base = candidate.parent_origin * 2;
    let child_origins = [
        child_base,
        child_base + IVec2::new(span_x, 0),
        child_base + IVec2::new(0, span_y),
        child_base + IVec2::new(span_x, span_y),
    ];
    experiment_state.blocks.retain(|block| {
        !(block.level == candidate.child_level && child_origins.contains(&block.origin_node))
    });
    experiment_state.blocks.push(BlockColoringLayoutBlock {
        origin_node: candidate.parent_origin,
        level: candidate.child_level.saturating_add(1),
    });
    true
}

fn rebuild_grid_from_block_coloring_layout(
    grid_hierarchy: &mut GridHierarchy,
    mpm_block_index_table: &mut MpmBlockIndexTable,
    experiment_state: &BlockColoringExperimentState,
    fixed_dt: f32,
) {
    let cell_dims = experiment_state.block_cell_dims.max(UVec2::ONE);
    let node_dims = cell_dims + UVec2::ONE;
    let mut blocks = Vec::with_capacity(experiment_state.blocks.len());
    for block in &experiment_state.blocks {
        let h_b = CELL_SIZE_M * (1_u32 << block.level.min(30)) as f32;
        blocks.push(GridBlock::new(
            block.level,
            h_b,
            fixed_dt.max(1e-6),
            block.origin_node,
            node_dims,
        ));
    }
    grid_hierarchy.replace_blocks(blocks);
    mpm_block_index_table.clear();
}

fn next_random_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E_37_79_B9_7F_4A_7C_15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

pub(crate) fn reset_mpm_grid_hierarchy(
    grid_hierarchy: &mut GridHierarchy,
    dt: f32,
    block_node_span: i32,
) {
    grid_hierarchy.clear();
    let span = block_node_span.max(1);
    reset_mpm_grid_hierarchy_with_span(grid_hierarchy, dt, span);
}

pub(crate) fn reset_mpm_grid_hierarchy_for_mode(
    grid_hierarchy: &mut GridHierarchy,
    dt: f32,
    block_node_span: i32,
    force_single_block: bool,
    block_divisions: Option<UVec2>,
    level_map: &[(IVec2, u8, UVec2)],
) {
    // level_map が指定されている場合は最優先で適用する（空間LoDシナリオ用）
    if !level_map.is_empty() {
        reset_mpm_grid_hierarchy_with_level_map(grid_hierarchy, dt, block_node_span, level_map);
        return;
    }
    let (min_cell, total_node_dims) = mpm_grid_bounds_with_padding();
    let total_cell_dims = (total_node_dims - IVec2::ONE).max(IVec2::ONE);
    if force_single_block {
        grid_hierarchy.replace_blocks(vec![GridBlock::new(
            0,
            CELL_SIZE_M,
            dt.max(1e-6),
            min_cell,
            UVec2::new(total_node_dims.x as u32, total_node_dims.y as u32),
        )]);
        return;
    }
    if let Some(divisions) = block_divisions {
        let div_x = divisions.x.max(1) as i32;
        let div_y = divisions.y.max(1) as i32;
        let mut blocks = Vec::with_capacity((div_x * div_y) as usize);
        for by in 0..div_y {
            let y0 = by * total_cell_dims.y / div_y;
            let y1 = (by + 1) * total_cell_dims.y / div_y;
            let block_cell_h = y1 - y0;
            if block_cell_h <= 0 {
                continue;
            }
            for bx in 0..div_x {
                let x0 = bx * total_cell_dims.x / div_x;
                let x1 = (bx + 1) * total_cell_dims.x / div_x;
                let block_cell_w = x1 - x0;
                if block_cell_w <= 0 {
                    continue;
                }
                blocks.push(GridBlock::new(
                    0,
                    CELL_SIZE_M,
                    dt.max(1e-6),
                    min_cell + IVec2::new(x0, y0),
                    UVec2::new((block_cell_w + 1) as u32, (block_cell_h + 1) as u32),
                ));
            }
        }
        grid_hierarchy.replace_blocks(blocks);
        return;
    }
    reset_mpm_grid_hierarchy(grid_hierarchy, dt, block_node_span);
}

fn reset_mpm_grid_hierarchy_with_span(grid_hierarchy: &mut GridHierarchy, dt: f32, span: i32) {
    let (min_cell, total_node_dims) = mpm_grid_bounds_with_padding();
    let total_cell_dims = (total_node_dims - IVec2::ONE).max(IVec2::ONE);
    let mut blocks = Vec::new();
    let mut by = 0;
    while by < total_cell_dims.y {
        let block_cell_h = (total_cell_dims.y - by).min(span);
        let mut bx = 0;
        while bx < total_cell_dims.x {
            let block_cell_w = (total_cell_dims.x - bx).min(span);
            blocks.push(GridBlock::new(
                0,
                CELL_SIZE_M,
                dt.max(1e-6),
                min_cell + IVec2::new(bx, by),
                UVec2::new((block_cell_w + 1) as u32, (block_cell_h + 1) as u32),
            ));
            bx += span;
        }
        by += span;
    }
    grid_hierarchy.replace_blocks(blocks);
}

/// 指定されたlevel mapからMPM gridを構築する（空間LoD対応）。
///
/// `level_entries` は `(block_origin_node, level, block_cell_dims)` のリスト。
/// level L のblockは `h_b = CELL_SIZE_M * 2^L` の空間解像度を持つ。
/// 各blockは `block_cell_dims + 1` のノード数を持ち、境界ノードを隣接blockと共有する。
/// `block_cell_dims` が `UVec2::ZERO` の場合は `block_node_span × block_node_span` を使う。
pub(crate) fn reset_mpm_grid_hierarchy_with_level_map(
    grid_hierarchy: &mut GridHierarchy,
    dt: f32,
    block_node_span: i32,
    level_entries: &[(IVec2, u8, UVec2)],
) {
    let default_span = block_node_span.max(1) as u32;
    let mut blocks = Vec::with_capacity(level_entries.len());
    for &(origin, level, block_cell_dims) in level_entries {
        let h_b = CELL_SIZE_M * (1u32 << level) as f32;
        let cell_dims = if block_cell_dims == UVec2::ZERO {
            UVec2::new(default_span, default_span)
        } else {
            UVec2::new(block_cell_dims.x.max(1), block_cell_dims.y.max(1))
        };
        let node_dims = cell_dims + UVec2::ONE;
        blocks.push(GridBlock::new(level, h_b, dt.max(1e-6), origin, node_dims));
    }
    grid_hierarchy.replace_blocks(blocks);
}

fn mpm_grid_bounds_with_padding() -> (IVec2, IVec2) {
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
    (min_cell, node_dims)
}

#[cfg(test)]
mod tests {
    use super::parent_origin_from_child_origin;
    use bevy::prelude::IVec2;

    #[test]
    fn parent_origin_from_child_origin_snaps_to_block_span_grid() {
        let span = IVec2::new(16, 16);
        assert_eq!(
            parent_origin_from_child_origin(IVec2::new(-128, -96), span),
            IVec2::new(-64, -48)
        );
        assert_eq!(
            parent_origin_from_child_origin(IVec2::new(-112, -80), span),
            IVec2::new(-64, -48)
        );
        assert_eq!(
            parent_origin_from_child_origin(IVec2::new(16, 16), span),
            IVec2::new(0, 0)
        );
    }
}
