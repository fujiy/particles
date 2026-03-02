use std::time::Instant;

use bevy::log::tracing;
use bevy::prelude::*;
use rayon::prelude::*;

use super::terrain_boundary::{TerrainBoundarySample, TerrainBoundarySampler};
use crate::physics::profiler::process_cpu_time_seconds;
use crate::physics::world::continuum::{ContinuumParticleWorld, ContinuumPhase};
use crate::physics::world::grid::{GridBlock, GridHierarchy, MpmBlockIndexTable};
use crate::physics::world::kernel::evaluate_quadratic_bspline_stencil_2d;
use crate::physics::world::particle::{ParticleMaterial, ParticleWorld};
use crate::physics::world::terrain::{CELL_SIZE_M, TerrainWorld};

const GRID_MASS_EPSILON: f32 = 1e-8;
const DET_EPSILON: f32 = 1e-6;
const SUPPORT_HALO_CELLS: f32 = 2.0;

fn node_scale_for_h(h: f32) -> i32 {
    ((h.max(DET_EPSILON) / CELL_SIZE_M).round() as i32).max(1)
}

fn missing_stencil_nodes_for_block(
    grid: &GridHierarchy,
    block_index: usize,
    world_pos: Vec2,
) -> usize {
    let Some(block) = grid.blocks().get(block_index) else {
        return usize::MAX / 4;
    };
    let h = block.h_b.max(DET_EPSILON);
    let scale = node_scale_for_h(h);
    let inv_h = 1.0 / h;
    let stencil = evaluate_quadratic_bspline_stencil_2d(world_pos, inv_h);
    stencil
        .samples()
        .iter()
        .filter(|sample| grid.node_location(sample.node * scale).is_none())
        .count()
}

#[derive(Clone, Copy, Debug)]
pub struct MpmWaterParams {
    pub dt: f32,
    pub gravity: Vec2,
    pub rho0: f32,
    pub bulk_modulus: f32,
    pub viscosity: f32,
    pub j_min: f32,
    pub j_max: f32,
    pub active_mass_threshold: f32,
    pub cfl_limit: f32,
    pub c_max_norm: f32,
    pub c_damping: f32,
    pub f_relaxation: f32,
}

impl Default for MpmWaterParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            gravity: Vec2::new(0.0, -9.81),
            rho0: 1_000.0,
            bulk_modulus: 6.0e4,
            viscosity: 0.0,
            j_min: 0.6,
            j_max: 1.4,
            active_mass_threshold: 1e-6,
            cfl_limit: 1.0,
            c_max_norm: 80.0,
            c_damping: 0.05,
            f_relaxation: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MpmWaterStepMetrics {
    pub particle_mass_sum: f32,
    pub grid_mass_sum: f32,
    pub max_particle_speed_mps: f32,
    pub max_cfl_ratio: f32,
    pub clamped_particle_count: usize,
    pub boundary_penetrating_node_ratio: f32,
    pub boundary_penetrating_particle_ratio: f32,
    pub boundary_momentum_exchange: Vec2,
    pub boundary_query_wall_secs: f64,
    /// Wall-clock time for P2G Phase 1 (mass/momentum transfer).
    pub p2g_mass_momentum_wall_secs: f64,
    /// Process CPU time for P2G Phase 1 (mass/momentum transfer).
    pub p2g_mass_momentum_cpu_secs: f64,
    /// Wall-clock time for P2G Phase 2 (density estimation + pressure/viscosity).
    pub p2g_pressure_wall_secs: f64,
    /// Process CPU time for P2G Phase 2 (density estimation + pressure/viscosity).
    pub p2g_pressure_cpu_secs: f64,
    /// Wall-clock time for terrain boundary node sampling.
    pub terrain_boundary_sample_wall_secs: f64,
    /// Process CPU time for terrain boundary node sampling.
    pub terrain_boundary_sample_cpu_secs: f64,
    /// Wall-clock time for grid node velocity update.
    pub grid_update_wall_secs: f64,
    /// Process CPU time for grid node velocity update.
    pub grid_update_cpu_secs: f64,
    /// Wall-clock time for G2P collect (gather per-particle updates from grid).
    pub g2p_collect_wall_secs: f64,
    /// Process CPU time for G2P collect (gather per-particle updates from grid).
    pub g2p_collect_cpu_secs: f64,
    /// Wall-clock time for G2P sort/apply (sort staged updates and write back to particles).
    pub g2p_sort_apply_wall_secs: f64,
    /// Process CPU time for G2P sort/apply (sort staged updates and write back to particles).
    pub g2p_sort_apply_cpu_secs: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MpmTerrainBoundaryParams {
    pub sdf_velocity_threshold_m: f32,
    pub deep_push_gain_per_s: f32,
    pub deep_push_speed_cap_mps: f32,
    pub tangential_damping: f32,
}

impl Default for MpmTerrainBoundaryParams {
    fn default() -> Self {
        Self {
            sdf_velocity_threshold_m: 0.01,
            deep_push_gain_per_s: 48.0,
            deep_push_speed_cap_mps: 12.0,
            tangential_damping: 0.05,
        }
    }
}

pub fn sound_speed_mps(params: &MpmWaterParams) -> f32 {
    (params.bulk_modulus.max(0.0) / params.rho0.max(DET_EPSILON)).sqrt()
}

pub fn cfl_ratio(params: &MpmWaterParams, h_b: f32, max_particle_speed_mps: f32) -> f32 {
    let h_b = h_b.max(DET_EPSILON);
    let c = sound_speed_mps(params);
    params.dt.max(0.0) * (max_particle_speed_mps.max(0.0) + c) / h_b
}

pub fn cfl_violated(params: &MpmWaterParams, h_b: f32, max_particle_speed_mps: f32) -> bool {
    cfl_ratio(params, h_b, max_particle_speed_mps) > params.cfl_limit
}

pub fn continuum_phase_for_particle(material: ParticleMaterial) -> Option<ContinuumPhase> {
    match material {
        ParticleMaterial::WaterLiquid => Some(ContinuumPhase::Water),
        ParticleMaterial::SoilGranular => Some(ContinuumPhase::GranularSoil),
        ParticleMaterial::SandGranular => Some(ContinuumPhase::GranularSand),
        _ => None,
    }
}

pub fn is_mpm_managed_particle(material: ParticleMaterial) -> bool {
    continuum_phase_for_particle(material).is_some()
}

pub fn rebuild_continuum_from_particle_world(
    particles: &ParticleWorld,
    continuum: &mut ContinuumParticleWorld,
    params: &MpmWaterParams,
) -> usize {
    continuum.clear();
    let inv_rho0 = 1.0 / params.rho0.max(DET_EPSILON);
    for ((&position, &velocity), (&mass, &material)) in particles
        .positions()
        .iter()
        .zip(particles.vel.iter())
        .zip(particles.masses().iter().zip(particles.materials().iter()))
    {
        let Some(phase) = continuum_phase_for_particle(material) else {
            continue;
        };
        continuum.spawn_particle(
            position,
            velocity,
            mass,
            mass.max(0.0) * inv_rho0,
            phase,
            1.0,
        );
    }
    continuum.len()
}

pub fn sync_continuum_to_particle_world(
    particles: &mut ParticleWorld,
    continuum: &ContinuumParticleWorld,
) -> bool {
    let mpm_indices: Vec<usize> = particles
        .materials()
        .iter()
        .enumerate()
        .filter_map(|(i, &material)| is_mpm_managed_particle(material).then_some(i))
        .collect();
    if mpm_indices.len() != continuum.len() {
        return false;
    }
    for (continuum_index, &particle_index) in mpm_indices.iter().enumerate() {
        particles.prev_pos[particle_index] = particles.pos[particle_index];
        particles.pos[particle_index] = continuum.x[continuum_index];
        particles.vel[particle_index] = continuum.v[continuum_index];
    }
    true
}

pub fn refresh_block_index_table(
    particles: &mut ContinuumParticleWorld,
    grid: &mut GridHierarchy,
    index_table: &mut MpmBlockIndexTable,
    owner_block_drift_secs: &[f32],
    gravity: Vec2,
) {
    let block_count = grid.block_count();
    let particle_count = particles.len();
    let prev_block_count = index_table.block_count();
    index_table.ensure_block_count(block_count);
    // blockレイアウトが変化した場合のみ隣接マップを再構築する
    if prev_block_count != block_count
        || (block_count > 0 && index_table.neighbor_block_indices(0).is_empty())
    {
        index_table.rebuild_neighbor_map(grid);
    }
    index_table.set_rebinned_this_step(false);
    index_table.set_moved_particle_count(0);
    if particle_count == 0 || block_count == 0 {
        index_table.clear();
        particles.owner_block_id.clear();
        for block in grid.blocks_mut() {
            block.clear_nodes();
        }
        return;
    }

    let mut needs_rebin = false;
    if particles.owner_block_id.len() != particle_count {
        particles.owner_block_id.resize(particle_count, 0);
        needs_rebin = true;
    }
    let assigned_owner_count: usize = (0..index_table.block_count())
        .map(|block_index| index_table.resident_indices(block_index).len())
        .sum();
    if assigned_owner_count != particle_count {
        needs_rebin = true;
    }

    let mut moved = 0usize;
    for i in 0..particle_count {
        let owner_block = particles
            .owner_block_id
            .get(i)
            .copied()
            .unwrap_or(0)
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(owner_block)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let x_pred = particles.x[i] + particles.v[i] * drift + 0.5 * gravity * drift * drift;
        let mut next_resident = grid
            .block_index_for_position(x_pred)
            .unwrap_or_else(|| particles.owner_block_id[i].min(block_count.saturating_sub(1)));
        let mut best_missing = missing_stencil_nodes_for_block(grid, next_resident, x_pred);
        if best_missing > 0 {
            let mut candidates = Vec::with_capacity(9);
            candidates.push(next_resident);
            for &neighbor in index_table.neighbor_block_indices(next_resident) {
                candidates.push(neighbor);
            }
            candidates.sort_unstable();
            candidates.dedup();
            for candidate in candidates {
                let missing = missing_stencil_nodes_for_block(grid, candidate, x_pred);
                if missing < best_missing {
                    next_resident = candidate;
                    best_missing = missing;
                    if best_missing == 0 {
                        break;
                    }
                }
            }
        }
        if particles.owner_block_id[i] != next_resident {
            particles.owner_block_id[i] = next_resident;
            moved += 1;
            needs_rebin = true;
        }
    }

    for block_index in 0..block_count {
        if let Some(resident_indices) = index_table.resident_indices_mut(block_index) {
            resident_indices.clear();
        }
        if let Some(support_indices) = index_table.support_indices_mut(block_index) {
            support_indices.clear();
        }
        if let Some(outgoing_particles) = index_table.outgoing_particles_mut(block_index) {
            outgoing_particles.clear();
        }
    }
    for (particle_index, &resident) in particles.owner_block_id.iter().enumerate() {
        if let Some(resident_indices) = index_table.resident_indices_mut(resident) {
            resident_indices.push(particle_index);
        }
    }
    if !needs_rebin {
        for block_index in 0..block_count {
            if let Some(resident_indices) = index_table.resident_indices_mut(block_index) {
                resident_indices.sort_unstable();
                resident_indices.dedup();
            }
        }
    }

    rebuild_support_indices(
        particles,
        grid,
        index_table,
        owner_block_drift_secs,
        gravity,
        SUPPORT_HALO_CELLS,
    );

    index_table.set_moved_particle_count(moved);
    index_table.set_rebinned_this_step(needs_rebin);
    clear_inactive_block_nodes(grid, index_table);
}

fn block_world_aabb(block: &GridBlock) -> (Vec2, Vec2) {
    let h = block.h_b.max(DET_EPSILON);
    let min = block.origin_node.as_vec2() * h;
    let max = (block.origin_node + block.cell_dims().as_ivec2()).as_vec2() * h;
    (min, max)
}

fn point_inside_aabb(point: Vec2, min: Vec2, max: Vec2) -> bool {
    point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y
}

fn rebuild_support_indices(
    particles: &ContinuumParticleWorld,
    grid: &GridHierarchy,
    index_table: &mut MpmBlockIndexTable,
    owner_block_drift_secs: &[f32],
    gravity: Vec2,
    halo_cells: f32,
) {
    let block_count = grid.block_count();
    if block_count == 0 {
        return;
    }

    for block_index in 0..block_count {
        if let Some(support) = index_table.support_indices_mut(block_index) {
            support.clear();
        }
    }

    for particle_index in 0..particles.len() {
        let resident = particles
            .owner_block_id
            .get(particle_index)
            .copied()
            .unwrap_or(0)
            .min(block_count.saturating_sub(1))
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(resident)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let x_pred = particles.x[particle_index]
            + particles.v[particle_index] * drift
            + 0.5 * gravity * drift * drift;

        let mut resident_added = false;
        for candidate in 0..block_count {
            let Some(block) = grid.blocks().get(candidate) else {
                continue;
            };
            let (min, max) = block_world_aabb(block);
            let halo = halo_cells.max(0.0) * block.h_b.max(DET_EPSILON);
            let expanded_min = min - Vec2::splat(halo);
            let expanded_max = max + Vec2::splat(halo);
            if !point_inside_aabb(x_pred, expanded_min, expanded_max) {
                continue;
            }
            if let Some(support) = index_table.support_indices_mut(candidate) {
                support.push(particle_index);
            }
            if candidate == resident {
                resident_added = true;
            }
        }
        if !resident_added {
            if let Some(support) = index_table.support_indices_mut(resident) {
                support.push(particle_index);
            }
        }
    }

    for block_index in 0..block_count {
        if let Some(support) = index_table.support_indices_mut(block_index) {
            support.sort_unstable();
            support.dedup();
        }
    }
}

fn enqueue_outgoing_particles_for_blocks(
    particles: &ContinuumParticleWorld,
    grid: &GridHierarchy,
    index_table: &mut MpmBlockIndexTable,
    g2p_block_indices: &[usize],
    inner_halo_cells: f32,
) {
    for &block_index in g2p_block_indices {
        if let Some(queue) = index_table.outgoing_particles_mut(block_index) {
            queue.clear();
        }
        let Some(block) = grid.blocks().get(block_index) else {
            continue;
        };
        let (min, max) = block_world_aabb(block);
        let halo = inner_halo_cells.max(0.0) * block.h_b.max(DET_EPSILON);
        let inner_min = min + Vec2::splat(halo);
        let inner_max = max - Vec2::splat(halo);
        let resident_snapshot = index_table.resident_indices(block_index).to_vec();
        for particle_index in resident_snapshot {
            let Some(&position) = particles.x.get(particle_index) else {
                continue;
            };
            let should_enqueue = if inner_min.x > inner_max.x || inner_min.y > inner_max.y {
                !point_inside_aabb(position, min, max)
            } else {
                !point_inside_aabb(position, inner_min, inner_max)
            };
            if should_enqueue {
                if let Some(queue) = index_table.outgoing_particles_mut(block_index) {
                    queue.push(particle_index);
                }
            }
        }
    }
}

fn apply_outgoing_resident_migrations(
    particles: &mut ContinuumParticleWorld,
    grid: &GridHierarchy,
    index_table: &mut MpmBlockIndexTable,
) {
    let block_count = grid.block_count();
    if block_count == 0 {
        return;
    }
    let mut moved = 0usize;
    for src_block in 0..block_count {
        let mut queue = index_table.outgoing_particles(src_block).to_vec();
        if queue.is_empty() {
            continue;
        }
        queue.sort_unstable();
        queue.dedup();
        for particle_index in queue {
            let Some(&position) = particles.x.get(particle_index) else {
                continue;
            };
            let mut candidate_blocks = Vec::with_capacity(9);
            candidate_blocks.push(src_block);
            candidate_blocks.extend_from_slice(index_table.neighbor_block_indices(src_block));
            candidate_blocks.sort_unstable();
            candidate_blocks.dedup();

            let mut next_resident = None;
            for candidate in candidate_blocks {
                let Some(block) = grid.blocks().get(candidate) else {
                    continue;
                };
                let (min, max) = block_world_aabb(block);
                if point_inside_aabb(position, min, max) {
                    next_resident = Some(candidate);
                    break;
                }
            }
            let next_resident = next_resident
                .or_else(|| grid.block_index_for_position(position))
                .unwrap_or(src_block.min(block_count.saturating_sub(1)));

            let current_resident = particles
                .owner_block_id
                .get(particle_index)
                .copied()
                .unwrap_or(src_block)
                .min(block_count.saturating_sub(1));
            if next_resident == current_resident {
                continue;
            }
            particles.owner_block_id[particle_index] = next_resident;
            if let Some(residents) = index_table.resident_indices_mut(current_resident) {
                if let Some(remove_at) = residents.iter().position(|&i| i == particle_index) {
                    residents.swap_remove(remove_at);
                }
            }
            if let Some(residents) = index_table.resident_indices_mut(next_resident) {
                residents.push(particle_index);
            }
            moved += 1;
        }
    }
    if moved > 0 {
        index_table
            .set_moved_particle_count(index_table.moved_particle_count().saturating_add(moved));
        index_table.set_rebinned_this_step(true);
    }
    for block_index in 0..block_count {
        if let Some(residents) = index_table.resident_indices_mut(block_index) {
            residents.sort_unstable();
            residents.dedup();
        }
        if let Some(queue) = index_table.outgoing_particles_mut(block_index) {
            queue.clear();
        }
    }
}

pub fn estimate_block_max_speed(
    particles: &ContinuumParticleWorld,
    index_table: &MpmBlockIndexTable,
    block_index: usize,
) -> f32 {
    let mut max_speed = index_table
        .resident_indices(block_index)
        .iter()
        .map(|&index| {
            particles
                .v
                .get(index)
                .copied()
                .unwrap_or(Vec2::ZERO)
                .length()
        })
        .fold(0.0, f32::max);
    if max_speed <= 1e-6 {
        max_speed = index_table
            .support_indices(block_index)
            .iter()
            .map(|&index| {
                particles
                    .v
                    .get(index)
                    .copied()
                    .unwrap_or(Vec2::ZERO)
                    .length()
            })
            .fold(0.0, f32::max);
    }
    max_speed
}

pub fn active_blocks_from_index_table(index_table: &MpmBlockIndexTable) -> Vec<usize> {
    let mut active = Vec::with_capacity(index_table.block_count());
    for block_index in 0..index_table.block_count() {
        if !index_table.support_indices(block_index).is_empty() {
            active.push(block_index);
        }
    }
    active
}

fn clear_inactive_block_nodes(grid: &mut GridHierarchy, index_table: &MpmBlockIndexTable) {
    for block_index in 0..grid.block_count() {
        if !index_table.support_indices(block_index).is_empty() {
            continue;
        }
        if let Some(block) = grid.blocks_mut().get_mut(block_index) {
            block.clear_nodes();
        }
    }
}

pub fn step_single_rate(
    particles: &mut ContinuumParticleWorld,
    grid: &mut GridHierarchy,
    params: &MpmWaterParams,
) -> MpmWaterStepMetrics {
    step_single_rate_coupled(
        particles,
        grid,
        None,
        None,
        params,
        &MpmTerrainBoundaryParams::default(),
    )
}

pub fn step_block_set_coupled(
    particles: &mut ContinuumParticleWorld,
    grid: &mut GridHierarchy,
    index_table: &mut MpmBlockIndexTable,
    grid_block_indices: &[usize],
    g2p_block_indices: &[usize],
    owner_block_drift_secs: &[f32],
    terrain: Option<&TerrainWorld>,
    terrain_sampler: Option<&mut TerrainBoundarySampler>,
    params: &MpmWaterParams,
    terrain_boundary_params: &MpmTerrainBoundaryParams,
    parallel_enabled: bool,
) -> MpmWaterStepMetrics {
    let mut terrain_sampler = terrain_sampler;
    let mut metrics = MpmWaterStepMetrics {
        particle_mass_sum: particles.m.iter().copied().sum(),
        ..Default::default()
    };
    if particles.is_empty() {
        for block in grid.blocks_mut() {
            block.clear_nodes();
        }
        return metrics;
    }
    if grid_block_indices.is_empty() || g2p_block_indices.is_empty() {
        return metrics;
    }

    let mut grid_blocks = grid_block_indices.to_vec();
    grid_blocks.sort_unstable();
    grid_blocks.dedup();
    let mut g2p_blocks = g2p_block_indices.to_vec();
    g2p_blocks.sort_unstable();
    g2p_blocks.dedup();
    debug_assert!(
        grid_blocks.windows(2).all(|pair| pair[0] != pair[1]),
        "target block indices must be unique"
    );
    debug_assert!(
        g2p_blocks.windows(2).all(|pair| pair[0] != pair[1]),
        "g2p block indices must be unique"
    );
    debug_assert_resident_layout(particles, index_table, grid.block_count());
    let grid_color_phases = block_indices_grouped_by_color(grid, &grid_blocks);
    let g2p_color_phases = block_indices_grouped_by_color(grid, &g2p_blocks);

    // P2G Phase 1: mass/momentum transfer into each target block's local node array.
    // Store both pointers as usize so rayon closures can capture them (raw pointers are not Sync).
    let grid_ptr = grid as *mut GridHierarchy as usize;
    let (blocks_ptr, blocks_len) = {
        let blocks = unsafe { &mut *(grid_ptr as *mut GridHierarchy) }.blocks_mut();
        (blocks.as_mut_ptr() as usize, blocks.len())
    };
    // 各blockのh_bスライス: support粒子がresident block h_bでカーネルを評価するために使う。
    // ブロックレイアウトはこのステップ中に変わらないため、一度だけ収集する。
    let block_h_b: Vec<f32> = {
        let grid_ref = unsafe { &*(grid_ptr as *const GridHierarchy) };
        grid_ref.blocks().iter().map(|b| b.h_b).collect()
    };
    {
        let _span = tracing::info_span!("physics::mpm::p2g_mass_momentum").entered();
        let t0 = Instant::now();
        let cpu0 = process_cpu_time_seconds().unwrap_or(0.0);
        if parallel_enabled {
            for color_blocks in &grid_color_phases {
                color_blocks.par_iter().for_each(|&block_index| {
                    if block_index >= blocks_len {
                        return;
                    }
                    let support_indices = index_table.support_indices(block_index);
                    // SAFETY: same color blocks do not share edge/vertex, so they never write same owner nodes.
                    let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                    p2g_mass_momentum(
                        particles,
                        support_indices,
                        block_index,
                        block,
                        owner_block_drift_secs,
                        &block_h_b,
                        params,
                    );
                });
            }
        } else {
            for &block_index in &grid_blocks {
                if block_index >= blocks_len {
                    continue;
                }
                let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                p2g_mass_momentum(
                    particles,
                    index_table.support_indices(block_index),
                    block_index,
                    block,
                    owner_block_drift_secs,
                    &block_h_b,
                    params,
                );
            }
        }
        metrics.p2g_mass_momentum_wall_secs += t0.elapsed().as_secs_f64();
        metrics.p2g_mass_momentum_cpu_secs +=
            (process_cpu_time_seconds().unwrap_or(cpu0) - cpu0).max(0.0);
    }

    // P2G Phase 2: density estimation (reads from full GridHierarchy) + pressure/viscosity.
    // All blocks have complete mass/momentum from Phase 1 before this loop starts.
    //
    // SAFETY: p2g_pressure reads node.m via grid.node_by_world() (written in Phase 1,
    // not modified during Phase 2) and writes node.p/v to the LOCAL block's nodes only.
    // The density read loop and the pressure write loop are sequential within each call
    // (no &GridNode and &mut GridNode overlap at the same time). Block indices are
    // deduplicated, so no two concurrent calls write to the same block.
    {
        let _span = tracing::info_span!("physics::mpm::p2g_pressure").entered();
        let t0 = Instant::now();
        let cpu0 = process_cpu_time_seconds().unwrap_or(0.0);
        if parallel_enabled {
            for color_blocks in &grid_color_phases {
                color_blocks.par_iter().for_each(|&block_index| {
                    if block_index >= blocks_len {
                        return;
                    }
                    let support_indices = index_table.support_indices(block_index);
                    let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                    let grid_ref = unsafe { &*(grid_ptr as *const GridHierarchy) };
                    p2g_pressure(
                        particles,
                        grid_ref,
                        support_indices,
                        block,
                        block.dt_b,
                        owner_block_drift_secs,
                        params,
                    );
                });
            }
        } else {
            for &block_index in &grid_blocks {
                if block_index >= blocks_len {
                    continue;
                }
                let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                let grid_ref = unsafe { &*(grid_ptr as *const GridHierarchy) };
                p2g_pressure(
                    particles,
                    grid_ref,
                    index_table.support_indices(block_index),
                    block,
                    block.dt_b,
                    owner_block_drift_secs,
                    params,
                );
            }
        }
        metrics.p2g_pressure_wall_secs += t0.elapsed().as_secs_f64();
        metrics.p2g_pressure_cpu_secs +=
            (process_cpu_time_seconds().unwrap_or(cpu0) - cpu0).max(0.0);
    }

    let mut boundary_samples_by_block = Vec::<Option<Vec<TerrainBoundarySample>>>::new();
    boundary_samples_by_block.resize_with(grid.block_count(), || None);
    {
        let _span = tracing::info_span!("physics::mpm::terrain_boundary").entered();
        let t0 = Instant::now();
        let cpu0 = process_cpu_time_seconds().unwrap_or(0.0);
        if let (Some(terrain), Some(sampler)) = (terrain, terrain_sampler.as_deref_mut()) {
            for &block_index in &grid_blocks {
                let Some(block) = grid.blocks().get(block_index) else {
                    continue;
                };
                let before = sampler.step_stats();
                let samples = sampler.sample_block_nodes(block, terrain);
                let after = sampler.step_stats();
                metrics.boundary_query_wall_secs +=
                    (after.query_wall_secs - before.query_wall_secs).max(0.0);
                if let Some(slot) = boundary_samples_by_block.get_mut(block_index) {
                    *slot = Some(samples);
                }
            }
        }
        metrics.terrain_boundary_sample_wall_secs += t0.elapsed().as_secs_f64();
        metrics.terrain_boundary_sample_cpu_secs +=
            (process_cpu_time_seconds().unwrap_or(cpu0) - cpu0).max(0.0);
    }

    // Grid update phase.
    {
        let _span = tracing::info_span!("physics::mpm::grid_update").entered();
        let t0 = Instant::now();
        let cpu0 = process_cpu_time_seconds().unwrap_or(0.0);
        let blocks = grid.blocks_mut();
        let blocks_ptr = blocks.as_mut_ptr() as usize;
        if parallel_enabled {
            for color_blocks in &grid_color_phases {
                color_blocks.par_iter().for_each(|&block_index| {
                    if block_index >= blocks.len() {
                        return;
                    }
                    let boundary_samples = boundary_samples_by_block
                        .get(block_index)
                        .and_then(|entry| entry.as_deref());
                    // SAFETY: same color blocks do not share boundary nodes and are mutably disjoint.
                    let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                    grid_update(
                        block,
                        block.dt_b,
                        params,
                        boundary_samples,
                        terrain_boundary_params,
                    );
                });
            }
        } else {
            for &block_index in &grid_blocks {
                let boundary_samples = boundary_samples_by_block
                    .get(block_index)
                    .and_then(|entry| entry.as_deref());
                let Some(block) = grid.blocks_mut().get_mut(block_index) else {
                    continue;
                };
                grid_update(
                    block,
                    block.dt_b,
                    params,
                    boundary_samples,
                    terrain_boundary_params,
                );
            }
        }
        metrics.grid_update_wall_secs += t0.elapsed().as_secs_f64();
        metrics.grid_update_cpu_secs +=
            (process_cpu_time_seconds().unwrap_or(cpu0) - cpu0).max(0.0);
    }

    let mut staged_updates = Vec::new();
    {
        let _span = tracing::info_span!("physics::mpm::g2p_collect").entered();
        let t0 = Instant::now();
        let cpu0 = process_cpu_time_seconds().unwrap_or(0.0);
        if parallel_enabled {
            for color_blocks in &g2p_color_phases {
                let mut color_updates = color_blocks
                    .par_iter()
                    .map(|&block_index| {
                        let resident_indices = index_table.resident_indices(block_index);
                        let Some(block) = grid.blocks().get(block_index) else {
                            return Vec::new();
                        };
                        g2p_collect_owner_updates(
                            particles,
                            grid,
                            resident_indices,
                            block.h_b,
                            block.dt_b,
                            params,
                        )
                    })
                    .reduce(Vec::new, |mut lhs, mut rhs| {
                        lhs.append(&mut rhs);
                        lhs
                    });
                staged_updates.append(&mut color_updates);
            }
        } else {
            for &block_index in &g2p_blocks {
                let resident_indices = index_table.resident_indices(block_index);
                let Some(block) = grid.blocks().get(block_index) else {
                    continue;
                };
                staged_updates.extend(g2p_collect_owner_updates(
                    particles,
                    grid,
                    resident_indices,
                    block.h_b,
                    block.dt_b,
                    params,
                ));
            }
        }
        metrics.g2p_collect_wall_secs += t0.elapsed().as_secs_f64();
        metrics.g2p_collect_cpu_secs +=
            (process_cpu_time_seconds().unwrap_or(cpu0) - cpu0).max(0.0);
    }
    {
        let _span = tracing::info_span!("physics::mpm::g2p_sort_apply").entered();
        let t0 = Instant::now();
        let cpu0 = process_cpu_time_seconds().unwrap_or(0.0);
        staged_updates.sort_unstable_by_key(|update| update.particle_index);
        for update in &staged_updates {
            particles.v[update.particle_index] = update.velocity;
            particles.c[update.particle_index] = update.c_matrix;
            particles.x[update.particle_index] = update.position;
            particles.f[update.particle_index] = update.deformation_gradient;
            metrics.max_particle_speed_mps =
                metrics.max_particle_speed_mps.max(update.velocity.length());
            metrics.clamped_particle_count += update.clamped_count;
        }
        metrics.g2p_sort_apply_wall_secs += t0.elapsed().as_secs_f64();
        metrics.g2p_sort_apply_cpu_secs +=
            (process_cpu_time_seconds().unwrap_or(cpu0) - cpu0).max(0.0);
    }
    enqueue_outgoing_particles_for_blocks(
        particles,
        grid,
        index_table,
        &g2p_blocks,
        SUPPORT_HALO_CELLS,
    );
    apply_outgoing_resident_migrations(particles, grid, index_table);

    for &block_index in &grid_blocks {
        let Some(block) = grid.blocks_mut().get_mut(block_index) else {
            continue;
        };
        metrics.grid_mass_sum += block.nodes().iter().map(|node| node.m).sum::<f32>();
        let mut params_for_block = *params;
        params_for_block.dt = block.dt_b;
        let block_speed = estimate_block_max_speed(particles, index_table, block_index);
        metrics.max_cfl_ratio =
            metrics
                .max_cfl_ratio
                .max(cfl_ratio(&params_for_block, block.h_b, block_speed));
        block.rebuild_active_nodes(params.active_mass_threshold.max(0.0));
        let Some(boundary_samples) = boundary_samples_by_block
            .get(block_index)
            .and_then(|entry| entry.as_deref())
        else {
            continue;
        };
        if block.node_count() > 0 {
            let penetrating_nodes = boundary_samples
                .iter()
                .filter(|sample| {
                    sample.solid || sample.sdf_m < terrain_boundary_params.sdf_velocity_threshold_m
                })
                .count();
            metrics.boundary_penetrating_node_ratio += penetrating_nodes as f32;
            if g2p_blocks.binary_search(&block_index).is_ok() {
                metrics.boundary_penetrating_particle_ratio += estimate_penetrating_particle_ratio(
                    particles,
                    index_table.resident_indices(block_index),
                    block,
                    boundary_samples,
                    terrain_boundary_params.sdf_velocity_threshold_m,
                );
            }
        }
        for (node, sample) in block.nodes().iter().zip(boundary_samples.iter()) {
            if !(sample.solid || sample.sdf_m < terrain_boundary_params.sdf_velocity_threshold_m) {
                continue;
            }
            let normal = if sample.normal == Vec2::ZERO {
                Vec2::Y
            } else {
                sample.normal
            };
            let normal_speed = node.v.dot(normal);
            if normal_speed < 0.0 {
                metrics.boundary_momentum_exchange += -normal * normal_speed * node.m;
            }
        }
    }
    let total_nodes: usize = grid_blocks
        .iter()
        .filter_map(|&block_index| grid.blocks().get(block_index))
        .map(GridBlock::node_count)
        .sum();
    if total_nodes > 0 {
        metrics.boundary_penetrating_node_ratio /= total_nodes as f32;
    }
    if !g2p_blocks.is_empty() {
        metrics.boundary_penetrating_particle_ratio /= g2p_blocks.len() as f32;
    }
    metrics
}

fn block_indices_grouped_by_color(
    grid: &GridHierarchy,
    block_indices: &[usize],
) -> Vec<Vec<usize>> {
    let mut keyed = Vec::<(u16, usize)>::with_capacity(block_indices.len());
    for &block_index in block_indices {
        let Some(block) = grid.blocks().get(block_index) else {
            continue;
        };
        keyed.push((block.color_class(), block_index));
    }
    keyed.sort_unstable_by_key(|&(color, block_index)| (color, block_index));
    let mut phases = Vec::<Vec<usize>>::new();
    for (color, block_index) in keyed {
        let needs_new_phase = phases.last().is_none_or(|phase| {
            let first_block = phase.first().copied().unwrap_or(block_index);
            let first_color = grid
                .blocks()
                .get(first_block)
                .map(GridBlock::color_class)
                .unwrap_or(color);
            first_color != color
        });
        if needs_new_phase {
            phases.push(Vec::new());
        }
        if let Some(phase) = phases.last_mut() {
            phase.push(block_index);
        }
    }
    phases
}

pub fn step_single_rate_coupled(
    particles: &mut ContinuumParticleWorld,
    grid: &mut GridHierarchy,
    terrain: Option<&TerrainWorld>,
    terrain_sampler: Option<&mut TerrainBoundarySampler>,
    params: &MpmWaterParams,
    terrain_boundary_params: &MpmTerrainBoundaryParams,
) -> MpmWaterStepMetrics {
    let mut index_table = MpmBlockIndexTable::default();
    let owner_block_drift_secs = vec![0.0; grid.block_count()];
    refresh_block_index_table(
        particles,
        grid,
        &mut index_table,
        &owner_block_drift_secs,
        params.gravity,
    );
    let blocks: Vec<_> = (0..grid.block_count()).collect();
    step_block_set_coupled(
        particles,
        grid,
        &mut index_table,
        &blocks,
        &blocks,
        &owner_block_drift_secs,
        terrain,
        terrain_sampler,
        params,
        terrain_boundary_params,
        false,
    )
}

fn estimate_penetrating_particle_ratio(
    particles: &ContinuumParticleWorld,
    particle_indices: &[usize],
    block: &GridBlock,
    boundary_samples: &[TerrainBoundarySample],
    sdf_velocity_threshold_m: f32,
) -> f32 {
    if particles.is_empty() || particle_indices.is_empty() || boundary_samples.is_empty() {
        return 0.0;
    }
    let width = block.node_dims.x as i32;
    let height = block.node_dims.y as i32;
    if width <= 0 || height <= 0 {
        return 0.0;
    }

    let inv_h = 1.0 / block.h_b.max(DET_EPSILON);
    let mut penetrating = 0usize;
    for &particle_index in particle_indices {
        let Some(&x_p) = particles.x.get(particle_index) else {
            continue;
        };
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        let mut sdf_acc = 0.0f32;
        let mut w_acc = 0.0f32;
        for sample in stencil.samples() {
            let local = sample.node - block.origin_node;
            if local.x < 0 || local.y < 0 || local.x >= width || local.y >= height {
                continue;
            }
            let index = (local.y as usize) * block.node_dims.x as usize + local.x as usize;
            let Some(boundary) = boundary_samples.get(index) else {
                continue;
            };
            sdf_acc += sample.weight * boundary.sdf_m;
            w_acc += sample.weight;
        }
        if w_acc > 1e-6 && (sdf_acc / w_acc) < sdf_velocity_threshold_m {
            penetrating += 1;
        }
    }
    penetrating as f32 / particle_indices.len() as f32
}

/// P2G フェーズ1: 質量・運動量をブロックローカルノードへ転送する。
///
/// `support_indices` は `block + halo` へ寄与しうる粒子集合。各粒子の
/// カーネル評価は常に resident block の `h_b` で行い、書き込み先は
/// target block の owner node に限定する。
fn p2g_mass_momentum(
    particles: &ContinuumParticleWorld,
    support_indices: &[usize],
    block_index: usize,
    block: &mut GridBlock,
    owner_block_drift_secs: &[f32],
    // 各blockのh_b（block index順）: support粒子がresident block h_bでカーネルを評価するために使う
    block_h_b: &[f32],
    params: &MpmWaterParams,
) {
    block.clear_nodes();
    let h = block.h_b.max(DET_EPSILON);
    for &i in support_indices {
        let Some((&x_base, &v_base)) = particles.x.get(i).zip(particles.v.get(i)) else {
            continue;
        };
        let Some(&c_p) = particles.c.get(i) else {
            continue;
        };
        let Some(&m_p_raw) = particles.m.get(i) else {
            continue;
        };
        let resident_block = particles
            .owner_block_id
            .get(i)
            .copied()
            .unwrap_or(0)
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(resident_block)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let x_p = x_base + v_base * drift + 0.5 * params.gravity * drift * drift;
        let v_p = v_base + params.gravity * drift;
        let m_p = m_p_raw.max(0.0);
        let h_resident = block_h_b
            .get(resident_block)
            .copied()
            .unwrap_or(h)
            .max(DET_EPSILON);
        let resident_scale = node_scale_for_h(h_resident);
        let inv_h_resident = 1.0 / h_resident;
        if resident_block == block_index {
            let nearest_node = IVec2::new(
                (x_p.x * inv_h_resident).round() as i32,
                (x_p.y * inv_h_resident).round() as i32,
            );
            let nearest_key = nearest_node * resident_scale;
            if block.is_world_key_owned(nearest_key) {
                if let Some(node) = block.node_mut_by_world_key(nearest_key) {
                    node.render_mass_sum += m_p;
                    node.render_mass_pos_sum += m_p * x_p;
                }
            }
        }
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h_resident);
        for sample in stencil.samples() {
            let world_key = sample.node * resident_scale;
            if !block.is_world_key_owned(world_key) {
                continue;
            }
            let node_world = sample.node.as_vec2() * h_resident;
            let affine_velocity = c_p * (node_world - x_p);
            let Some(node) = block.node_mut_by_world_key(world_key) else {
                continue;
            };
            let mass_contrib = sample.weight * m_p;
            node.m += mass_contrib;
            node.p += mass_contrib * (v_p + affine_velocity);
        }
    }
}

/// P2G フェーズ2: 密度推定と圧力・粘性力をノードへ転送する。
///
/// **全ブロックで `p2g_mass_momentum` が完了した後に呼ぶこと。**
///
/// 密度推定には `grid`（全 `GridHierarchy`）を参照するため、ブロック境界を
/// またがる粒子の stencil ノードの質量も漏れなく取得できる。これにより
/// 境界付近での密度推定の非対称が解消し、境界通過時の誤った加速が抑制される。
///
/// 圧力力はローカルブロックのノードにのみ書き込む（外部ノードは隣接ブロックの
/// フェーズ2で処理される）ため、並列実行の安全性は維持される。
fn p2g_pressure(
    particles: &ContinuumParticleWorld,
    grid: &GridHierarchy,
    support_indices: &[usize],
    block: &mut GridBlock,
    dt: f32,
    owner_block_drift_secs: &[f32],
    params: &MpmWaterParams,
) {
    let dt = dt.max(0.0);
    let h = block.h_b.max(DET_EPSILON);

    // Pass 2: 全グリッド参照で完全な密度を推定する。
    // `block.node_by_world` ではなく `grid.node_by_world` を使うことで、
    // ブロック境界をまたぐ stencil ノードの質量も考慮される。
    let mut density = vec![0.0f32; support_indices.len()];
    for (local, &particle_index) in support_indices.iter().enumerate() {
        let Some((&x_base, &v_base)) = particles
            .x
            .get(particle_index)
            .zip(particles.v.get(particle_index))
        else {
            continue;
        };
        let resident_block = particles
            .owner_block_id
            .get(particle_index)
            .copied()
            .unwrap_or(0)
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(resident_block)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let h_p = grid
            .blocks()
            .get(resident_block)
            .map(|b| b.h_b)
            .unwrap_or(h)
            .max(DET_EPSILON);
        let node_scale_p = node_scale_for_h(h_p);
        let inv_h_p = 1.0 / h_p;
        let inv_cell_area_p = inv_h_p * inv_h_p;
        let x_p = x_base + v_base * drift + 0.5 * params.gravity * drift * drift;
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h_p);
        let mut rho_acc = 0.0f32;
        for sample in stencil.samples() {
            let world_key = sample.node * node_scale_p;
            // 全グリッド参照: 隣接ブロックのノードも参照可能
            let Some(node) = grid.node_by_world(world_key) else {
                continue;
            };
            rho_acc += sample.weight * node.m * inv_cell_area_p;
        }
        density[local] = rho_acc.max(0.0);
    }

    // Pass 3: 圧力・粘性力をローカルブロックのノードへ転送する。
    for (local, &rho_p) in density.iter().enumerate() {
        let particle_index = support_indices[local];
        let Some((&x_base, &v_base)) = particles
            .x
            .get(particle_index)
            .zip(particles.v.get(particle_index))
        else {
            continue;
        };
        let resident_block = particles
            .owner_block_id
            .get(particle_index)
            .copied()
            .unwrap_or(0)
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(resident_block)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let h_p = grid
            .blocks()
            .get(resident_block)
            .map(|b| b.h_b)
            .unwrap_or(h)
            .max(DET_EPSILON);
        let node_scale_p = node_scale_for_h(h_p);
        let inv_h_p = 1.0 / h_p;
        let x_p = x_base + v_base * drift + 0.5 * params.gravity * drift * drift;
        let v0_p = particles
            .v0
            .get(particle_index)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let compression = (rho_p / params.rho0.max(DET_EPSILON) - 1.0).max(0.0);
        let pressure = params.bulk_modulus * compression;
        let viscous = Mat2::ZERO;
        let stress =
            Mat2::from_cols(Vec2::new(-pressure, 0.0), Vec2::new(0.0, -pressure)) + viscous;

        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h_p);
        for sample in stencil.samples() {
            let world_key = sample.node * node_scale_p;
            if !block.is_world_key_owned(world_key) {
                continue;
            }
            // 圧力力はローカルブロックのownerノードにのみ書き込む
            let Some(node) = block.node_mut_by_world_key(world_key) else {
                continue;
            };
            let internal_force = -(stress * sample.grad) * v0_p;
            node.p += dt * internal_force;
        }
    }
}

fn grid_update(
    block: &mut GridBlock,
    dt: f32,
    params: &MpmWaterParams,
    boundary_samples: Option<&[TerrainBoundarySample]>,
    boundary_params: &MpmTerrainBoundaryParams,
) {
    let dt = dt.max(0.0);
    for (node_index, node) in block.nodes_mut().iter_mut().enumerate() {
        if node.m <= GRID_MASS_EPSILON {
            node.m = 0.0;
            node.p = Vec2::ZERO;
            node.v = Vec2::ZERO;
            continue;
        }
        let inv_m = 1.0 / node.m;
        node.v = node.p * inv_m;
        node.v += dt * params.gravity;
        if let Some(samples) = boundary_samples {
            if let Some(sample) = samples.get(node_index).copied() {
                if sample.solid || sample.sdf_m < boundary_params.sdf_velocity_threshold_m {
                    let normal = if sample.normal == Vec2::ZERO {
                        Vec2::Y
                    } else {
                        sample.normal
                    };
                    let mut corrected_v = node.v;
                    let normal_speed = corrected_v.dot(normal);
                    if normal_speed < 0.0 {
                        corrected_v -= normal * normal_speed;
                    }
                    if sample.sdf_m < boundary_params.sdf_velocity_threshold_m {
                        let depth = boundary_params.sdf_velocity_threshold_m - sample.sdf_m;
                        let push_gain = boundary_params.deep_push_gain_per_s.max(0.0);
                        let push_speed_cap = boundary_params.deep_push_speed_cap_mps.max(0.0);
                        let mut push_velocity = normal * (depth * push_gain);
                        let push_speed = push_velocity.length();
                        if push_speed_cap > 0.0 && push_speed > push_speed_cap {
                            push_velocity *= push_speed_cap / push_speed;
                        }
                        corrected_v += push_velocity;
                    }
                    let tangent = corrected_v - normal * corrected_v.dot(normal);
                    let next_v =
                        corrected_v - tangent * boundary_params.tangential_damping.clamp(0.0, 1.0);
                    node.v = next_v;
                }
            }
        }
        node.p = node.v * node.m;
    }
}

#[derive(Clone, Copy, Debug)]
struct G2pParticleUpdate {
    particle_index: usize,
    velocity: Vec2,
    c_matrix: Mat2,
    position: Vec2,
    deformation_gradient: Mat2,
    clamped_count: usize,
}

fn g2p_collect_owner_updates(
    particles: &ContinuumParticleWorld,
    grid: &GridHierarchy,
    owner_indices: &[usize],
    h: f32,
    dt: f32,
    params: &MpmWaterParams,
) -> Vec<G2pParticleUpdate> {
    let dt = dt.max(0.0);
    let h = h.max(DET_EPSILON);
    let node_scale = node_scale_for_h(h);
    let inv_h = 1.0 / h;
    let mut updates = Vec::with_capacity(owner_indices.len());
    for &i in owner_indices {
        let Some(&x_p) = particles.x.get(i) else {
            continue;
        };
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        let mut next_v = Vec2::ZERO;
        let mut next_grad_v = Mat2::ZERO;

        for sample in stencil.samples() {
            let world_key = sample.node * node_scale;
            let Some(node) = grid.node_by_world(world_key) else {
                continue;
            };
            if node.m <= GRID_MASS_EPSILON {
                continue;
            }

            next_v += sample.weight * node.v;
            next_grad_v += outer_product(node.v, sample.grad);
        }
        let next_c = clamp_mat2_by_frobenius(next_grad_v, params.c_max_norm.max(1e-3))
            * (1.0 - params.c_damping.clamp(0.0, 1.0));
        let next_x = x_p + dt * next_v;
        let mut clamped_count = 0usize;
        let prev_f = particles.f.get(i).copied().unwrap_or(Mat2::IDENTITY);
        let f_limited = clamp_f_determinant(
            (Mat2::IDENTITY + next_c * dt) * prev_f,
            params.j_min,
            params.j_max,
            &mut clamped_count,
        );
        let next_f = mix_mat2(
            f_limited,
            Mat2::IDENTITY,
            params.f_relaxation.clamp(0.0, 1.0),
        );
        updates.push(G2pParticleUpdate {
            particle_index: i,
            velocity: next_v,
            c_matrix: next_c,
            position: next_x,
            deformation_gradient: next_f,
            clamped_count,
        });
    }
    updates
}

fn debug_assert_resident_layout(
    particles: &ContinuumParticleWorld,
    index_table: &MpmBlockIndexTable,
    block_count: usize,
) {
    #[cfg(debug_assertions)]
    {
        let mut resident_count = vec![0u8; particles.len()];
        for block_index in 0..block_count {
            for &particle_index in index_table.resident_indices(block_index) {
                if particle_index >= particles.len() {
                    panic!(
                        "resident index out of range: block {} -> particle {}",
                        block_index, particle_index
                    );
                }
                if particles.owner_block_id[particle_index] != block_index {
                    panic!(
                        "resident mismatch: particle {} says {}, table says {}",
                        particle_index, particles.owner_block_id[particle_index], block_index
                    );
                }
                resident_count[particle_index] = resident_count[particle_index].saturating_add(1);
            }
        }
        for (particle_index, &count) in resident_count.iter().enumerate() {
            assert_eq!(
                count, 1,
                "resident uniqueness violated for particle {}: count {}",
                particle_index, count
            );
        }
    }
}

fn clamp_f_determinant(f: Mat2, j_min: f32, j_max: f32, clamped_counter: &mut usize) -> Mat2 {
    let det = f.determinant();
    let target = det.clamp(j_min, j_max).max(DET_EPSILON);
    if (target - det).abs() <= 1e-6 {
        return f;
    }
    *clamped_counter += 1;
    let scale = (target / det.max(DET_EPSILON)).sqrt();
    f * scale
}

fn outer_product(lhs: Vec2, rhs: Vec2) -> Mat2 {
    Mat2::from_cols(lhs * rhs.x, lhs * rhs.y)
}

fn clamp_mat2_by_frobenius(matrix: Mat2, max_norm: f32) -> Mat2 {
    let max_norm = max_norm.max(1e-6);
    let norm = matrix
        .to_cols_array()
        .into_iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    if norm <= max_norm {
        matrix
    } else {
        matrix * (max_norm / norm)
    }
}

fn mix_mat2(a: Mat2, b: Mat2, t: f32) -> Mat2 {
    let t = t.clamp(0.0, 1.0);
    let a_cols = a.to_cols_array();
    let b_cols = b.to_cols_array();
    Mat2::from_cols_array(&[
        a_cols[0] + (b_cols[0] - a_cols[0]) * t,
        a_cols[1] + (b_cols[1] - a_cols[1]) * t,
        a_cols[2] + (b_cols[2] - a_cols[2]) * t,
        a_cols[3] + (b_cols[3] - a_cols[3]) * t,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::material::{
        DEFAULT_MATERIAL_PARAMS, particle_radius_m, terrain_boundary_radius_m,
    };
    use crate::physics::solver::terrain_boundary::TerrainBoundarySampler;
    use crate::physics::world::constants::CELL_SIZE_M;
    use crate::physics::world::continuum::ContinuumPhase;
    use crate::physics::world::grid::{GridBlock, MpmBlockIndexTable};
    use crate::physics::world::particle::ParticleWorld;
    use crate::physics::world::terrain::{TerrainCell, TerrainWorld, cell_to_world_center};

    fn make_world_with_single_block() -> (ContinuumParticleWorld, GridHierarchy) {
        let mut particles = ContinuumParticleWorld::default();
        let mut grid = GridHierarchy::default();
        grid.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(-24, -24),
            UVec2::new(48, 48),
        ));
        particles.spawn_particle(
            Vec2::new(0.25, 0.25),
            Vec2::ZERO,
            1.0,
            1.0 / 1_000.0,
            ContinuumPhase::Water,
            1.0,
        );
        particles.spawn_particle(
            Vec2::new(0.45, 0.25),
            Vec2::ZERO,
            1.0,
            1.0 / 1_000.0,
            ContinuumPhase::Water,
            1.0,
        );
        (particles, grid)
    }

    fn make_world_with_two_blocks() -> (ContinuumParticleWorld, GridHierarchy) {
        let mut particles = ContinuumParticleWorld::default();
        let mut grid = GridHierarchy::default();
        grid.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 120.0,
            IVec2::new(-8, -8),
            UVec2::new(8, 16),
        ));
        grid.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 120.0,
            IVec2::new(0, -8),
            UVec2::new(8, 16),
        ));
        particles.spawn_particle(
            Vec2::new(-0.05, 0.25),
            Vec2::new(0.6, 0.0),
            1.0,
            1.0 / 1_000.0,
            ContinuumPhase::Water,
            1.0,
        );
        particles.spawn_particle(
            Vec2::new(0.05, 0.25),
            Vec2::new(-0.2, 0.0),
            1.0,
            1.0 / 1_000.0,
            ContinuumPhase::Water,
            1.0,
        );
        particles.spawn_particle(
            Vec2::new(0.45, 0.35),
            Vec2::new(0.1, -0.2),
            1.0,
            1.0 / 1_000.0,
            ContinuumPhase::Water,
            1.0,
        );
        (particles, grid)
    }

    fn make_world_with_coarse_fine_boundary() -> (ContinuumParticleWorld, GridHierarchy) {
        let mut particles = ContinuumParticleWorld::default();
        let mut grid = GridHierarchy::default();
        // Fine block on the left: x in [-4.0, 0.0].
        grid.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 120.0,
            IVec2::new(-16, -16),
            UVec2::new(17, 17),
        ));
        // Coarse block on the right: x in [0.0, 8.0].
        grid.add_block(GridBlock::new(
            1,
            0.5,
            1.0 / 120.0,
            IVec2::new(0, -8),
            UVec2::new(17, 17),
        ));
        particles.spawn_particle(
            Vec2::new(1.0, -0.25),
            Vec2::ZERO,
            1.0,
            1.0 / 1_000.0,
            ContinuumPhase::Water,
            1.0,
        );
        (particles, grid)
    }

    fn make_basin_contact_setup() -> (TerrainWorld, ContinuumParticleWorld, GridHierarchy) {
        let mut terrain = TerrainWorld::default();
        terrain.clear();
        terrain.ensure_chunk_loaded(IVec2::ZERO);
        terrain.clear_loaded_cells();

        let floor_y = 0;
        let left_wall_x = 1;
        let right_wall_x = 14;
        let wall_top_y = 15;
        for x in left_wall_x..=right_wall_x {
            terrain.set_cell(IVec2::new(x, floor_y), TerrainCell::stone());
        }
        for y in floor_y..=wall_top_y {
            terrain.set_cell(IVec2::new(left_wall_x, y), TerrainCell::stone());
            terrain.set_cell(IVec2::new(right_wall_x, y), TerrainCell::stone());
        }
        terrain
            .rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));

        let mut particles = ContinuumParticleWorld::default();
        let offset = 0.25 * CELL_SIZE_M;
        let particle_mass = 1_000.0 * offset * offset;
        let rest_volume = particle_mass / 1_000.0;
        for y in 6..=11 {
            for x in 3..=12 {
                let center = cell_to_world_center(IVec2::new(x, y));
                for oy in [-offset, offset] {
                    for ox in [-offset, offset] {
                        let vx = if x >= 9 { 1.0 } else { 0.0 };
                        particles.spawn_particle(
                            center + Vec2::new(ox, oy),
                            Vec2::new(vx, 0.0),
                            particle_mass,
                            rest_volume,
                            ContinuumPhase::Water,
                            1.0,
                        );
                    }
                }
            }
        }

        let mut grid = GridHierarchy::default();
        grid.add_block(GridBlock::new(
            0,
            CELL_SIZE_M,
            1.0 / 120.0,
            IVec2::new(-4, -4),
            UVec2::new(32, 32),
        ));
        (terrain, particles, grid)
    }

    fn deep_penetration_ratio(
        terrain: &TerrainWorld,
        particles: &ContinuumParticleWorld,
        deep_sdf_limit_m: f32,
    ) -> f32 {
        if particles.is_empty() {
            return 0.0;
        }
        let mut deep_count = 0usize;
        for &position in &particles.x {
            if let Some((sdf_m, _)) = terrain.sample_signed_distance_and_normal(position) {
                if sdf_m < deep_sdf_limit_m {
                    deep_count += 1;
                }
            }
        }
        deep_count as f32 / particles.len() as f32
    }

    #[test]
    fn p2g_conserves_mass_within_single_block() {
        let (mut particles, mut grid) = make_world_with_single_block();
        let params = MpmWaterParams {
            gravity: Vec2::ZERO,
            ..Default::default()
        };
        let metrics = step_single_rate(&mut particles, &mut grid, &params);
        assert!((metrics.particle_mass_sum - 2.0).abs() < 1e-6);
        assert!((metrics.grid_mass_sum - 2.0).abs() < 1e-4);
    }

    #[test]
    fn static_water_stays_nearly_static_without_external_force() {
        let (mut particles, mut grid) = make_world_with_single_block();
        let params = MpmWaterParams {
            gravity: Vec2::ZERO,
            viscosity: 0.0,
            ..Default::default()
        };
        for _ in 0..20 {
            step_single_rate(&mut particles, &mut grid, &params);
        }
        for &v in &particles.v {
            assert!(v.length() < 1e-4);
        }
    }

    #[test]
    fn g2p_applies_gravity_and_updates_particle_state() {
        let (mut particles, mut grid) = make_world_with_single_block();
        let params = MpmWaterParams {
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        };
        let y_before = particles.x[0].y;
        let metrics = step_single_rate(&mut particles, &mut grid, &params);
        assert!(particles.x[0].y < y_before);
        assert!(metrics.max_particle_speed_mps > 0.0);
    }

    #[test]
    fn deformation_gradient_is_clamped_by_j_bounds() {
        let (mut particles, mut grid) = make_world_with_single_block();
        particles.f[0] = Mat2::from_cols(Vec2::new(8.0, 0.0), Vec2::new(0.0, 8.0));
        let params = MpmWaterParams {
            gravity: Vec2::ZERO,
            j_min: 0.8,
            j_max: 1.2,
            ..Default::default()
        };
        let metrics = step_single_rate(&mut particles, &mut grid, &params);
        let j = particles.f[0].determinant();
        assert!(j >= 0.8 - 1e-4 && j <= 1.2 + 1e-4);
        assert!(metrics.clamped_particle_count >= 1);
    }

    #[test]
    fn cfl_detection_flags_large_dt_or_velocity() {
        let params = MpmWaterParams {
            dt: 0.05,
            bulk_modulus: 2.0e6,
            rho0: 1_000.0,
            cfl_limit: 1.0,
            ..Default::default()
        };
        let ratio = cfl_ratio(&params, 0.25, 10.0);
        assert!(ratio > 1.0);
        assert!(cfl_violated(&params, 0.25, 10.0));
    }

    #[test]
    fn rebuild_continuum_filters_to_mpm_managed_particles_only() {
        let mut particles = ParticleWorld::default();
        particles
            .restore_from_snapshot(
                vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(2.0, 0.0),
                    Vec2::new(3.0, 0.0),
                    Vec2::new(4.0, 0.0),
                ],
                vec![Vec2::ZERO; 5],
                vec![
                    ParticleMaterial::WaterLiquid,
                    ParticleMaterial::StoneSolid,
                    ParticleMaterial::SoilGranular,
                    ParticleMaterial::SandGranular,
                    ParticleMaterial::StoneGranular,
                ],
            )
            .unwrap();
        let mut continuum = ContinuumParticleWorld::default();
        let count = rebuild_continuum_from_particle_world(
            &particles,
            &mut continuum,
            &MpmWaterParams::default(),
        );
        assert_eq!(count, 3);
        assert_eq!(continuum.len(), 3);
        assert_eq!(continuum.x[0], Vec2::new(0.0, 0.0));
        assert_eq!(continuum.x[1], Vec2::new(2.0, 0.0));
        assert_eq!(continuum.x[2], Vec2::new(3.0, 0.0));
    }

    #[test]
    fn sync_continuum_updates_only_mpm_managed_particles() {
        let mut particles = ParticleWorld::default();
        particles
            .restore_from_snapshot(
                vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(2.0, 0.0),
                    Vec2::new(3.0, 0.0),
                    Vec2::new(4.0, 0.0),
                ],
                vec![Vec2::ZERO; 5],
                vec![
                    ParticleMaterial::WaterLiquid,
                    ParticleMaterial::StoneSolid,
                    ParticleMaterial::SoilGranular,
                    ParticleMaterial::SandGranular,
                    ParticleMaterial::StoneGranular,
                ],
            )
            .unwrap();

        let mut continuum = ContinuumParticleWorld::default();
        continuum.spawn_water_particle(Vec2::new(5.0, 6.0), Vec2::new(1.0, 2.0), 1.0, 0.001);
        continuum.spawn_granular_soil_particle(
            Vec2::new(7.0, 8.0),
            Vec2::new(3.0, 4.0),
            1.0,
            0.001,
            1.0,
        );
        continuum.spawn_granular_sand_particle(
            Vec2::new(9.0, 10.0),
            Vec2::new(5.0, 6.0),
            1.0,
            0.001,
            1.0,
        );

        assert!(sync_continuum_to_particle_world(&mut particles, &continuum));
        assert_eq!(particles.pos[0], Vec2::new(5.0, 6.0));
        assert_eq!(particles.vel[0], Vec2::new(1.0, 2.0));
        assert_eq!(particles.pos[1], Vec2::new(1.0, 0.0));
        assert_eq!(particles.pos[2], Vec2::new(7.0, 8.0));
        assert_eq!(particles.vel[2], Vec2::new(3.0, 4.0));
        assert_eq!(particles.pos[3], Vec2::new(9.0, 10.0));
        assert_eq!(particles.vel[3], Vec2::new(5.0, 6.0));
        assert_eq!(particles.pos[4], Vec2::new(4.0, 0.0));
    }

    #[test]
    fn grid_boundary_coupling_applies_non_penetration_on_mpm_nodes() {
        let mut terrain = TerrainWorld::default();
        terrain.clear();
        terrain.ensure_chunk_loaded(IVec2::ZERO);
        terrain.clear_loaded_cells();
        terrain.set_cell(IVec2::ZERO, TerrainCell::stone());
        terrain
            .rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));

        let (mut particles, mut grid) = make_world_with_single_block();
        particles.x[0] = cell_to_world_center(IVec2::ZERO);
        particles.v[0] = Vec2::new(0.0, -2.0);

        let boundary_params = MpmTerrainBoundaryParams::default();
        let mut sampler = TerrainBoundarySampler::default();
        sampler.begin_step();
        let metrics = step_single_rate_coupled(
            &mut particles,
            &mut grid,
            Some(&terrain),
            Some(&mut sampler),
            &MpmWaterParams {
                gravity: Vec2::ZERO,
                ..Default::default()
            },
            &boundary_params,
        );
        sampler.end_step();
        assert!(metrics.boundary_penetrating_node_ratio > 0.0);
        let block = &grid.blocks()[0];
        let boundary_samples = sampler.sample_block_nodes(block, &terrain);
        for (node, sample) in block.nodes().iter().zip(boundary_samples.iter()) {
            if !(sample.solid || sample.sdf_m < boundary_params.sdf_velocity_threshold_m) {
                continue;
            }
            let normal = if sample.normal == Vec2::ZERO {
                Vec2::Y
            } else {
                sample.normal
            };
            assert!(
                node.v.dot(normal) >= -1e-4,
                "boundary node still has inward normal velocity: {:?}",
                node.v
            );
        }
    }

    #[test]
    fn grid_only_boundary_policy_prevents_deep_penetration_accumulation() {
        let (terrain, mut particles, mut grid) = make_basin_contact_setup();
        let mut sampler = TerrainBoundarySampler::default();
        let params = MpmWaterParams {
            dt: 1.0 / 120.0,
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        };
        let boundary_params = MpmTerrainBoundaryParams {
            sdf_velocity_threshold_m: particle_radius_m(DEFAULT_MATERIAL_PARAMS) * 2.0,
            ..Default::default()
        };
        let deep_limit_m = -0.5 * boundary_params.sdf_velocity_threshold_m;

        let mut max_deep_ratio = 0.0f32;
        let mut final_deep_ratio = 0.0f32;
        for _ in 0..900 {
            sampler.begin_step();
            let _ = step_single_rate_coupled(
                &mut particles,
                &mut grid,
                Some(&terrain),
                Some(&mut sampler),
                &params,
                &boundary_params,
            );
            sampler.end_step();
            final_deep_ratio = deep_penetration_ratio(&terrain, &particles, deep_limit_m);
            max_deep_ratio = max_deep_ratio.max(final_deep_ratio);
        }

        assert!(
            final_deep_ratio <= 0.03,
            "final deep penetration ratio too high: {:.6}",
            final_deep_ratio
        );
        assert!(
            max_deep_ratio <= 0.10,
            "deep penetration accumulated too much during run: {:.6}",
            max_deep_ratio
        );
        for (i, &position) in particles.x.iter().enumerate() {
            assert!(
                position.is_finite(),
                "particle {i} has non-finite position: {:?}",
                position
            );
        }
    }

    #[test]
    fn block_index_table_assigns_unique_residents_and_supports() {
        let (mut particles, mut grid) = make_world_with_two_blocks();
        let mut table = MpmBlockIndexTable::default();
        refresh_block_index_table(
            &mut particles,
            &mut grid,
            &mut table,
            &[0.0, 0.0],
            Vec2::ZERO,
        );
        let owner_count: usize = (0..table.block_count())
            .map(|block_index| table.resident_indices(block_index).len())
            .sum();
        assert_eq!(owner_count, particles.len());
        let mut owned = vec![0u8; particles.len()];
        for block_index in 0..table.block_count() {
            for &particle_index in table.resident_indices(block_index) {
                owned[particle_index] = owned[particle_index].saturating_add(1);
                assert_eq!(particles.owner_block_id[particle_index], block_index);
            }
        }
        assert!(owned.into_iter().all(|count| count == 1));
        let support_count: usize = (0..table.block_count())
            .map(|block_index| table.support_indices(block_index).len())
            .sum();
        assert!(support_count >= particles.len());
    }

    #[test]
    fn refresh_rebins_owners_using_per_block_drift_prediction() {
        let (mut particles, mut grid) = make_world_with_two_blocks();
        particles.x[0] = Vec2::new(-0.05, 0.25);
        particles.v[0] = Vec2::new(1.0, 0.0);
        let mut table = MpmBlockIndexTable::default();
        refresh_block_index_table(
            &mut particles,
            &mut grid,
            &mut table,
            &[0.0, 0.0],
            Vec2::ZERO,
        );
        assert_eq!(particles.owner_block_id[0], 0);

        refresh_block_index_table(
            &mut particles,
            &mut grid,
            &mut table,
            &[0.10, 0.0],
            Vec2::ZERO,
        );
        assert_eq!(particles.owner_block_id[0], 1);
        assert!(table.resident_indices(1).contains(&0));
    }

    #[test]
    fn block_parallel_path_matches_serial_path() {
        let (mut particles_serial, mut grid_serial) = make_world_with_two_blocks();
        let (mut particles_parallel, mut grid_parallel) = make_world_with_two_blocks();
        let mut table_serial = MpmBlockIndexTable::default();
        let mut table_parallel = MpmBlockIndexTable::default();
        refresh_block_index_table(
            &mut particles_serial,
            &mut grid_serial,
            &mut table_serial,
            &[0.0, 0.0],
            Vec2::ZERO,
        );
        refresh_block_index_table(
            &mut particles_parallel,
            &mut grid_parallel,
            &mut table_parallel,
            &[0.0, 0.0],
            Vec2::ZERO,
        );
        let params = MpmWaterParams {
            dt: 1.0 / 120.0,
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        };
        let block_indices: Vec<_> = (0..grid_serial.block_count()).collect();
        let owner_block_drift_secs = vec![0.0; grid_serial.block_count()];
        let _serial_metrics = step_block_set_coupled(
            &mut particles_serial,
            &mut grid_serial,
            &mut table_serial,
            &block_indices,
            &block_indices,
            &owner_block_drift_secs,
            None,
            None,
            &params,
            &MpmTerrainBoundaryParams::default(),
            false,
        );
        let _parallel_metrics = step_block_set_coupled(
            &mut particles_parallel,
            &mut grid_parallel,
            &mut table_parallel,
            &block_indices,
            &block_indices,
            &owner_block_drift_secs,
            None,
            None,
            &params,
            &MpmTerrainBoundaryParams::default(),
            true,
        );

        for i in 0..particles_serial.len() {
            assert!((particles_serial.x[i] - particles_parallel.x[i]).length() <= 1e-6);
            assert!((particles_serial.v[i] - particles_parallel.v[i]).length() <= 1e-6);
            let lhs = particles_serial.f[i].to_cols_array();
            let rhs = particles_parallel.f[i].to_cols_array();
            for k in 0..4 {
                assert!((lhs[k] - rhs[k]).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn active_blocks_include_support_only_blocks() {
        let (mut particles, mut grid) = make_world_with_two_blocks();
        particles.x[0] = Vec2::new(-0.01, 0.25);
        particles.x[1] = Vec2::new(-0.02, 0.20);
        particles.x[2] = Vec2::new(-0.03, 0.30);
        let mut table = MpmBlockIndexTable::default();
        refresh_block_index_table(
            &mut particles,
            &mut grid,
            &mut table,
            &[0.0, 0.0],
            Vec2::ZERO,
        );
        let active = active_blocks_from_index_table(&table);
        assert!(active.len() >= 2);
        assert!(table.resident_indices(1).is_empty());
        assert!(!table.support_indices(1).is_empty());
    }

    #[test]
    fn support_indices_updated_when_particle_moves_within_block_toward_boundary() {
        let (mut particles, mut grid) = make_world_with_two_blocks();
        particles.x[0] = Vec2::new(-0.8, 0.0);
        particles.v[0] = Vec2::ZERO;
        let mut table = MpmBlockIndexTable::default();
        let drift = vec![0.0_f32; grid.block_count()];
        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);
        assert_eq!(particles.owner_block_id[0], 0);
        assert!(!table.support_indices(1).contains(&0));
        particles.x[0] = Vec2::new(-0.05, 0.0);
        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);
        assert!(
            table.support_indices(1).contains(&0),
            "particle 0 must appear in support list of block 1 near boundary"
        );
    }

    #[test]
    fn support_indices_cleared_when_particle_moves_away_from_boundary() {
        let (mut particles, mut grid) = make_world_with_two_blocks();
        particles.x[0] = Vec2::new(-0.05, 0.0);
        particles.v[0] = Vec2::ZERO;
        let mut table = MpmBlockIndexTable::default();
        let drift = vec![0.0_f32; grid.block_count()];
        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);
        assert_eq!(particles.owner_block_id[0], 0);
        assert!(table.support_indices(1).contains(&0));
        particles.x[0] = Vec2::new(-0.8, 0.0);
        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);
        assert!(
            !table.support_indices(1).contains(&0),
            "particle 0 must be removed from block 1 support list after moving away"
        );
    }

    #[test]
    fn coarse_to_fine_boundary_activates_receiver_support_block() {
        let (mut particles, mut grid) = make_world_with_coarse_fine_boundary();
        let mut table = MpmBlockIndexTable::default();
        let drift = vec![0.0_f32; grid.block_count()];
        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);
        assert_eq!(particles.owner_block_id[0], 1);
        assert!(!table.support_indices(0).contains(&0));
        particles.x[0] = Vec2::new(0.10, -0.25);
        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);
        assert!(table.support_indices(0).contains(&0));
        let active = active_blocks_from_index_table(&table);
        assert!(active.contains(&0));
    }

    #[test]
    fn coarse_fine_boundary_crossing_does_not_stall_without_forces() {
        let (mut particles, mut grid) = make_world_with_coarse_fine_boundary();
        particles.x[0] = Vec2::new(-0.20, -0.25);
        particles.v[0] = Vec2::new(1.0, 0.0);
        let params = MpmWaterParams {
            dt: 1.0 / 120.0,
            gravity: Vec2::ZERO,
            viscosity: 0.0,
            bulk_modulus: 0.0,
            ..Default::default()
        };

        let mut min_vx = f32::MAX;
        let mut crossed = false;
        for _ in 0..120 {
            let _ = step_single_rate(&mut particles, &mut grid, &params);
            min_vx = min_vx.min(particles.v[0].x);
            if particles.x[0].x > 0.40 {
                crossed = true;
                break;
            }
        }

        assert!(
            crossed,
            "particle did not cross the coarse/fine boundary in expected time; x={:.6}, vx={:.6}",
            particles.x[0].x, particles.v[0].x
        );
        assert!(
            min_vx > 0.15,
            "velocity dropped too much near coarse/fine boundary; min_vx={:.6}",
            min_vx
        );
    }

    #[test]
    fn refresh_clears_nodes_of_now_inactive_blocks() {
        let (mut particles, mut grid) = make_world_with_two_blocks();
        particles.x[0] = Vec2::new(-1.2, 0.25);
        particles.x[1] = Vec2::new(-1.1, 0.20);
        particles.x[2] = Vec2::new(-1.0, 0.30);
        if let Some(block) = grid.blocks_mut().get_mut(1) {
            if let Some(node) = block.nodes_mut().first_mut() {
                node.m = 3.0;
                node.render_mass_sum = 2.0;
            }
        }
        let mut table = MpmBlockIndexTable::default();
        refresh_block_index_table(
            &mut particles,
            &mut grid,
            &mut table,
            &[0.0, 0.0],
            Vec2::ZERO,
        );
        let right_block_mass: f32 = grid.blocks()[1].nodes().iter().map(|node| node.m).sum();
        assert!(right_block_mass <= 1e-8);
    }
}
