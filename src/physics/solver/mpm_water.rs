use std::time::Instant;

use bevy::log::tracing;
use bevy::prelude::*;
use rayon::prelude::*;

use super::terrain_boundary::{TerrainBoundarySample, TerrainBoundarySampler};
use crate::physics::profiler::process_cpu_time_seconds;
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::grid::{GridBlock, GridHierarchy, MpmBlockIndexTable};
use crate::physics::world::kernel::evaluate_quadratic_bspline_stencil_2d;
use crate::physics::world::particle::{ParticleMaterial, ParticleWorld};
use crate::physics::world::terrain::{CELL_SIZE_M, TerrainWorld};

const GRID_MASS_EPSILON: f32 = 1e-8;
const DET_EPSILON: f32 = 1e-6;

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
    /// Wall-clock time for G2P (particle state update from grid).
    pub g2p_wall_secs: f64,
    /// Process CPU time for G2P (particle state update from grid).
    pub g2p_cpu_secs: f64,
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
        if !matches!(material, ParticleMaterial::WaterLiquid) {
            continue;
        }
        continuum.spawn_water_particle(position, velocity, mass, mass.max(0.0) * inv_rho0);
    }
    continuum.len()
}

pub fn sync_continuum_to_particle_world(
    particles: &mut ParticleWorld,
    continuum: &ContinuumParticleWorld,
) -> bool {
    let water_indices: Vec<usize> = particles
        .materials()
        .iter()
        .enumerate()
        .filter_map(|(i, &material)| {
            if matches!(material, ParticleMaterial::WaterLiquid) {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if water_indices.len() != continuum.len() {
        return false;
    }
    for (continuum_index, &particle_index) in water_indices.iter().enumerate() {
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
    if prev_block_count != block_count || index_table.neighbor_block_indices(0) == [None; 8] {
        index_table.rebuild_neighbor_map(grid.blocks());
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
        .map(|block_index| index_table.owner_indices(block_index).len())
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
        let mut next_owner = grid
            .block_index_for_position(x_pred)
            .unwrap_or_else(|| particles.owner_block_id[i].min(block_count.saturating_sub(1)));
        let mut best_missing = missing_stencil_nodes_for_block(grid, next_owner, x_pred);
        if best_missing > 0 {
            let mut candidates = Vec::with_capacity(9);
            candidates.push(next_owner);
            for maybe_neighbor in index_table.neighbor_block_indices(next_owner) {
                if let Some(neighbor) = maybe_neighbor {
                    candidates.push(neighbor);
                }
            }
            candidates.sort_unstable();
            candidates.dedup();
            for candidate in candidates {
                let missing = missing_stencil_nodes_for_block(grid, candidate, x_pred);
                if missing < best_missing {
                    next_owner = candidate;
                    best_missing = missing;
                    if best_missing == 0 {
                        break;
                    }
                }
            }
        }
        if particles.owner_block_id[i] != next_owner {
            particles.owner_block_id[i] = next_owner;
            moved += 1;
            needs_rebin = true;
        }
    }

    if !needs_rebin {
        clear_inactive_block_nodes(grid, index_table);
        return;
    }

    for block_index in 0..block_count {
        if let Some(owner_indices) = index_table.owner_indices_mut(block_index) {
            owner_indices.clear();
        }
        if let Some(ghost_indices) = index_table.ghost_indices_mut(block_index) {
            ghost_indices.clear();
        }
    }
    for (particle_index, &owner) in particles.owner_block_id.iter().enumerate() {
        if let Some(owner_indices) = index_table.owner_indices_mut(owner) {
            owner_indices.push(particle_index);
        }
    }

    for particle_index in 0..particle_count {
        let owner = particles.owner_block_id[particle_index];
        let mut touched_blocks = Vec::with_capacity(4);
        let owner_block = particles
            .owner_block_id
            .get(particle_index)
            .copied()
            .unwrap_or(0)
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(owner_block)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let x_pred = particles.x[particle_index]
            + particles.v[particle_index] * drift
            + 0.5 * gravity * drift * drift;
        // 各粒子はowner blockのh_bでカーネルstencilを評価する（可変h_b対応）
        let h_owner = grid
            .blocks()
            .get(owner)
            .map(|b| b.h_b)
            .unwrap_or(1.0)
            .max(DET_EPSILON);
        let owner_scale = node_scale_for_h(h_owner);
        let inv_h = 1.0 / h_owner;
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_pred, inv_h);
        for sample in stencil.samples() {
            let world_key = sample.node * owner_scale;
            let Some(location) = grid.node_location(world_key) else {
                continue;
            };
            if location.block_index != owner {
                touched_blocks.push(location.block_index);
            }
        }
        touched_blocks.sort_unstable();
        touched_blocks.dedup();
        for block_index in touched_blocks {
            if let Some(ghost_indices) = index_table.ghost_indices_mut(block_index) {
                ghost_indices.push(particle_index);
            }
        }
    }
    for block_index in 0..block_count {
        if let Some(ghost_indices) = index_table.ghost_indices_mut(block_index) {
            ghost_indices.sort_unstable();
            ghost_indices.dedup();
        }
    }
    index_table.set_moved_particle_count(moved);
    index_table.set_rebinned_this_step(true);
    clear_inactive_block_nodes(grid, index_table);
}

/// Rebuilds `ghost_indices` for a single block by scanning owner particles
/// from world-space nearby blocks.
///
/// This is the per-boundary ghost update described in MPM-WATER-04: instead
/// of a full global scan (which `refresh_block_index_table` performs only on
/// rebin), we rebuild the ghost list for *one* block from nearby blocks'
/// owner particles. Callers can execute this for all blocks before P2G so
/// ghosts reflect current predicted positions — even when no ownership
/// change has occurred.
pub fn refresh_ghost_indices_for_block(
    particles: &ContinuumParticleWorld,
    grid: &GridHierarchy,
    index_table: &mut MpmBlockIndexTable,
    block_index: usize,
    owner_block_drift_secs: &[f32],
    gravity: Vec2,
) {
    let block_count = grid.block_count();
    if block_index >= block_count {
        return;
    }
    let Some(target_block) = grid.blocks().get(block_index) else {
        return;
    };
    let t_h = target_block.h_b.max(DET_EPSILON);
    let t_origin = target_block.origin_node;
    let t_end = t_origin + target_block.node_dims.as_ivec2(); // exclusive upper bound
    let t_world_min = t_origin.as_vec2() * t_h - Vec2::splat(0.5 * t_h);
    let t_world_max = (t_end - IVec2::ONE).as_vec2() * t_h + Vec2::splat(0.5 * t_h);

    // Collect owner particle indices from world-space nearby blocks.
    // We clone the slices into a local Vec to release the immutable borrow on
    // `index_table` before taking a mutable borrow for ghost_indices_mut below.
    let mut candidate_particles: Vec<usize> = Vec::new();
    for nk in 0..block_count {
        if nk == block_index {
            continue;
        }
        let Some(block) = grid.blocks().get(nk) else {
            continue;
        };
        let b_h = block.h_b.max(DET_EPSILON);
        let b_origin = block.origin_node;
        let b_end = b_origin + block.node_dims.as_ivec2();
        let b_world_min = b_origin.as_vec2() * b_h - Vec2::splat(0.5 * b_h);
        let b_world_max = (b_end - IVec2::ONE).as_vec2() * b_h + Vec2::splat(0.5 * b_h);
        let margin = (t_h.max(b_h)) * 2.5;
        let separated = b_world_max.x + margin < t_world_min.x
            || b_world_min.x - margin > t_world_max.x
            || b_world_max.y + margin < t_world_min.y
            || b_world_min.y - margin > t_world_max.y;
        if separated {
            continue;
        }
        candidate_particles.extend_from_slice(index_table.owner_indices(nk));
    }

    // Check each candidate particle's predicted kernel stencil for overlap
    // with any node belonging to `block_index`.
    //
    // 可変h_b対応: 各粒子はowner blockのh_bでstencilを評価するが、target blockの
    // ノード範囲チェックはtarget blockのh_bの整数座標系で行う必要がある。
    // owner block h_b != target block h_bの場合、stencilのノード座標はowner座標系の
    // ためtargetのノード範囲と一致しない。そこでtarget h_bでも stencil を評価し、
    // どちらかのstencilのノードがtarget範囲に届くかを判定する。
    // これにより粗→細・細→粗の境界をまたぐゴーストを正しく捕捉できる。
    let mut new_ghost_indices: Vec<usize> = Vec::with_capacity(candidate_particles.len());
    for particle_index in candidate_particles {
        let owner_blk = particles
            .owner_block_id
            .get(particle_index)
            .copied()
            .unwrap_or(0)
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(owner_blk)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let x_pred = particles.x[particle_index]
            + particles.v[particle_index] * drift
            + 0.5 * gravity * drift * drift;

        // owner blockのh_bでstencilを評価してtarget範囲に届くか判定する。
        // target blockと同じh_bを使う場合は最適化済みrange checkを使う。
        // 異なるh_bの場合はworld座標ベースで物理的な重なりを判定する。
        let h_owner = grid
            .blocks()
            .get(owner_blk)
            .map(|b| b.h_b)
            .unwrap_or(t_h)
            .max(DET_EPSILON);
        let in_target_world_range = |node: IVec2, inv_h: f32| -> bool {
            // このノードのworld座標範囲と target blockのworld座標範囲が交差するか
            let node_world_min = node.as_vec2() * (1.0 / inv_h) - Vec2::splat(0.5 / inv_h);
            let node_world_max = node.as_vec2() * (1.0 / inv_h) + Vec2::splat(0.5 / inv_h);
            node_world_min.x <= t_world_max.x
                && node_world_max.x >= t_world_min.x
                && node_world_min.y <= t_world_max.y
                && node_world_max.y >= t_world_min.y
        };
        // 常にworld座標ベースで判定することで、可変h_b境界での取りこぼしを防ぐ。
        let inv_h_owner = 1.0 / h_owner;
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_pred, inv_h_owner);
        let touches_target = stencil
            .samples()
            .iter()
            .any(|s| in_target_world_range(s.node, inv_h_owner));
        if touches_target {
            new_ghost_indices.push(particle_index);
        }
    }
    new_ghost_indices.sort_unstable();
    new_ghost_indices.dedup();

    if let Some(ghosts) = index_table.ghost_indices_mut(block_index) {
        *ghosts = new_ghost_indices;
    }
}

pub fn estimate_block_max_speed(
    particles: &ContinuumParticleWorld,
    index_table: &MpmBlockIndexTable,
    block_index: usize,
) -> f32 {
    let mut max_speed = index_table
        .owner_indices(block_index)
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
            .ghost_indices(block_index)
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
        if !(index_table.owner_indices(block_index).is_empty()
            && index_table.ghost_indices(block_index).is_empty())
        {
            active.push(block_index);
        }
    }
    active
}

fn clear_inactive_block_nodes(grid: &mut GridHierarchy, index_table: &MpmBlockIndexTable) {
    for block_index in 0..grid.block_count() {
        if !(index_table.owner_indices(block_index).is_empty()
            && index_table.ghost_indices(block_index).is_empty())
        {
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
    index_table: &MpmBlockIndexTable,
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
    debug_assert_owner_layout(particles, index_table, grid.block_count());

    // P2G Phase 1: mass/momentum transfer into each target block's local node array.
    // Store both pointers as usize so rayon closures can capture them (raw pointers are not Sync).
    let grid_ptr = grid as *mut GridHierarchy as usize;
    let (blocks_ptr, blocks_len) = {
        let blocks = unsafe { &mut *(grid_ptr as *mut GridHierarchy) }.blocks_mut();
        (blocks.as_mut_ptr() as usize, blocks.len())
    };
    // 各blockのh_bスライス: ghost粒子がowner block h_bでカーネルを評価するために使う。
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
            grid_blocks.par_iter().for_each(|&block_index| {
                if block_index >= blocks_len {
                    return;
                }
                let owner_indices = index_table.owner_indices(block_index);
                let ghost_indices = index_table.ghost_indices(block_index);
                // SAFETY: block_index entries are deduplicated above, so each block is mutably borrowed once.
                let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                p2g_mass_momentum(
                    particles,
                    owner_indices,
                    ghost_indices,
                    block,
                    owner_block_drift_secs,
                    &block_h_b,
                    params,
                );
            });
        } else {
            for &block_index in &grid_blocks {
                if block_index >= blocks_len {
                    continue;
                }
                let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                p2g_mass_momentum(
                    particles,
                    index_table.owner_indices(block_index),
                    index_table.ghost_indices(block_index),
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
            grid_blocks.par_iter().for_each(|&block_index| {
                if block_index >= blocks_len {
                    return;
                }
                let owner_indices = index_table.owner_indices(block_index);
                let ghost_indices = index_table.ghost_indices(block_index);
                let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                let grid_ref = unsafe { &*(grid_ptr as *const GridHierarchy) };
                p2g_pressure(
                    particles,
                    grid_ref,
                    owner_indices,
                    ghost_indices,
                    block,
                    block.dt_b,
                    owner_block_drift_secs,
                    params,
                );
            });
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
                    index_table.owner_indices(block_index),
                    index_table.ghost_indices(block_index),
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
            grid_blocks.par_iter().for_each(|&block_index| {
                if block_index >= blocks.len() {
                    return;
                }
                let boundary_samples = boundary_samples_by_block
                    .get(block_index)
                    .and_then(|entry| entry.as_deref());
                // SAFETY: block_index entries are deduplicated above, so each block is mutably borrowed once.
                let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
                grid_update(
                    block,
                    block.dt_b,
                    params,
                    boundary_samples,
                    terrain_boundary_params,
                );
            });
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
        let _span = tracing::info_span!("physics::mpm::g2p").entered();
        let t0 = Instant::now();
        let cpu0 = process_cpu_time_seconds().unwrap_or(0.0);
        if parallel_enabled {
            staged_updates = g2p_blocks
                .par_iter()
                .map(|&block_index| {
                    let owner_indices = index_table.owner_indices(block_index);
                    let Some(block) = grid.blocks().get(block_index) else {
                        return Vec::new();
                    };
                    g2p_collect_owner_updates(
                        particles,
                        grid,
                        owner_indices,
                        block.h_b,
                        block.dt_b,
                        params,
                    )
                })
                .reduce(Vec::new, |mut lhs, mut rhs| {
                    lhs.append(&mut rhs);
                    lhs
                });
        } else {
            for &block_index in &g2p_blocks {
                let owner_indices = index_table.owner_indices(block_index);
                let Some(block) = grid.blocks().get(block_index) else {
                    continue;
                };
                staged_updates.extend(g2p_collect_owner_updates(
                    particles,
                    grid,
                    owner_indices,
                    block.h_b,
                    block.dt_b,
                    params,
                ));
            }
        }
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
        metrics.g2p_wall_secs += t0.elapsed().as_secs_f64();
        metrics.g2p_cpu_secs += (process_cpu_time_seconds().unwrap_or(cpu0) - cpu0).max(0.0);
    }

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
                    index_table.owner_indices(block_index),
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
        &index_table,
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
/// owner と ghost の両方の粒子を処理する。ブロックの外側のノード
/// (`block.node_mut_by_world` が `None` を返すノード) への書き込みは
/// スキップされる — そのノードは隣接ブロックの P2G で処理される。
///
/// この関数は全ブロックで完了してから `p2g_pressure` を呼ぶこと。
fn p2g_mass_momentum(
    particles: &ContinuumParticleWorld,
    owner_indices: &[usize],
    ghost_indices: &[usize],
    block: &mut GridBlock,
    owner_block_drift_secs: &[f32],
    // 各blockのh_b（block index順）: ghost粒子がowner block h_bでカーネルを評価するために使う
    block_h_b: &[f32],
    params: &MpmWaterParams,
) {
    block.clear_nodes();
    let h = block.h_b.max(DET_EPSILON);
    let block_scale = node_scale_for_h(h);
    let inv_h = 1.0 / h;

    // Pass 1A: owner particles — render mass 集計も行う。
    for &i in owner_indices {
        let Some((&x_base, &v_base)) = particles.x.get(i).zip(particles.v.get(i)) else {
            continue;
        };
        let Some(&c_p) = particles.c.get(i) else {
            continue;
        };
        let Some(&m_p_raw) = particles.m.get(i) else {
            continue;
        };
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
        let x_p = x_base + v_base * drift + 0.5 * params.gravity * drift * drift;
        let v_p = v_base + params.gravity * drift;
        let m_p = m_p_raw.max(0.0);
        let nearest_node = IVec2::new(
            (x_p.x * inv_h).round() as i32,
            (x_p.y * inv_h).round() as i32,
        );
        let nearest_key = nearest_node * block_scale;
        if block.is_world_key_owned(nearest_key) {
            if let Some(node) = block.node_mut_by_world_key(nearest_key) {
                node.render_mass_sum += m_p;
                node.render_mass_pos_sum += m_p * x_p;
            }
        }
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        for sample in stencil.samples() {
            let world_key = sample.node * block_scale;
            if !block.is_world_key_owned(world_key) {
                continue;
            }
            let node_world = sample.node.as_vec2() * h;
            let affine_velocity = c_p * (node_world - x_p);
            let Some(node) = block.node_mut_by_world_key(world_key) else {
                continue;
            };
            let mass_contrib = sample.weight * m_p;
            node.m += mass_contrib;
            node.p += mass_contrib * (v_p + affine_velocity);
        }
    }
    // Pass 1B: ghost particles — owner block h_bでカーネルを評価する（可変h_b対応）。
    // coarse/fine境界でも owner h_b 基準のAPIC affine項を使って運動量転送する。
    for &i in ghost_indices {
        let Some((&x_base, &v_base)) = particles.x.get(i).zip(particles.v.get(i)) else {
            continue;
        };
        let Some(&c_p) = particles.c.get(i) else {
            continue;
        };
        let Some(&m_p_raw) = particles.m.get(i) else {
            continue;
        };
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
        let x_p = x_base + v_base * drift + 0.5 * params.gravity * drift * drift;
        let v_p = v_base + params.gravity * drift;
        let m_p = m_p_raw.max(0.0);
        // ghost粒子のカーネルはowner blockのh_bで評価する
        let h_owner = block_h_b
            .get(owner_block)
            .copied()
            .unwrap_or(h)
            .max(DET_EPSILON);
        let owner_scale = node_scale_for_h(h_owner);
        let inv_h_owner = 1.0 / h_owner;
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h_owner);
        for sample in stencil.samples() {
            let world_key = sample.node * owner_scale;
            if !block.is_world_key_owned(world_key) {
                continue;
            }
            let Some(node) = block.node_mut_by_world_key(world_key) else {
                continue;
            };
            let mass_contrib = sample.weight * m_p;
            node.m += mass_contrib;
            // 異解像度境界でも owner block のh_bで評価したAPIC affine項を使う。
            // これにより境界での不自然な運動量減衰を抑える。
            let node_world = sample.node.as_vec2() * h_owner;
            let affine_velocity = c_p * (node_world - x_p);
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
    owner_indices: &[usize],
    ghost_indices: &[usize],
    block: &mut GridBlock,
    dt: f32,
    owner_block_drift_secs: &[f32],
    params: &MpmWaterParams,
) {
    let dt = dt.max(0.0);
    let h = block.h_b.max(DET_EPSILON);

    let mut particle_indices = Vec::with_capacity(owner_indices.len() + ghost_indices.len());
    particle_indices.extend_from_slice(owner_indices);
    particle_indices.extend_from_slice(ghost_indices);

    // Pass 2: 全グリッド参照で完全な密度を推定する。
    // `block.node_by_world` ではなく `grid.node_by_world` を使うことで、
    // ブロック境界をまたぐ stencil ノードの質量も考慮される。
    let mut density = vec![0.0f32; particle_indices.len()];
    for (local, &particle_index) in particle_indices.iter().enumerate() {
        let Some((&x_base, &v_base)) = particles
            .x
            .get(particle_index)
            .zip(particles.v.get(particle_index))
        else {
            continue;
        };
        let owner_block = particles
            .owner_block_id
            .get(particle_index)
            .copied()
            .unwrap_or(0)
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(owner_block)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let h_p = grid
            .blocks()
            .get(owner_block)
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
        let particle_index = particle_indices[local];
        let Some((&x_base, &v_base)) = particles
            .x
            .get(particle_index)
            .zip(particles.v.get(particle_index))
        else {
            continue;
        };
        let owner_block = particles
            .owner_block_id
            .get(particle_index)
            .copied()
            .unwrap_or(0)
            .min(owner_block_drift_secs.len().saturating_sub(1));
        let drift = owner_block_drift_secs
            .get(owner_block)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let h_p = grid
            .blocks()
            .get(owner_block)
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

fn debug_assert_owner_layout(
    particles: &ContinuumParticleWorld,
    index_table: &MpmBlockIndexTable,
    block_count: usize,
) {
    #[cfg(debug_assertions)]
    {
        let mut ownership = vec![0u8; particles.len()];
        for block_index in 0..block_count {
            for &particle_index in index_table.owner_indices(block_index) {
                if particle_index >= particles.len() {
                    panic!(
                        "owner index out of range: block {} -> particle {}",
                        block_index, particle_index
                    );
                }
                if particles.owner_block_id[particle_index] != block_index {
                    panic!(
                        "owner mismatch: particle {} says {}, table says {}",
                        particle_index, particles.owner_block_id[particle_index], block_index
                    );
                }
                ownership[particle_index] = ownership[particle_index].saturating_add(1);
            }
        }
        for (particle_index, &count) in ownership.iter().enumerate() {
            assert_eq!(
                count, 1,
                "owner uniqueness violated for particle {}: count {}",
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
    use crate::physics::world::continuum::ContinuumMaterial;
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
            ContinuumMaterial::Water,
        );
        particles.spawn_particle(
            Vec2::new(0.45, 0.25),
            Vec2::ZERO,
            1.0,
            1.0 / 1_000.0,
            ContinuumMaterial::Water,
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
            ContinuumMaterial::Water,
        );
        particles.spawn_particle(
            Vec2::new(0.05, 0.25),
            Vec2::new(-0.2, 0.0),
            1.0,
            1.0 / 1_000.0,
            ContinuumMaterial::Water,
        );
        particles.spawn_particle(
            Vec2::new(0.45, 0.35),
            Vec2::new(0.1, -0.2),
            1.0,
            1.0 / 1_000.0,
            ContinuumMaterial::Water,
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
            ContinuumMaterial::Water,
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
                            ContinuumMaterial::Water,
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
    fn rebuild_continuum_filters_water_particles_only() {
        let mut particles = ParticleWorld::default();
        particles
            .restore_from_snapshot(
                vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(2.0, 0.0),
                ],
                vec![Vec2::ZERO; 3],
                vec![
                    ParticleMaterial::WaterLiquid,
                    ParticleMaterial::StoneSolid,
                    ParticleMaterial::WaterLiquid,
                ],
            )
            .unwrap();
        let mut continuum = ContinuumParticleWorld::default();
        let count = rebuild_continuum_from_particle_world(
            &particles,
            &mut continuum,
            &MpmWaterParams::default(),
        );
        assert_eq!(count, 2);
        assert_eq!(continuum.len(), 2);
        assert_eq!(continuum.x[0], Vec2::new(0.0, 0.0));
        assert_eq!(continuum.x[1], Vec2::new(2.0, 0.0));
    }

    #[test]
    fn sync_continuum_updates_only_water_particles() {
        let mut particles = ParticleWorld::default();
        particles
            .restore_from_snapshot(
                vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(2.0, 0.0),
                ],
                vec![Vec2::ZERO; 3],
                vec![
                    ParticleMaterial::WaterLiquid,
                    ParticleMaterial::StoneSolid,
                    ParticleMaterial::WaterLiquid,
                ],
            )
            .unwrap();

        let mut continuum = ContinuumParticleWorld::default();
        continuum.spawn_water_particle(Vec2::new(5.0, 6.0), Vec2::new(1.0, 2.0), 1.0, 0.001);
        continuum.spawn_water_particle(Vec2::new(7.0, 8.0), Vec2::new(3.0, 4.0), 1.0, 0.001);

        assert!(sync_continuum_to_particle_world(&mut particles, &continuum));
        assert_eq!(particles.pos[0], Vec2::new(5.0, 6.0));
        assert_eq!(particles.vel[0], Vec2::new(1.0, 2.0));
        assert_eq!(particles.pos[1], Vec2::new(1.0, 0.0));
        assert_eq!(particles.pos[2], Vec2::new(7.0, 8.0));
        assert_eq!(particles.vel[2], Vec2::new(3.0, 4.0));
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
    fn block_index_table_assigns_unique_owners_and_ghosts() {
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
            .map(|block_index| table.owner_indices(block_index).len())
            .sum();
        assert_eq!(owner_count, particles.len());
        let mut owned = vec![0u8; particles.len()];
        for block_index in 0..table.block_count() {
            for &particle_index in table.owner_indices(block_index) {
                owned[particle_index] = owned[particle_index].saturating_add(1);
                assert_eq!(particles.owner_block_id[particle_index], block_index);
            }
        }
        assert!(owned.into_iter().all(|count| count == 1));
        let ghost_count: usize = (0..table.block_count())
            .map(|block_index| table.ghost_indices(block_index).len())
            .sum();
        assert!(ghost_count > 0);
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
        assert!(table.owner_indices(1).contains(&0));
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
            &table_serial,
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
            &table_parallel,
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
    fn active_blocks_include_ghost_only_blocks() {
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
        assert!(table.owner_indices(1).is_empty());
        assert!(!table.ghost_indices(1).is_empty());
    }

    /// Regression test for the ghost-staleness bug (MPM-WATER-04).
    ///
    /// When a particle moves *within* its owner block toward a block boundary
    /// without crossing it, `refresh_block_index_table` exits early (no
    /// ownership change → `needs_rebin = false`) and leaves the neighboring
    /// block's `ghost_indices` stale.  `refresh_ghost_indices_for_block` must
    /// detect the new overlap and update the ghost list correctly.
    #[test]
    fn ghost_indices_updated_when_particle_moves_within_block_toward_boundary() {
        // Two side-by-side blocks sharing the x=0 boundary.
        // Block 0: nodes x in [-8, -1], block 1: nodes x in [0, 7].
        let (mut particles, mut grid) = make_world_with_two_blocks();

        // Place particle 0 well inside block 0 (left block) so its stencil
        // does NOT reach block 1's nodes.
        particles.x[0] = Vec2::new(-0.8, 0.0);
        particles.v[0] = Vec2::ZERO;

        let mut table = MpmBlockIndexTable::default();
        let drift = vec![0.0_f32; grid.block_count()];
        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);

        // Confirm: particle 0 is owned by block 0 and not a ghost of block 1.
        assert_eq!(particles.owner_block_id[0], 0);
        assert!(!table.ghost_indices(1).contains(&0));

        // Now move particle 0 close to x=0 (still in block 0, stencil reaches
        // block 1).  Do NOT call refresh_block_index_table so no rebin fires.
        particles.x[0] = Vec2::new(-0.05, 0.0);

        // --- Before fix: ghost_indices(1) still stale (misses particle 0) ---
        // Verify the per-block refresh correctly detects the overlap.
        refresh_ghost_indices_for_block(&particles, &grid, &mut table, 1, &drift, Vec2::ZERO);
        assert!(
            table.ghost_indices(1).contains(&0),
            "particle 0 must appear as ghost in block 1 after moving close to the boundary"
        );

        // Also verify that a particle far from the boundary does NOT leak into
        // ghost_indices (particle 2 starts deep in block 1).
        assert!(
            !table.ghost_indices(0).contains(&2),
            "particle 2 deep inside block 1 should not appear as ghost in block 0"
        );
    }

    /// Complementary check: after `refresh_ghost_indices_for_block`, a
    /// particle that previously appeared as a ghost (when near the boundary)
    /// is removed when it moves away — i.e., stale ghost entries are replaced,
    /// not merely appended.
    #[test]
    fn ghost_indices_cleared_when_particle_moves_away_from_boundary() {
        let (mut particles, mut grid) = make_world_with_two_blocks();

        // Start near boundary so particle 0 is a ghost of block 1.
        particles.x[0] = Vec2::new(-0.05, 0.0);
        particles.v[0] = Vec2::ZERO;

        let mut table = MpmBlockIndexTable::default();
        let drift = vec![0.0_f32; grid.block_count()];
        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);
        assert_eq!(particles.owner_block_id[0], 0);
        assert!(table.ghost_indices(1).contains(&0));

        // Move particle away from the boundary — stencil no longer reaches
        // block 1.  Simulate the "no rebin" scenario by calling only the
        // per-block refresh.
        particles.x[0] = Vec2::new(-0.8, 0.0);
        refresh_ghost_indices_for_block(&particles, &grid, &mut table, 1, &drift, Vec2::ZERO);
        assert!(
            !table.ghost_indices(1).contains(&0),
            "particle 0 must be removed from block 1 ghost list after moving away"
        );
    }

    #[test]
    fn coarse_to_fine_boundary_requires_refreshing_receiver_block_ghosts() {
        let (mut particles, mut grid) = make_world_with_coarse_fine_boundary();
        let mut table = MpmBlockIndexTable::default();
        let drift = vec![0.0_f32; grid.block_count()];

        refresh_block_index_table(&mut particles, &mut grid, &mut table, &drift, Vec2::ZERO);
        assert_eq!(particles.owner_block_id[0], 1);
        assert!(!table.ghost_indices(0).contains(&0));

        // Move within the coarse owner block near shared boundary x=0.
        particles.x[0] = Vec2::new(0.10, -0.25);

        // Refreshing only the owner-side block leaves the fine receiver stale.
        refresh_ghost_indices_for_block(&particles, &grid, &mut table, 1, &drift, Vec2::ZERO);
        assert!(!table.ghost_indices(0).contains(&0));

        // Refreshing all blocks activates the fine receiver as expected.
        for block_index in 0..grid.block_count() {
            refresh_ghost_indices_for_block(
                &particles,
                &grid,
                &mut table,
                block_index,
                &drift,
                Vec2::ZERO,
            );
        }
        assert!(table.ghost_indices(0).contains(&0));
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
            particles.x[0].x,
            particles.v[0].x
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
