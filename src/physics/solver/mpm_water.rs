use bevy::prelude::*;
use rayon::prelude::*;

use super::terrain_boundary::{TerrainBoundarySample, TerrainBoundarySampler};
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::grid::{GridBlock, GridHierarchy, MpmBlockIndexTable};
use crate::physics::world::kernel::evaluate_quadratic_bspline_stencil_2d;
use crate::physics::world::particle::{ParticleMaterial, ParticleWorld};
use crate::physics::world::terrain::TerrainWorld;

const GRID_MASS_EPSILON: f32 = 1e-8;
const DET_EPSILON: f32 = 1e-6;

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
}

#[derive(Clone, Copy, Debug)]
pub struct MpmTerrainBoundaryParams {
    pub penetration_slop_m: f32,
    pub tangential_damping: f32,
}

impl Default for MpmTerrainBoundaryParams {
    fn default() -> Self {
        Self {
            penetration_slop_m: 0.0,
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
    index_table.ensure_block_count(block_count);
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
        let next_owner = grid
            .block_index_for_position(x_pred)
            .unwrap_or_else(|| particles.owner_block_id[i].min(block_count.saturating_sub(1)));
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

    let inv_h = 1.0
        / grid
            .blocks()
            .first()
            .map(|b| b.h_b)
            .unwrap_or(1.0)
            .max(DET_EPSILON);
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
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_pred, inv_h);
        for sample in stencil.samples() {
            let Some(location) = grid.node_location(sample.node) else {
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

pub fn apply_terrain_boundary_to_continuum(
    particles: &mut ContinuumParticleWorld,
    terrain: &TerrainWorld,
    penetration_slop_m: f32,
    tangential_damping: f32,
) -> usize {
    let mut corrected = 0usize;
    let penetration_slop_m = penetration_slop_m.max(0.0);
    let tangential_scale = (1.0 - tangential_damping).clamp(0.0, 1.0);
    for i in 0..particles.len() {
        let mut had_penetration = false;
        let mut boundary_normal = Vec2::ZERO;
        for _ in 0..4 {
            let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(particles.x[i])
            else {
                break;
            };
            if normal == Vec2::ZERO {
                break;
            }
            let penetration = penetration_slop_m - signed_distance;
            if penetration <= 0.0 {
                break;
            }
            particles.x[i] += normal * penetration;
            boundary_normal = normal;
            had_penetration = true;
        }
        if !had_penetration {
            continue;
        }

        let normal = if boundary_normal == Vec2::ZERO {
            Vec2::Y
        } else {
            boundary_normal
        };
        let normal_speed = particles.v[i].dot(normal);
        let mut corrected_v = particles.v[i];
        if normal_speed < 0.0 {
            corrected_v -= normal * normal_speed;
        }
        let tangent = corrected_v - normal * corrected_v.dot(normal);
        particles.v[i] = corrected_v - tangent * (1.0 - tangential_scale);
        corrected += 1;
    }
    corrected
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

    // P2G phase: write only into each target block's local node array.
    let blocks = grid.blocks_mut();
    let blocks_ptr = blocks.as_mut_ptr() as usize;
    if parallel_enabled {
        grid_blocks.par_iter().for_each(|&block_index| {
            if block_index >= blocks.len() {
                return;
            }
            let owner_indices = index_table.owner_indices(block_index);
            let ghost_indices = index_table.ghost_indices(block_index);
            // SAFETY: block_index entries are deduplicated above, so each block is mutably borrowed once.
            let block = unsafe { &mut *((blocks_ptr as *mut GridBlock).add(block_index)) };
            p2g_block(
                particles,
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
            let Some(block) = grid.blocks_mut().get_mut(block_index) else {
                continue;
            };
            p2g_block(
                particles,
                index_table.owner_indices(block_index),
                index_table.ghost_indices(block_index),
                block,
                block.dt_b,
                owner_block_drift_secs,
                params,
            );
        }
    }

    let mut boundary_samples_by_block = Vec::<Option<Vec<TerrainBoundarySample>>>::new();
    boundary_samples_by_block.resize_with(grid.block_count(), || None);
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

    // Grid update phase.
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

    let mut staged_updates = Vec::new();
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
    for update in staged_updates {
        particles.v[update.particle_index] = update.velocity;
        particles.c[update.particle_index] = update.c_matrix;
        particles.x[update.particle_index] = update.position;
        particles.f[update.particle_index] = update.deformation_gradient;
        metrics.max_particle_speed_mps =
            metrics.max_particle_speed_mps.max(update.velocity.length());
        metrics.clamped_particle_count += update.clamped_count;
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
                    sample.solid || sample.sdf_m < terrain_boundary_params.penetration_slop_m
                })
                .count();
            metrics.boundary_penetrating_node_ratio += penetrating_nodes as f32;
            if g2p_blocks.binary_search(&block_index).is_ok() {
                metrics.boundary_penetrating_particle_ratio += estimate_penetrating_particle_ratio(
                    particles,
                    index_table.owner_indices(block_index),
                    block,
                    boundary_samples,
                    terrain_boundary_params.penetration_slop_m,
                );
            }
        }
        for (node, sample) in block.nodes().iter().zip(boundary_samples.iter()) {
            if !(sample.solid || sample.sdf_m < terrain_boundary_params.penetration_slop_m) {
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
    penetration_slop_m: f32,
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
        if w_acc > 1e-6 && (sdf_acc / w_acc) < penetration_slop_m {
            penetrating += 1;
        }
    }
    penetrating as f32 / particle_indices.len() as f32
}

fn p2g_block(
    particles: &ContinuumParticleWorld,
    owner_indices: &[usize],
    ghost_indices: &[usize],
    block: &mut GridBlock,
    dt: f32,
    owner_block_drift_secs: &[f32],
    params: &MpmWaterParams,
) {
    block.clear_nodes();
    let dt = dt.max(0.0);
    let h = block.h_b.max(DET_EPSILON);
    let inv_h = 1.0 / h;
    let inv_cell_area = inv_h * inv_h;
    let mut particle_indices = Vec::with_capacity(owner_indices.len() + ghost_indices.len());
    particle_indices.extend_from_slice(owner_indices);
    particle_indices.extend_from_slice(ghost_indices);

    // Pass 1A: owner particles transfer render mass and momentum.
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
        if let Some(node) = block.node_mut_by_world(nearest_node) {
            node.render_mass_sum += m_p;
            node.render_mass_pos_sum += m_p * x_p;
        }
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        for sample in stencil.samples() {
            let node_world = sample.node.as_vec2() * h;
            let affine_velocity = c_p * (node_world - x_p);
            let Some(node) = block.node_mut_by_world(sample.node) else {
                continue;
            };
            let mass_contrib = sample.weight * m_p;
            node.m += mass_contrib;
            node.p += mass_contrib * (v_p + affine_velocity);
        }
    }
    // Pass 1B: ghost particles contribute only to mass/momentum.
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
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        for sample in stencil.samples() {
            let node_world = sample.node.as_vec2() * h;
            let affine_velocity = c_p * (node_world - x_p);
            let Some(node) = block.node_mut_by_world(sample.node) else {
                continue;
            };
            let mass_contrib = sample.weight * m_p;
            node.m += mass_contrib;
            node.p += mass_contrib * (v_p + affine_velocity);
        }
    }

    // Pass 2: estimate particle density from the transferred grid mass.
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
        let x_p = x_base + v_base * drift + 0.5 * params.gravity * drift * drift;
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        let mut rho_acc = 0.0f32;
        for sample in stencil.samples() {
            let Some(node) = block.node_by_world(sample.node) else {
                continue;
            };
            rho_acc += sample.weight * node.m * inv_cell_area;
        }
        density[local] = rho_acc.max(0.0);
    }

    // Pass 3: pressure/viscous internal force contribution.
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

        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        for sample in stencil.samples() {
            let Some(node) = block.node_mut_by_world(sample.node) else {
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
                if sample.solid || sample.sdf_m < boundary_params.penetration_slop_m {
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
    let inv_h = 1.0 / h.max(DET_EPSILON);
    let mut updates = Vec::with_capacity(owner_indices.len());
    for &i in owner_indices {
        let Some(&x_p) = particles.x.get(i) else {
            continue;
        };
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        let mut next_v = Vec2::ZERO;
        let mut next_grad_v = Mat2::ZERO;

        for sample in stencil.samples() {
            let Some(node) = grid.node_by_world(sample.node) else {
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
    use crate::physics::material::{DEFAULT_MATERIAL_PARAMS, terrain_boundary_radius_m};
    use crate::physics::solver::terrain_boundary::TerrainBoundarySampler;
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
    fn terrain_boundary_projection_pushes_particle_outside_solid() {
        let mut terrain = TerrainWorld::default();
        terrain.clear();
        terrain.ensure_chunk_loaded(IVec2::ZERO);
        terrain.clear_loaded_cells();
        terrain.set_cell(IVec2::ZERO, TerrainCell::stone());
        terrain
            .rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));

        let mut continuum = ContinuumParticleWorld::default();
        continuum.spawn_water_particle(
            cell_to_world_center(IVec2::ZERO),
            Vec2::new(0.0, -1.0),
            1.0,
            0.001,
        );

        let corrected = apply_terrain_boundary_to_continuum(&mut continuum, &terrain, 0.0, 0.1);
        assert_eq!(corrected, 1);
        let (signed_distance, _) = terrain
            .sample_signed_distance_and_normal(continuum.x[0])
            .expect("signed distance should be available after projection");
        assert!(signed_distance >= -1e-3);
        let (_, normal) = terrain
            .sample_signed_distance_and_normal(continuum.x[0])
            .expect("normal should be available after projection");
        assert!(continuum.v[0].dot(normal) >= -1e-4);
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
            &MpmTerrainBoundaryParams::default(),
        );
        sampler.end_step();
        assert!(metrics.boundary_penetrating_node_ratio > 0.0);
        assert!(metrics.boundary_momentum_exchange.length() > 0.0);
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
