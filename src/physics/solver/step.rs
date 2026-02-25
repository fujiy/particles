use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use bevy::log::tracing;

use super::mpm_water::{
    MpmTerrainBoundaryParams, MpmWaterParams, active_blocks_from_index_table, cfl_violated,
    estimate_block_max_speed, rebuild_continuum_from_particle_world, refresh_block_index_table,
    sound_speed_mps, step_block_set_coupled, sync_continuum_to_particle_world,
};
use super::terrain_boundary::TerrainBoundarySampler;
use super::types::StepSimulationTiming;
use crate::physics::profiler::process_cpu_time_seconds;
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::grid::{GridHierarchy, MpmBlockIndexTable};
use crate::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::{ParticleMaterial, ParticleWorld};
use crate::physics::world::terrain::{CELL_SIZE_M, TerrainWorld};

const MPM_CFL_SAFETY: f32 = 0.35;
const MPM_CFL_ACCEL_SAFETY: f32 = 0.35;
const MPM_TARGET_SOUND_SPEED_MPS: f32 = 16.0;
const MPM_TARGET_RHO0: f32 = 1_000.0;
const MPM_MIN_DT_SUB: f32 = 1.0e-4;
const MPM_BOUNDARY_SLOP_SCALE_DIAMETER: f32 = 2.0;
const MPM_RATE_LEVEL_CAP: u8 = 12;

fn max_pow2_rate_level(frame_dt: f32, min_dt: f32) -> u8 {
    let frame_dt = frame_dt.max(min_dt.max(1e-8));
    let min_dt = min_dt.max(1e-8);
    let mut level = 0u8;
    let mut dt = frame_dt;
    while level < MPM_RATE_LEVEL_CAP && dt * 0.5 >= min_dt {
        dt *= 0.5;
        level = level.saturating_add(1);
    }
    level
}

fn quantize_pow2_rate_level(frame_dt: f32, target_dt: f32, max_level: u8) -> u8 {
    let frame_dt = frame_dt.max(1e-8);
    let target_dt = target_dt.max(1e-8).min(frame_dt);
    let mut level = 0u8;
    let mut dt = frame_dt;
    while level < max_level && dt > target_dt {
        dt *= 0.5;
        level = level.saturating_add(1);
    }
    level
}

#[allow(clippy::too_many_arguments)]
fn configure_block_rate_level(
    block_index: usize,
    continuum_world: &ContinuumParticleWorld,
    grid_hierarchy: &mut GridHierarchy,
    mpm_block_index_table: &MpmBlockIndexTable,
    frame_dt: f32,
    base_dt_unit: f32,
    max_rate_level: u8,
    min_rate_level: u8,
    c: f32,
    a_max: f32,
    rate_level_by_block: &mut [u8],
    step_units_by_block: &mut [u64],
) {
    let Some(h_b) = grid_hierarchy
        .blocks()
        .get(block_index)
        .map(|block| block.h_b.max(1e-6))
    else {
        return;
    };
    let u_max = estimate_block_max_speed(continuum_world, mpm_block_index_table, block_index);
    let dt_u = if u_max > 1e-5 {
        MPM_CFL_SAFETY * h_b / u_max
    } else {
        frame_dt
    };
    let dt_c = MPM_CFL_SAFETY * h_b / (c + u_max).max(1e-6);
    let dt_a = MPM_CFL_ACCEL_SAFETY * (h_b / a_max).sqrt();
    let target_dt = dt_u.min(dt_c).min(dt_a).clamp(MPM_MIN_DT_SUB, frame_dt);
    let rate_level =
        quantize_pow2_rate_level(frame_dt, target_dt, max_rate_level).max(min_rate_level);
    rate_level_by_block[block_index] = rate_level;
    let step_units = 1_u64 << (max_rate_level - rate_level);
    step_units_by_block[block_index] = step_units.max(1);
    if let Some(block_mut) = grid_hierarchy.blocks_mut().get_mut(block_index) {
        block_mut.dt_b = (step_units_by_block[block_index] as f32 * base_dt_unit)
            .clamp(MPM_MIN_DT_SUB, frame_dt);
    }
}

pub(crate) fn step_simulation_once(
    terrain_world: &mut TerrainWorld,
    particle_world: &mut ParticleWorld,
    continuum_world: &mut ContinuumParticleWorld,
    grid_hierarchy: &mut GridHierarchy,
    mpm_block_index_table: &mut MpmBlockIndexTable,
    object_world: &mut ObjectWorld,
    object_field: &mut ObjectPhysicsField,
    terrain_boundary_sampler: &mut TerrainBoundarySampler,
    parallel_enabled: bool,
    terrain_boundary_radius_m: f32,
) -> StepSimulationTiming {
    let water_particle_count = particle_world
        .materials()
        .iter()
        .filter(|&&m| matches!(m, ParticleMaterial::WaterLiquid))
        .count();
    if water_particle_count == 0 {
        continuum_world.clear();
        mpm_block_index_table.clear();
        for block in grid_hierarchy.blocks_mut() {
            block.clear_nodes();
        }
    }
    if water_particle_count > 0 {
        if grid_hierarchy.block_count() == 0 {
            return StepSimulationTiming::default();
        }
        let particle_step_start = Instant::now();
        let particle_step_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
        let _span = tracing::info_span!("physics::mpm_water_step").entered();
        let mpm_params = MpmWaterParams {
            dt: particle_world.solver_params.fixed_dt,
            gravity: particle_world.solver_params.gravity_mps2,
            rho0: MPM_TARGET_RHO0,
            bulk_modulus: MPM_TARGET_RHO0 * MPM_TARGET_SOUND_SPEED_MPS * MPM_TARGET_SOUND_SPEED_MPS,
            ..Default::default()
        };
        let particle_radius_m = (terrain_boundary_radius_m - 0.5 * CELL_SIZE_M).max(0.0);
        let terrain_boundary_params = MpmTerrainBoundaryParams {
            penetration_slop_m: particle_radius_m * MPM_BOUNDARY_SLOP_SCALE_DIAMETER,
            ..Default::default()
        };
        if continuum_world.len() != water_particle_count {
            let _ =
                rebuild_continuum_from_particle_world(particle_world, continuum_world, &mpm_params);
        }
        let block_count = grid_hierarchy.block_count();
        let zero_owner_block_drift_secs = vec![0.0_f32; block_count];
        refresh_block_index_table(
            continuum_world,
            grid_hierarchy,
            mpm_block_index_table,
            &zero_owner_block_drift_secs,
            mpm_params.gravity,
        );
        let mut active_blocks = active_blocks_from_index_table(mpm_block_index_table);
        if active_blocks.is_empty() {
            return StepSimulationTiming::default();
        }
        let frame_dt = particle_world.solver_params.fixed_dt.max(MPM_MIN_DT_SUB);
        let max_rate_level = max_pow2_rate_level(frame_dt, MPM_MIN_DT_SUB);
        let min_rate_level = particle_world
            .solver_params
            .mpm_block_rate_level_min
            .min(max_rate_level);
        let frame_units = 1_u64 << max_rate_level;
        let base_dt_unit = frame_dt / frame_units as f32;
        let c = sound_speed_mps(&mpm_params).max(1e-4);
        let a_max = mpm_params.gravity.length().max(1e-4);
        let mut active_mask = vec![false; block_count];
        let mut rate_level_by_block = vec![0_u8; block_count];
        let mut step_units_by_block = vec![frame_units; block_count];
        let mut block_time_units = vec![0_u64; block_count];
        let mut owner_block_drift_secs = vec![0.0_f32; block_count];
        let mut rate_initialized = vec![false; block_count];
        let mut scheduled_tick_by_block = vec![None; block_count];
        for &block_index in &active_blocks {
            active_mask[block_index] = true;
            configure_block_rate_level(
                block_index,
                continuum_world,
                grid_hierarchy,
                mpm_block_index_table,
                frame_dt,
                base_dt_unit,
                max_rate_level,
                min_rate_level,
                c,
                a_max,
                &mut rate_level_by_block,
                &mut step_units_by_block,
            );
            rate_initialized[block_index] = true;
        }

        let mut schedule = BinaryHeap::<Reverse<(u64, usize)>>::new();
        for &block_index in &active_blocks {
            schedule.push(Reverse((0, block_index)));
            scheduled_tick_by_block[block_index] = Some(0);
        }
        terrain_boundary_sampler.begin_step();
        while let Some(Reverse((tick, block_index))) = schedule.pop() {
            if tick >= frame_units {
                continue;
            }
            if scheduled_tick_by_block
                .get(block_index)
                .copied()
                .flatten()
                != Some(tick)
            {
                continue;
            }
            scheduled_tick_by_block[block_index] = None;
            if block_time_units[block_index] != tick {
                continue;
            }

            owner_block_drift_secs.fill(0.0);
            for block in 0..block_count {
                let drift_units = tick.saturating_sub(block_time_units[block]);
                owner_block_drift_secs[block] = drift_units as f32 * base_dt_unit;
            }
            refresh_block_index_table(
                continuum_world,
                grid_hierarchy,
                mpm_block_index_table,
                &owner_block_drift_secs,
                mpm_params.gravity,
            );
            active_blocks = active_blocks_from_index_table(mpm_block_index_table);
            if active_blocks.is_empty() {
                continue;
            }
            active_mask.fill(false);
            for &active in &active_blocks {
                active_mask[active] = true;
                if !rate_initialized[active] {
                    configure_block_rate_level(
                        active,
                        continuum_world,
                        grid_hierarchy,
                        mpm_block_index_table,
                        frame_dt,
                        base_dt_unit,
                        max_rate_level,
                        min_rate_level,
                        c,
                        a_max,
                        &mut rate_level_by_block,
                        &mut step_units_by_block,
                    );
                    rate_initialized[active] = true;
                }
                let next_tick = block_time_units[active];
                if next_tick >= frame_units || scheduled_tick_by_block[active].is_some() {
                    continue;
                }
                schedule.push(Reverse((next_tick, active)));
                scheduled_tick_by_block[active] = Some(next_tick);
            }
            if !active_mask[block_index] || block_time_units[block_index] != tick {
                continue;
            }
            let mut due_blocks: Vec<usize> = active_blocks
                .iter()
                .copied()
                .filter(|&active| block_time_units[active] == tick)
                .collect();
            if due_blocks.is_empty() {
                continue;
            }
            due_blocks.sort_unstable();
            due_blocks.dedup();
            for &due_block in &due_blocks {
                scheduled_tick_by_block[due_block] = None;
            }
            let remaining_units = frame_units.saturating_sub(tick).max(1);
            let mut applied_step_units = Vec::with_capacity(due_blocks.len());
            for &due_block in &due_blocks {
                let step_units = step_units_by_block[due_block].min(remaining_units).max(1);
                applied_step_units.push((due_block, step_units));
                if let Some(block_mut) = grid_hierarchy.blocks_mut().get_mut(due_block) {
                    block_mut.dt_b =
                        (step_units as f32 * base_dt_unit).clamp(MPM_MIN_DT_SUB, frame_dt);
                }
            }

            let _ = step_block_set_coupled(
                continuum_world,
                grid_hierarchy,
                mpm_block_index_table,
                &active_blocks,
                &due_blocks,
                &owner_block_drift_secs,
                Some(terrain_world),
                Some(terrain_boundary_sampler),
                &mpm_params,
                &terrain_boundary_params,
                parallel_enabled,
            );

            for (due_block, step_units) in applied_step_units {
                block_time_units[due_block] = block_time_units[due_block]
                    .saturating_add(step_units)
                    .min(frame_units);
                let Some(block) = grid_hierarchy.blocks().get(due_block) else {
                    continue;
                };
                let dt_now = block.dt_b.max(MPM_MIN_DT_SUB);
                let mut cfl_params = mpm_params;
                cfl_params.dt = dt_now;
                let u_max =
                    estimate_block_max_speed(continuum_world, mpm_block_index_table, due_block);
                if cfl_violated(&cfl_params, block.h_b, u_max) {
                    let next_level = rate_level_by_block[due_block]
                        .saturating_add(1)
                        .min(max_rate_level);
                    if next_level != rate_level_by_block[due_block] {
                        rate_level_by_block[due_block] = next_level;
                        step_units_by_block[due_block] = 1_u64 << (max_rate_level - next_level);
                    }
                }
                let next_tick = block_time_units[due_block];
                if next_tick < frame_units
                    && active_mask[due_block]
                    && scheduled_tick_by_block[due_block].is_none()
                {
                    schedule.push(Reverse((next_tick, due_block)));
                    scheduled_tick_by_block[due_block] = Some(next_tick);
                }
            }
        }
        terrain_boundary_sampler.end_step();
        if !sync_continuum_to_particle_world(particle_world, continuum_world) {
            let _ =
                rebuild_continuum_from_particle_world(particle_world, continuum_world, &mpm_params);
            refresh_block_index_table(
                continuum_world,
                grid_hierarchy,
                mpm_block_index_table,
                &zero_owner_block_drift_secs,
                mpm_params.gravity,
            );
            let _ = sync_continuum_to_particle_world(particle_world, continuum_world);
        }
        let particle_step_secs = particle_step_start.elapsed().as_secs_f64();
        let particle_step_cpu_secs = (process_cpu_time_seconds()
            .unwrap_or(particle_step_cpu_start)
            - particle_step_cpu_start)
            .max(0.0);
        return StepSimulationTiming {
            particle_step_secs,
            particle_step_cpu_secs,
            ..StepSimulationTiming::default()
        };
    }
    particle_world.set_parallel_enabled(parallel_enabled);
    let particle_step_start = Instant::now();
    let particle_step_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
    let particle_breakdown = {
        let _span = tracing::info_span!("physics::particle_step").entered();
        particle_world.step_if_running(terrain_world, object_field, object_world, true)
    };
    let particle_step_secs = particle_step_start.elapsed().as_secs_f64();
    let particle_step_cpu_secs = (process_cpu_time_seconds().unwrap_or(particle_step_cpu_start)
        - particle_step_cpu_start)
        .max(0.0);
    let terrain_fracture_start = Instant::now();
    let terrain_fracture_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
    {
        let _span = tracing::info_span!("physics::terrain_fracture_commit").entered();
        if particle_world.apply_pending_terrain_fractures(terrain_world, object_world) {
            terrain_world.rebuild_static_particles_if_dirty(terrain_boundary_radius_m);
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

#[cfg(test)]
mod tests {
    use super::{max_pow2_rate_level, quantize_pow2_rate_level};

    #[test]
    fn quantized_rate_level_is_power_of_two_division_of_frame_dt() {
        let frame_dt = 1.0 / 60.0;
        let max_level = max_pow2_rate_level(frame_dt, 1.0e-4);
        let target_dt = frame_dt * 0.19;
        let level = quantize_pow2_rate_level(frame_dt, target_dt, max_level);
        let quantized_dt = frame_dt / (1_u32 << level) as f32;
        assert!(quantized_dt <= target_dt + 1e-7);
        if level > 0 {
            let coarser = frame_dt / (1_u32 << (level - 1)) as f32;
            assert!(coarser > target_dt);
        }
    }

    #[test]
    fn rate_level_respects_min_dt_constraint() {
        let frame_dt = 1.0 / 60.0;
        let max_level = max_pow2_rate_level(frame_dt, 1.0e-4);
        let finest_dt = frame_dt / (1_u32 << max_level) as f32;
        assert!(finest_dt >= 1.0e-4 - 1e-7);
        let finer = frame_dt / (1_u32 << (max_level + 1)) as f32;
        assert!(finer < 1.0e-4 + 1e-7);
    }

    #[test]
    fn quantized_rate_level_respects_configured_min_level() {
        let frame_dt = 1.0 / 60.0;
        let max_level = max_pow2_rate_level(frame_dt, 1.0e-4);
        let min_level = 3_u8.min(max_level);
        let cfl_level = quantize_pow2_rate_level(frame_dt, frame_dt * 0.92, max_level);
        assert_eq!(cfl_level, 1);
        let applied_level = cfl_level.max(min_level);
        assert_eq!(applied_level, min_level);
        let applied_dt = frame_dt / (1_u32 << applied_level) as f32;
        let min_dt = frame_dt / (1_u32 << min_level) as f32;
        assert!((applied_dt - min_dt).abs() <= 1e-8);
    }
}
