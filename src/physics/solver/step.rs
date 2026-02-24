use std::time::Instant;

use bevy::log::tracing;

use super::mpm_water::{
    MpmTerrainBoundaryParams, MpmWaterParams, rebuild_continuum_from_particle_world,
    sound_speed_mps, step_single_rate_coupled, sync_continuum_to_particle_world,
};
use super::terrain_boundary::TerrainBoundarySampler;
use super::types::StepSimulationTiming;
use crate::physics::profiler::process_cpu_time_seconds;
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::grid::GridHierarchy;
use crate::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::{ParticleMaterial, ParticleWorld};
use crate::physics::world::terrain::{CELL_SIZE_M, TerrainWorld};

const MPM_CFL_SAFETY: f32 = 0.35;
const MPM_TARGET_SOUND_SPEED_MPS: f32 = 16.0;
const MPM_TARGET_RHO0: f32 = 1_000.0;
const MPM_MIN_DT_SUB: f32 = 1.0e-4;
const MPM_BOUNDARY_SLOP_SCALE_DIAMETER: f32 = 2.0;

pub(crate) fn step_simulation_once(
    terrain_world: &mut TerrainWorld,
    particle_world: &mut ParticleWorld,
    continuum_world: &mut ContinuumParticleWorld,
    grid_hierarchy: &mut GridHierarchy,
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
    }
    if water_particle_count > 0 {
        if grid_hierarchy.block_count() == 0 {
            return StepSimulationTiming::default();
        }
        let particle_step_start = Instant::now();
        let particle_step_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
        let _span = tracing::info_span!("physics::mpm_water_step").entered();
        let mut mpm_params = MpmWaterParams {
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
            let _ = rebuild_continuum_from_particle_world(
                particle_world,
                continuum_world,
                &mpm_params,
            );
        }
        let h = grid_hierarchy
            .blocks()
            .first()
            .map(|b| b.h_b.max(1e-6))
            .unwrap_or(0.25);
        let c = sound_speed_mps(&mpm_params).max(1e-4);
        let dt_cfl = (MPM_CFL_SAFETY * h / c).max(MPM_MIN_DT_SUB);
        let substeps = (particle_world.solver_params.fixed_dt / dt_cfl).ceil() as usize;
        let substeps = substeps.max(1);
        let dt_sub = particle_world.solver_params.fixed_dt / substeps as f32;
        mpm_params.dt = dt_sub;
        terrain_boundary_sampler.begin_step();
        for _ in 0..substeps {
            let _ = step_single_rate_coupled(
                continuum_world,
                grid_hierarchy,
                Some(terrain_world),
                Some(terrain_boundary_sampler),
                &mpm_params,
                &terrain_boundary_params,
            );
        }
        terrain_boundary_sampler.end_step();
        if !sync_continuum_to_particle_world(particle_world, continuum_world) {
            let _ = rebuild_continuum_from_particle_world(
                particle_world,
                continuum_world,
                &mpm_params,
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
