use std::time::Instant;

use bevy::log::tracing;

use super::mpm_water::{MpmWaterParams, step_single_rate};
use super::types::StepSimulationTiming;
use crate::physics::profiler::process_cpu_time_seconds;
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::grid::GridHierarchy;
use crate::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::ParticleWorld;
use crate::physics::world::terrain::TerrainWorld;

pub(crate) fn step_simulation_once(
    terrain_world: &mut TerrainWorld,
    particle_world: &mut ParticleWorld,
    continuum_world: &mut ContinuumParticleWorld,
    grid_hierarchy: &mut GridHierarchy,
    object_world: &mut ObjectWorld,
    object_field: &mut ObjectPhysicsField,
    parallel_enabled: bool,
    terrain_boundary_radius_m: f32,
) -> StepSimulationTiming {
    if !continuum_world.is_empty() && grid_hierarchy.block_count() > 0 {
        let _span = tracing::info_span!("physics::mpm_water_step").entered();
        let _ = step_single_rate(
            continuum_world,
            grid_hierarchy,
            &MpmWaterParams {
                dt: particle_world.solver_params.fixed_dt,
                gravity: particle_world.solver_params.gravity_mps2,
                ..Default::default()
            },
        );
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
