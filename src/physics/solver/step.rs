use super::mpm_water::{
    MpmWaterParams, is_mpm_managed_particle, rebuild_continuum_from_particle_world,
};
use super::types::StepSimulationTiming;
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::particle::ParticleWorld;
use crate::physics::world::terrain::TerrainWorld;

pub(crate) fn step_simulation_once(
    _terrain_world: &mut TerrainWorld,
    particle_world: &mut ParticleWorld,
    continuum_world: &mut ContinuumParticleWorld,
    _parallel_enabled: bool,
    _terrain_boundary_radius_m: f32,
) -> StepSimulationTiming {
    // CPU-side MPM has been removed. Keep continuum/grid bookkeeping consistent
    // so GPU MPM upload sources do not retain stale data when MPM-managed particles disappear.
    let mpm_particle_count = particle_world
        .materials()
        .iter()
        .filter(|&&material| is_mpm_managed_particle(material))
        .count();
    if mpm_particle_count == 0 {
        continuum_world.clear();
    } else if continuum_world.len() != mpm_particle_count {
        let _ = rebuild_continuum_from_particle_world(
            particle_world,
            continuum_world,
            &MpmWaterParams {
                dt: particle_world.solver_params.fixed_dt,
                gravity: particle_world.solver_params.gravity_mps2,
                ..Default::default()
            },
        );
    }

    // Pure GPU-lean mode: CPU particle stepping/fracture commit is skipped here.
    StepSimulationTiming::default()
}
