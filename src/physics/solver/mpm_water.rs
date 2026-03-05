use bevy::prelude::*;

use crate::physics::world::continuum::{ContinuumParticleWorld, ContinuumPhase};
use crate::physics::world::particle::{ParticleMaterial, ParticleWorld};

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

pub fn continuum_phase_for_particle(material: ParticleMaterial) -> Option<ContinuumPhase> {
    match material {
        ParticleMaterial::WaterLiquid => Some(ContinuumPhase::Water),
        ParticleMaterial::StoneGranular => Some(ContinuumPhase::GranularSoil),
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
            0.0,
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
