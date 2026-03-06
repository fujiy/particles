use crate::physics::world::constants::CELL_SIZE_M;

use super::types::{MaterialParams, ParticleMaterial, ParticlePhaseProperties};

pub const DEFAULT_MATERIAL_PARAMS: MaterialParams = MaterialParams {
    rest_density: 1000.0,
    particle_speed_limit_mps: 20.0,
    h_water_over_dx: 2.0,
    water_liquid_particles_per_cell: 4,
    solid_particles_per_cell: 1,
    stone_granular_particles_per_cell: 4,
    soil_granular_particles_per_cell: 4,
    sand_granular_particles_per_cell: 4,
    enable_granular_to_solid_reconversion: false,
};

impl Default for MaterialParams {
    fn default() -> Self {
        DEFAULT_MATERIAL_PARAMS
    }
}

const REST_DENSITY: f32 = 1000.0;

const PARTICLES_PER_CELL: u32 = 16;
const WATER_LIQUID_PARTICLES_PER_CELL: u32 = PARTICLES_PER_CELL;
const SOLID_PARTICLES_PER_CELL: u32 = 1;
const STONE_GRANULAR_PARTICLES_PER_CELL: u32 = PARTICLES_PER_CELL;
const SOIL_GRANULAR_PARTICLES_PER_CELL: u32 = PARTICLES_PER_CELL;
const SAND_GRANULAR_PARTICLES_PER_CELL: u32 = PARTICLES_PER_CELL;

const fn square_grid_particles_axis(particles_per_cell: u32) -> f32 {
    match particles_per_cell {
        1 => 1.0,
        4 => 2.0,
        9 => 3.0,
        16 => 4.0,
        256 => 16.0,
        _ => 1.0,
    }
}

const fn particle_spacing_from_particles_per_cell(particles_per_cell: u32) -> f32 {
    CELL_SIZE_M / square_grid_particles_axis(particles_per_cell)
}

pub const fn particle_spacing_m(params: MaterialParams) -> f32 {
    particle_spacing_from_particles_per_cell(params.water_liquid_particles_per_cell)
}

pub const fn particle_radius_m(params: MaterialParams) -> f32 {
    particle_spacing_m(params) * 0.5
}

pub const fn water_kernel_radius_m(params: MaterialParams) -> f32 {
    particle_spacing_m(params) * params.h_water_over_dx
}

pub const fn terrain_boundary_radius_m(params: MaterialParams) -> f32 {
    particle_radius_m(params) + CELL_SIZE_M * 0.5
}

const WATER_LIQUID_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M;
const STONE_SOLID_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 3.0;
const STONE_GRANULAR_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.8;
const SOIL_SOLID_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.9;
const SOIL_GRANULAR_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.5;
const SAND_SOLID_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.7;
const SAND_GRANULAR_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 2.6;

const fn particle_radius_from_particles_per_cell(particles_per_cell: u32) -> f32 {
    particle_spacing_from_particles_per_cell(particles_per_cell) * 0.5
}

const fn particle_mass_from_cell_mass(cell_mass: f32, particles_per_cell: u32) -> f32 {
    let denom = if particles_per_cell == 0 {
        1
    } else {
        particles_per_cell
    };
    cell_mass / denom as f32
}

pub const fn particles_per_cell(material: ParticleMaterial) -> u32 {
    match material {
        ParticleMaterial::WaterLiquid => WATER_LIQUID_PARTICLES_PER_CELL,
        ParticleMaterial::StoneSolid => SOLID_PARTICLES_PER_CELL,
        ParticleMaterial::StoneGranular => STONE_GRANULAR_PARTICLES_PER_CELL,
        ParticleMaterial::SoilSolid => SOLID_PARTICLES_PER_CELL,
        ParticleMaterial::SoilGranular => SOIL_GRANULAR_PARTICLES_PER_CELL,
        ParticleMaterial::SandSolid => SOLID_PARTICLES_PER_CELL,
        ParticleMaterial::SandGranular => SAND_GRANULAR_PARTICLES_PER_CELL,
    }
}

const WATER_LIQUID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: WATER_LIQUID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(WATER_LIQUID_CELL_MASS, WATER_LIQUID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(WATER_LIQUID_PARTICLES_PER_CELL),
};

const STONE_SOLID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOLID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(STONE_SOLID_CELL_MASS, SOLID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
};

const STONE_GRANULAR_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: STONE_GRANULAR_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(STONE_GRANULAR_CELL_MASS, STONE_GRANULAR_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(STONE_GRANULAR_PARTICLES_PER_CELL),
};

const SOIL_SOLID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOLID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(SOIL_SOLID_CELL_MASS, SOLID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
};

const SOIL_GRANULAR_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOIL_GRANULAR_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(SOIL_GRANULAR_CELL_MASS, SOIL_GRANULAR_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOIL_GRANULAR_PARTICLES_PER_CELL),
};

const SAND_SOLID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOLID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(SAND_SOLID_CELL_MASS, SOLID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
};

const SAND_GRANULAR_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SAND_GRANULAR_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(SAND_GRANULAR_CELL_MASS, SAND_GRANULAR_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SAND_GRANULAR_PARTICLES_PER_CELL),
};

pub fn particle_properties(material: ParticleMaterial) -> &'static ParticlePhaseProperties {
    match material {
        ParticleMaterial::WaterLiquid => &WATER_LIQUID_PHASE,
        ParticleMaterial::StoneSolid => &STONE_SOLID_PHASE,
        ParticleMaterial::StoneGranular => &STONE_GRANULAR_PHASE,
        ParticleMaterial::SoilSolid => &SOIL_SOLID_PHASE,
        ParticleMaterial::SoilGranular => &SOIL_GRANULAR_PHASE,
        ParticleMaterial::SandSolid => &SAND_SOLID_PHASE,
        ParticleMaterial::SandGranular => &SAND_GRANULAR_PHASE,
    }
}
