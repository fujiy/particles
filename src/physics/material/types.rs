use bevy::prelude::Resource;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TerrainMaterial {
    Stone,
    Soil,
    Sand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParticleMaterial {
    WaterLiquid,
    StoneSolid,
    StoneGranular,
    SoilSolid,
    SoilGranular,
    SandSolid,
    SandGranular,
}

#[derive(Clone, Copy, Debug)]
pub struct ParticlePhaseProperties {
    pub particles_per_cell: u32,
    pub mass: f32,
    pub radius_m: f32,
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct MaterialParams {
    pub rest_density: f32,
    pub particle_speed_limit_mps: f32,
    pub h_water_over_dx: f32,
    pub water_liquid_particles_per_cell: u32,
    pub solid_particles_per_cell: u32,
    pub stone_granular_particles_per_cell: u32,
    pub soil_granular_particles_per_cell: u32,
    pub sand_granular_particles_per_cell: u32,
    pub enable_granular_to_solid_reconversion: bool,
}
