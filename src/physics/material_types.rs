use bevy::prelude::Resource;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialKind {
    Water,
    Stone,
    Soil,
    Sand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialForm {
    Liquid,
    Solid,
    Granular,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TerrainMaterial {
    Stone,
    Soil,
    Sand,
}

impl TerrainMaterial {
    pub fn kind(self) -> MaterialKind {
        match self {
            TerrainMaterial::Stone => MaterialKind::Stone,
            TerrainMaterial::Soil => MaterialKind::Soil,
            TerrainMaterial::Sand => MaterialKind::Sand,
        }
    }
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

impl ParticleMaterial {
    pub fn kind(self) -> MaterialKind {
        match self {
            ParticleMaterial::WaterLiquid => MaterialKind::Water,
            ParticleMaterial::StoneSolid | ParticleMaterial::StoneGranular => MaterialKind::Stone,
            ParticleMaterial::SoilSolid | ParticleMaterial::SoilGranular => MaterialKind::Soil,
            ParticleMaterial::SandSolid | ParticleMaterial::SandGranular => MaterialKind::Sand,
        }
    }

    pub fn form(self) -> MaterialForm {
        match self {
            ParticleMaterial::WaterLiquid => MaterialForm::Liquid,
            ParticleMaterial::StoneSolid
            | ParticleMaterial::SoilSolid
            | ParticleMaterial::SandSolid => MaterialForm::Solid,
            ParticleMaterial::StoneGranular
            | ParticleMaterial::SoilGranular
            | ParticleMaterial::SandGranular => MaterialForm::Granular,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParticlePhaseProperties {
    pub particles_per_cell: u32,
    pub mass: f32,
    pub radius_m: f32,
    pub terrain_push_radius_m: f32,
    pub terrain_repulsion_stiffness: f32,
    pub object_push_radius_m: f32,
    pub object_repulsion_stiffness: f32,
    pub friction_static: f32,
    pub friction_dynamic: f32,
    pub contact_restitution: f32,
    pub xsph_viscosity: f32,
    pub apply_contact_velocity_response: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct SolidBreakProperties {
    pub break_collision_impulse_threshold: f32,
    pub break_strain_threshold: f32,
    pub fracture_to: Option<ParticleMaterial>,
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialProperties {
    pub liquid: Option<ParticlePhaseProperties>,
    pub solid: Option<ParticlePhaseProperties>,
    pub granular: Option<ParticlePhaseProperties>,
    pub solid_break: Option<SolidBreakProperties>,
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
