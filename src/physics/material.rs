use super::terrain::CELL_SIZE_M;

pub const REST_DENSITY: f32 = 1000.0;
pub const PARTICLE_SPACING_M: f32 = 0.125;
pub const PARTICLE_RADIUS_M: f32 = 0.06;
pub const PARTICLE_SPEED_LIMIT_MPS: f32 = 20.0;
pub const H_WATER_OVER_DX: f32 = 2.0;
pub const WATER_KERNEL_RADIUS_M: f32 = PARTICLE_SPACING_M * H_WATER_OVER_DX;
pub const WATER_PARTICLE_MASS: f32 = REST_DENSITY * PARTICLE_SPACING_M * PARTICLE_SPACING_M;
pub const STONE_PARTICLE_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 3.0;
pub const STONE_PARTICLE_RADIUS_M: f32 = CELL_SIZE_M * 0.5;
pub const TERRAIN_BOUNDARY_RADIUS_M: f32 = PARTICLE_RADIUS_M + CELL_SIZE_M * 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleMaterial {
    Water,
    Stone,
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialProperties {
    pub mass: f32,
    pub radius_m: f32,
    pub terrain_push_radius_m: f32,
    pub terrain_repulsion_stiffness: f32,
    pub object_push_radius_m: f32,
    pub object_repulsion_stiffness: f32,
    pub contact_friction: f32,
    pub contact_restitution: f32,
    pub xsph_viscosity: f32,
    pub apply_contact_velocity_response: bool,
}

const WATER_PROPERTIES: MaterialProperties = MaterialProperties {
    mass: WATER_PARTICLE_MASS,
    radius_m: PARTICLE_RADIUS_M,
    terrain_push_radius_m: PARTICLE_RADIUS_M,
    terrain_repulsion_stiffness: 1.0,
    object_push_radius_m: PARTICLE_RADIUS_M,
    object_repulsion_stiffness: 0.70,
    contact_friction: 0.0,
    contact_restitution: 0.0,
    xsph_viscosity: 0.01,
    apply_contact_velocity_response: false,
};

const STONE_PROPERTIES: MaterialProperties = MaterialProperties {
    mass: STONE_PARTICLE_MASS,
    radius_m: STONE_PARTICLE_RADIUS_M,
    terrain_push_radius_m: STONE_PARTICLE_RADIUS_M,
    terrain_repulsion_stiffness: 0.35,
    object_push_radius_m: STONE_PARTICLE_RADIUS_M,
    object_repulsion_stiffness: 0.35,
    contact_friction: 0.85,
    contact_restitution: 0.00,
    xsph_viscosity: 0.0,
    apply_contact_velocity_response: true,
};

pub fn material_properties(material: ParticleMaterial) -> &'static MaterialProperties {
    match material {
        ParticleMaterial::Water => &WATER_PROPERTIES,
        ParticleMaterial::Stone => &STONE_PROPERTIES,
    }
}
