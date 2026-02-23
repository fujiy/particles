use crate::physics::world::constants::CELL_SIZE_M;

use super::material_types::{
    MaterialForm, MaterialKind, MaterialParams, MaterialProperties, ParticleMaterial,
    ParticlePhaseProperties, SolidBreakProperties, TerrainMaterial,
};

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

const WATER_LIQUID_PARTICLES_PER_CELL: u32 = 4;
const SOLID_PARTICLES_PER_CELL: u32 = 1;
const STONE_GRANULAR_PARTICLES_PER_CELL: u32 = 4;
const SOIL_GRANULAR_PARTICLES_PER_CELL: u32 = 4;
const SAND_GRANULAR_PARTICLES_PER_CELL: u32 = 4;

const fn square_grid_particles_axis(particles_per_cell: u32) -> f32 {
    match particles_per_cell {
        1 => 1.0,
        4 => 2.0,
        9 => 3.0,
        16 => 4.0,
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
const SAND_GRANULAR_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.3;

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
    terrain_push_radius_m: particle_radius_from_particles_per_cell(WATER_LIQUID_PARTICLES_PER_CELL),
    terrain_repulsion_stiffness: 1.0,
    object_push_radius_m: particle_radius_from_particles_per_cell(WATER_LIQUID_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.70,
    friction_static: 0.01,
    friction_dynamic: 0.01,
    contact_restitution: 0.0,
    xsph_viscosity: 0.01,
    apply_contact_velocity_response: false,
};

const STONE_SOLID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOLID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(STONE_SOLID_CELL_MASS, SOLID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    terrain_push_radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    terrain_repulsion_stiffness: 0.85,
    object_push_radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.35,
    friction_static: 3.0,
    friction_dynamic: 2.0,
    contact_restitution: 0.00,
    xsph_viscosity: 0.0,
    apply_contact_velocity_response: true,
};

const STONE_GRANULAR_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: STONE_GRANULAR_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(STONE_GRANULAR_CELL_MASS, STONE_GRANULAR_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(STONE_GRANULAR_PARTICLES_PER_CELL),
    terrain_push_radius_m: particle_radius_from_particles_per_cell(
        STONE_GRANULAR_PARTICLES_PER_CELL,
    ),
    terrain_repulsion_stiffness: 0.28,
    object_push_radius_m: particle_radius_from_particles_per_cell(
        STONE_GRANULAR_PARTICLES_PER_CELL,
    ),
    object_repulsion_stiffness: 0.22,
    friction_static: 10.0,
    friction_dynamic: 8.0,
    contact_restitution: 0.00,
    xsph_viscosity: 0.0,
    apply_contact_velocity_response: true,
};

const SOIL_SOLID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOLID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(SOIL_SOLID_CELL_MASS, SOLID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    terrain_push_radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    terrain_repulsion_stiffness: 0.33,
    object_push_radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.26,
    friction_static: 10.0,
    friction_dynamic: 8.0,
    contact_restitution: 0.00,
    xsph_viscosity: 0.0,
    apply_contact_velocity_response: true,
};

const SOIL_GRANULAR_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOIL_GRANULAR_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(SOIL_GRANULAR_CELL_MASS, SOIL_GRANULAR_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOIL_GRANULAR_PARTICLES_PER_CELL),
    terrain_push_radius_m: particle_radius_from_particles_per_cell(
        SOIL_GRANULAR_PARTICLES_PER_CELL,
    ),
    terrain_repulsion_stiffness: 0.26,
    object_push_radius_m: particle_radius_from_particles_per_cell(SOIL_GRANULAR_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.20,
    friction_static: 20.0,
    friction_dynamic: 16.0,
    contact_restitution: 0.00,
    xsph_viscosity: 0.0,
    apply_contact_velocity_response: true,
};

const SAND_SOLID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOLID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(SAND_SOLID_CELL_MASS, SOLID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    terrain_push_radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    terrain_repulsion_stiffness: 0.31,
    object_push_radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.24,
    friction_static: 10.0,
    friction_dynamic: 8.0,
    contact_restitution: 0.00,
    xsph_viscosity: 0.0,
    apply_contact_velocity_response: true,
};

const SAND_GRANULAR_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SAND_GRANULAR_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(SAND_GRANULAR_CELL_MASS, SAND_GRANULAR_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SAND_GRANULAR_PARTICLES_PER_CELL),
    terrain_push_radius_m: particle_radius_from_particles_per_cell(
        SAND_GRANULAR_PARTICLES_PER_CELL,
    ),
    terrain_repulsion_stiffness: 0.24,
    object_push_radius_m: particle_radius_from_particles_per_cell(SAND_GRANULAR_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.18,
    friction_static: 10.0,
    friction_dynamic: 8.0,
    contact_restitution: 0.00,
    xsph_viscosity: 0.0,
    apply_contact_velocity_response: true,
};

const STONE_SOLID_BREAK: SolidBreakProperties = SolidBreakProperties {
    break_collision_impulse_threshold: 12000.0,
    break_strain_threshold: 1.6,
    fracture_to: Some(ParticleMaterial::StoneGranular),
};

const SOIL_SOLID_BREAK: SolidBreakProperties = SolidBreakProperties {
    break_collision_impulse_threshold: 50.0,
    break_strain_threshold: 0.10,
    fracture_to: Some(ParticleMaterial::SoilGranular),
};

const SAND_SOLID_BREAK: SolidBreakProperties = SolidBreakProperties {
    break_collision_impulse_threshold: 10.0,
    break_strain_threshold: 0.05,
    fracture_to: Some(ParticleMaterial::SandGranular),
};

const WATER_PROPERTIES: MaterialProperties = MaterialProperties {
    liquid: Some(WATER_LIQUID_PHASE),
    solid: None,
    granular: None,
    solid_break: None,
};

const STONE_PROPERTIES: MaterialProperties = MaterialProperties {
    liquid: None,
    solid: Some(STONE_SOLID_PHASE),
    granular: Some(STONE_GRANULAR_PHASE),
    solid_break: Some(STONE_SOLID_BREAK),
};

const SOIL_PROPERTIES: MaterialProperties = MaterialProperties {
    liquid: None,
    solid: Some(SOIL_SOLID_PHASE),
    granular: Some(SOIL_GRANULAR_PHASE),
    solid_break: Some(SOIL_SOLID_BREAK),
};

const SAND_PROPERTIES: MaterialProperties = MaterialProperties {
    liquid: None,
    solid: Some(SAND_SOLID_PHASE),
    granular: Some(SAND_GRANULAR_PHASE),
    solid_break: Some(SAND_SOLID_BREAK),
};

pub fn material_properties(kind: MaterialKind) -> &'static MaterialProperties {
    match kind {
        MaterialKind::Water => &WATER_PROPERTIES,
        MaterialKind::Stone => &STONE_PROPERTIES,
        MaterialKind::Soil => &SOIL_PROPERTIES,
        MaterialKind::Sand => &SAND_PROPERTIES,
    }
}

pub fn particle_properties(material: ParticleMaterial) -> &'static ParticlePhaseProperties {
    let material_props = material_properties(material.kind());
    match material.form() {
        MaterialForm::Liquid => material_props
            .liquid
            .as_ref()
            .expect("liquid properties must exist for liquid material"),
        MaterialForm::Solid => material_props
            .solid
            .as_ref()
            .expect("solid properties must exist for solid material"),
        MaterialForm::Granular => material_props
            .granular
            .as_ref()
            .expect("granular properties must exist for granular material"),
    }
}

pub fn solid_break_properties(material: ParticleMaterial) -> Option<&'static SolidBreakProperties> {
    if material.form() != MaterialForm::Solid {
        return None;
    }
    material_properties(material.kind()).solid_break.as_ref()
}

pub fn terrain_break_collision_impulse_threshold(material: TerrainMaterial) -> f32 {
    terrain_solid_break_properties(material)
        .map(|props| props.break_collision_impulse_threshold)
        .unwrap_or(f32::INFINITY)
}

pub fn terrain_fracture_particle(material: TerrainMaterial) -> Option<ParticleMaterial> {
    terrain_solid_break_properties(material).and_then(|props| props.fracture_to)
}

pub fn terrain_solid_particle(material: TerrainMaterial) -> Option<ParticleMaterial> {
    match material {
        TerrainMaterial::Stone => Some(ParticleMaterial::StoneSolid),
        TerrainMaterial::Soil => Some(ParticleMaterial::SoilSolid),
        TerrainMaterial::Sand => Some(ParticleMaterial::SandSolid),
    }
}

fn terrain_solid_break_properties(
    material: TerrainMaterial,
) -> Option<&'static SolidBreakProperties> {
    terrain_solid_particle(material).and_then(solid_break_properties)
}
