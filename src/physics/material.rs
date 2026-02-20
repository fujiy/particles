use super::terrain::CELL_SIZE_M;

pub const REST_DENSITY: f32 = 1000.0;
pub const PARTICLE_SPEED_LIMIT_MPS: f32 = 20.0;
pub const H_WATER_OVER_DX: f32 = 2.0;

pub const WATER_LIQUID_PARTICLES_PER_CELL: u32 = 4;
pub const SOLID_PARTICLES_PER_CELL: u32 = 1;
pub const STONE_GRANULAR_PARTICLES_PER_CELL: u32 = 4;
pub const SOIL_GRANULAR_PARTICLES_PER_CELL: u32 = 4;
pub const SAND_GRANULAR_PARTICLES_PER_CELL: u32 = 4;
pub const ENABLE_GRANULAR_TO_SOLID_RECONVERSION: bool = false;

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

pub const PARTICLE_SPACING_M: f32 =
    particle_spacing_from_particles_per_cell(WATER_LIQUID_PARTICLES_PER_CELL);
pub const PARTICLE_RADIUS_M: f32 = PARTICLE_SPACING_M * 0.5;
pub const WATER_KERNEL_RADIUS_M: f32 = PARTICLE_SPACING_M * H_WATER_OVER_DX;
pub const WATER_LIQUID_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M;
pub const STONE_SOLID_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 3.0;
pub const STONE_GRANULAR_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.8;
pub const SOIL_SOLID_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.9;
pub const SOIL_GRANULAR_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.5;
pub const SAND_SOLID_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.7;
pub const SAND_GRANULAR_CELL_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M * 1.3;
pub const TERRAIN_BOUNDARY_RADIUS_M: f32 = PARTICLE_RADIUS_M + CELL_SIZE_M * 0.5;

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

const WATER_LIQUID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: WATER_LIQUID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(WATER_LIQUID_CELL_MASS, WATER_LIQUID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(WATER_LIQUID_PARTICLES_PER_CELL),
    terrain_push_radius_m: particle_radius_from_particles_per_cell(WATER_LIQUID_PARTICLES_PER_CELL),
    terrain_repulsion_stiffness: 1.0,
    object_push_radius_m: particle_radius_from_particles_per_cell(WATER_LIQUID_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.70,
    friction_static: 0.0,
    friction_dynamic: 0.0,
    contact_restitution: 0.0,
    xsph_viscosity: 0.01,
    apply_contact_velocity_response: false,
};

const STONE_SOLID_PHASE: ParticlePhaseProperties = ParticlePhaseProperties {
    particles_per_cell: SOLID_PARTICLES_PER_CELL,
    mass: particle_mass_from_cell_mass(STONE_SOLID_CELL_MASS, SOLID_PARTICLES_PER_CELL),
    radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    terrain_push_radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    terrain_repulsion_stiffness: 0.35,
    object_push_radius_m: particle_radius_from_particles_per_cell(SOLID_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.35,
    friction_static: 0.90,
    friction_dynamic: 0.75,
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
    terrain_repulsion_stiffness: 0.32,
    object_push_radius_m: particle_radius_from_particles_per_cell(
        STONE_GRANULAR_PARTICLES_PER_CELL,
    ),
    object_repulsion_stiffness: 0.28,
    friction_static: 0.78,
    friction_dynamic: 0.66,
    contact_restitution: 0.02,
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
    friction_static: 0.92,
    friction_dynamic: 0.78,
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
    terrain_repulsion_stiffness: 0.31,
    object_push_radius_m: particle_radius_from_particles_per_cell(SOIL_GRANULAR_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.24,
    friction_static: 0.88,
    friction_dynamic: 0.74,
    contact_restitution: 0.01,
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
    friction_static: 0.90,
    friction_dynamic: 0.76,
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
    terrain_repulsion_stiffness: 0.30,
    object_push_radius_m: particle_radius_from_particles_per_cell(SAND_GRANULAR_PARTICLES_PER_CELL),
    object_repulsion_stiffness: 0.22,
    friction_static: 0.86,
    friction_dynamic: 0.72,
    contact_restitution: 0.01,
    xsph_viscosity: 0.0,
    apply_contact_velocity_response: true,
};

const STONE_SOLID_BREAK: SolidBreakProperties = SolidBreakProperties {
    break_collision_impulse_threshold: 5000.0,
    break_strain_threshold: 1.0,
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
