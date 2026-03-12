use crate::physics::world::constants::CELL_SIZE_M;

use super::types::{
    GranularMaterialFamilyParams, MaterialParams, ParticleMaterial, ParticleMaterialCountParams,
    ParticleMaterialMassScaleParams, ParticlePhaseProperties, TerrainBreakOutput, TerrainBreakParams,
    TerrainBreakRule, TerrainMaterial, WaterMaterialParams,
};

pub const DEFAULT_MATERIAL_PARAMS: MaterialParams = MaterialParams {
    rest_density: 1000.0,
    particle_speed_limit_mps: 20.0,
    h_water_over_dx: 2.0,
    water: WaterMaterialParams {
        alpha_apic: 0.95,
        boundary_friction: 0.3,
    },
    soil_family: GranularMaterialFamilyParams {
        alpha_apic: 0.78,
        boundary_friction: 1.6,
        youngs_modulus_pa: 2.0e4,
        poisson_ratio: 0.28,
        friction_deg: 52.0,
        cohesion_pa: 400.0,
        hardening: 1.0,
    },
    sand_family: GranularMaterialFamilyParams {
        alpha_apic: 0.78,
        boundary_friction: 1.6,
        youngs_modulus_pa: 1.8e4,
        poisson_ratio: 0.25,
        friction_deg: 48.0,
        cohesion_pa: 120.0,
        hardening: 0.5,
    },
    particle_counts: ParticleMaterialCountParams {
        water_liquid: 16,
        stone_solid: 1,
        stone_granular: 16,
        soil_solid: 1,
        soil_granular: 16,
        sand_solid: 1,
        sand_granular: 16,
        grass_solid: 1,
        grass_granular: 16,
    },
    particle_mass_scales: ParticleMaterialMassScaleParams {
        water_liquid: 1.0,
        stone_solid: 3.0,
        stone_granular: 1.8,
        soil_solid: 1.9,
        soil_granular: 1.5,
        sand_solid: 1.7,
        sand_granular: 2.6,
        grass_solid: 1.9,
        grass_granular: 1.5,
    },
    granular_tensile_clamp: 0.0,
    enable_granular_to_solid_reconversion: false,
    terrain_break: TerrainBreakParams {
        stone: TerrainBreakRule {
            output_count: 1,
            outputs: [
                TerrainBreakOutput {
                    material: ParticleMaterial::StoneGranular,
                    count_per_cell: 16,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::StoneGranular,
                    count_per_cell: 0,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::StoneGranular,
                    count_per_cell: 0,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::StoneGranular,
                    count_per_cell: 0,
                },
            ],
        },
        soil: TerrainBreakRule {
            output_count: 1,
            outputs: [
                TerrainBreakOutput {
                    material: ParticleMaterial::SoilGranular,
                    count_per_cell: 16,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::SoilGranular,
                    count_per_cell: 0,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::SoilGranular,
                    count_per_cell: 0,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::SoilGranular,
                    count_per_cell: 0,
                },
            ],
        },
        sand: TerrainBreakRule {
            output_count: 1,
            outputs: [
                TerrainBreakOutput {
                    material: ParticleMaterial::SandGranular,
                    count_per_cell: 16,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::SandGranular,
                    count_per_cell: 0,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::SandGranular,
                    count_per_cell: 0,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::SandGranular,
                    count_per_cell: 0,
                },
            ],
        },
        grass: TerrainBreakRule {
            output_count: 2,
            outputs: [
                TerrainBreakOutput {
                    material: ParticleMaterial::GrassGranular,
                    count_per_cell: 4,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::SoilGranular,
                    count_per_cell: 12,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::GrassGranular,
                    count_per_cell: 0,
                },
                TerrainBreakOutput {
                    material: ParticleMaterial::GrassGranular,
                    count_per_cell: 0,
                },
            ],
        },
    },
};

impl Default for MaterialParams {
    fn default() -> Self {
        DEFAULT_MATERIAL_PARAMS
    }
}

const fn square_grid_particles_axis(particles_per_cell: u32) -> f32 {
    match particles_per_cell {
        1 => 1.0,
        4 => 2.0,
        9 => 3.0,
        16 => 4.0,
        25 => 5.0,
        36 => 6.0,
        64 => 8.0,
        256 => 16.0,
        _ => 1.0,
    }
}

const fn particle_spacing_from_particles_per_cell(particles_per_cell: u32) -> f32 {
    CELL_SIZE_M / square_grid_particles_axis(particles_per_cell)
}

pub const fn particle_spacing_m(params: MaterialParams) -> f32 {
    particle_spacing_from_particles_per_cell(params.particle_counts.water_liquid)
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
    particles_per_cell_with_params(DEFAULT_MATERIAL_PARAMS, material)
}

pub const fn particles_per_cell_with_params(
    params: MaterialParams,
    material: ParticleMaterial,
) -> u32 {
    match material {
        ParticleMaterial::WaterLiquid => params.particle_counts.water_liquid,
        ParticleMaterial::StoneSolid => params.particle_counts.stone_solid,
        ParticleMaterial::StoneGranular => params.particle_counts.stone_granular,
        ParticleMaterial::SoilSolid => params.particle_counts.soil_solid,
        ParticleMaterial::SoilGranular => params.particle_counts.soil_granular,
        ParticleMaterial::SandSolid => params.particle_counts.sand_solid,
        ParticleMaterial::SandGranular => params.particle_counts.sand_granular,
        ParticleMaterial::GrassSolid => params.particle_counts.grass_solid,
        ParticleMaterial::GrassGranular => params.particle_counts.grass_granular,
    }
}

pub const fn terrain_break_rule(
    params: MaterialParams,
    material: TerrainMaterial,
) -> TerrainBreakRule {
    match material {
        TerrainMaterial::Stone => params.terrain_break.stone,
        TerrainMaterial::Soil => params.terrain_break.soil,
        TerrainMaterial::Sand => params.terrain_break.sand,
        TerrainMaterial::Grass => params.terrain_break.grass,
    }
}

const fn particle_mass_scale_with_params(
    params: MaterialParams,
    material: ParticleMaterial,
) -> f32 {
    match material {
        ParticleMaterial::WaterLiquid => params.particle_mass_scales.water_liquid,
        ParticleMaterial::StoneSolid => params.particle_mass_scales.stone_solid,
        ParticleMaterial::StoneGranular => params.particle_mass_scales.stone_granular,
        ParticleMaterial::SoilSolid => params.particle_mass_scales.soil_solid,
        ParticleMaterial::SoilGranular => params.particle_mass_scales.soil_granular,
        ParticleMaterial::SandSolid => params.particle_mass_scales.sand_solid,
        ParticleMaterial::SandGranular => params.particle_mass_scales.sand_granular,
        ParticleMaterial::GrassSolid => params.particle_mass_scales.grass_solid,
        ParticleMaterial::GrassGranular => params.particle_mass_scales.grass_granular,
    }
}

pub const fn particle_properties(material: ParticleMaterial) -> ParticlePhaseProperties {
    particle_properties_with_params(DEFAULT_MATERIAL_PARAMS, material)
}

pub const fn particle_properties_with_params(
    params: MaterialParams,
    material: ParticleMaterial,
) -> ParticlePhaseProperties {
    let particles_per_cell = particles_per_cell_with_params(params, material);
    let cell_mass =
        params.rest_density * CELL_SIZE_M * CELL_SIZE_M * particle_mass_scale_with_params(params, material);
    ParticlePhaseProperties {
        particles_per_cell,
        mass: particle_mass_from_cell_mass(cell_mass, particles_per_cell),
        radius_m: particle_radius_from_particles_per_cell(particles_per_cell),
    }
}
