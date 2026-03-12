use bevy::prelude::{Asset, Resource, TypePath};
use serde::{Deserialize, Serialize};

pub const PARTICLE_HOME_SLOT_MASK: u32 = 0x00FF_FFFF;
pub const PARTICLE_HOME_MATERIAL_SHIFT: u32 = 24;
pub const INVALID_PACKED_PARTICLE_SLOT: u32 = PARTICLE_HOME_SLOT_MASK;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainMaterial {
    Stone,
    Soil,
    Sand,
    Grass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleMaterial {
    WaterLiquid,
    StoneSolid,
    StoneGranular,
    SoilSolid,
    SoilGranular,
    SandSolid,
    SandGranular,
    GrassSolid,
    GrassGranular,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WaterMaterialParams {
    pub alpha_apic: f32,
    pub boundary_friction: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GranularMaterialFamilyParams {
    pub alpha_apic: f32,
    pub boundary_friction: f32,
    pub youngs_modulus_pa: f32,
    pub poisson_ratio: f32,
    pub friction_deg: f32,
    pub cohesion_pa: f32,
    pub hardening: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ParticleMaterialCountParams {
    pub water_liquid: u32,
    pub stone_solid: u32,
    pub stone_granular: u32,
    pub soil_solid: u32,
    pub soil_granular: u32,
    pub sand_solid: u32,
    pub sand_granular: u32,
    pub grass_solid: u32,
    pub grass_granular: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ParticleMaterialMassScaleParams {
    pub water_liquid: f32,
    pub stone_solid: f32,
    pub stone_granular: f32,
    pub soil_solid: f32,
    pub soil_granular: f32,
    pub sand_solid: f32,
    pub sand_granular: f32,
    pub grass_solid: f32,
    pub grass_granular: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TerrainBreakOutput {
    pub material: ParticleMaterial,
    pub count_per_cell: u32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TerrainBreakRule {
    pub output_count: u32,
    pub outputs: [TerrainBreakOutput; 4],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TerrainBreakParams {
    pub stone: TerrainBreakRule,
    pub soil: TerrainBreakRule,
    pub sand: TerrainBreakRule,
    pub grass: TerrainBreakRule,
}

#[derive(Clone, Copy, Debug)]
pub struct ParticlePhaseProperties {
    pub particles_per_cell: u32,
    pub mass: f32,
    pub radius_m: f32,
}

#[derive(Resource, Asset, TypePath, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MaterialParams {
    pub rest_density: f32,
    pub particle_speed_limit_mps: f32,
    pub h_water_over_dx: f32,
    pub water: WaterMaterialParams,
    pub soil_family: GranularMaterialFamilyParams,
    pub sand_family: GranularMaterialFamilyParams,
    pub particle_counts: ParticleMaterialCountParams,
    pub particle_mass_scales: ParticleMaterialMassScaleParams,
    pub granular_tensile_clamp: f32,
    pub enable_granular_to_solid_reconversion: bool,
    pub terrain_break: TerrainBreakParams,
}

impl MaterialParams {
    pub fn validate(&self) -> Result<(), String> {
        macro_rules! check {
            ($val:expr, $name:expr, $lo:expr, $hi:expr) => {
                if !($val >= $lo && $val <= $hi) {
                    return Err(format!(
                        "{}: {} は [{}, {}] の範囲外",
                        $name, $val, $lo, $hi
                    ));
                }
            };
        }

        check!(self.rest_density, "rest_density", 100.0, 10_000.0);
        check!(
            self.particle_speed_limit_mps,
            "particle_speed_limit_mps",
            0.1,
            1_000.0
        );
        check!(self.h_water_over_dx, "h_water_over_dx", 0.25, 8.0);
        check!(self.water.alpha_apic, "water.alpha_apic", 0.0, 1.0);
        check!(
            self.water.boundary_friction,
            "water.boundary_friction",
            0.0,
            2.0
        );
        Self::validate_granular_family(&self.soil_family, "soil_family")?;
        Self::validate_granular_family(&self.sand_family, "sand_family")?;
        check!(
            self.granular_tensile_clamp,
            "granular_tensile_clamp",
            0.0,
            1.0e5
        );

        for (label, count) in [
            ("particle_counts.water_liquid", self.particle_counts.water_liquid),
            ("particle_counts.stone_solid", self.particle_counts.stone_solid),
            (
                "particle_counts.stone_granular",
                self.particle_counts.stone_granular,
            ),
            ("particle_counts.soil_solid", self.particle_counts.soil_solid),
            (
                "particle_counts.soil_granular",
                self.particle_counts.soil_granular,
            ),
            ("particle_counts.sand_solid", self.particle_counts.sand_solid),
            (
                "particle_counts.sand_granular",
                self.particle_counts.sand_granular,
            ),
            ("particle_counts.grass_solid", self.particle_counts.grass_solid),
            (
                "particle_counts.grass_granular",
                self.particle_counts.grass_granular,
            ),
        ] {
            if count == 0 || count > 256 {
                return Err(format!("{label}: {count} は [1, 256] の範囲外"));
            }
        }

        for (label, scale) in [
            (
                "particle_mass_scales.water_liquid",
                self.particle_mass_scales.water_liquid,
            ),
            (
                "particle_mass_scales.stone_solid",
                self.particle_mass_scales.stone_solid,
            ),
            (
                "particle_mass_scales.stone_granular",
                self.particle_mass_scales.stone_granular,
            ),
            (
                "particle_mass_scales.soil_solid",
                self.particle_mass_scales.soil_solid,
            ),
            (
                "particle_mass_scales.soil_granular",
                self.particle_mass_scales.soil_granular,
            ),
            (
                "particle_mass_scales.sand_solid",
                self.particle_mass_scales.sand_solid,
            ),
            (
                "particle_mass_scales.sand_granular",
                self.particle_mass_scales.sand_granular,
            ),
            (
                "particle_mass_scales.grass_solid",
                self.particle_mass_scales.grass_solid,
            ),
            (
                "particle_mass_scales.grass_granular",
                self.particle_mass_scales.grass_granular,
            ),
        ] {
            check!(scale, label, 0.01, 100.0);
        }

        for (label, rule) in [
            ("terrain_break.stone", self.terrain_break.stone),
            ("terrain_break.soil", self.terrain_break.soil),
            ("terrain_break.sand", self.terrain_break.sand),
            ("terrain_break.grass", self.terrain_break.grass),
        ] {
            if rule.output_count == 0 || rule.output_count > rule.outputs.len() as u32 {
                return Err(format!(
                    "{label}.output_count: {} は [1, {}] の範囲外",
                    rule.output_count,
                    rule.outputs.len()
                ));
            }
            for output in rule.outputs.iter().take(rule.output_count as usize) {
                if output.count_per_cell == 0 {
                    return Err(format!(
                        "{label}: count_per_cell=0 の break 出力は許可しない"
                    ));
                }
                if !matches!(
                    output.material,
                    ParticleMaterial::StoneGranular
                        | ParticleMaterial::SoilGranular
                        | ParticleMaterial::SandGranular
                        | ParticleMaterial::GrassGranular
                ) {
                    return Err(format!(
                        "{label}: break 出力 material {:?} は granular である必要がある",
                        output.material
                    ));
                }
            }
        }

        Ok(())
    }

    fn validate_granular_family(
        params: &GranularMaterialFamilyParams,
        prefix: &str,
    ) -> Result<(), String> {
        macro_rules! check {
            ($val:expr, $name:expr, $lo:expr, $hi:expr) => {
                if !($val >= $lo && $val <= $hi) {
                    return Err(format!(
                        "{}.{}: {} は [{}, {}] の範囲外",
                        prefix, $name, $val, $lo, $hi
                    ));
                }
            };
        }

        check!(params.alpha_apic, "alpha_apic", 0.0, 1.0);
        check!(params.boundary_friction, "boundary_friction", 0.0, 2.0);
        check!(params.youngs_modulus_pa, "youngs_modulus_pa", 1.0, 1.0e8);
        check!(params.poisson_ratio, "poisson_ratio", 0.0, 0.45);
        check!(params.friction_deg, "friction_deg", 0.0, 60.0);
        check!(params.cohesion_pa, "cohesion_pa", 0.0, 1.0e6);
        check!(params.hardening, "hardening", 0.0, 10.0);
        Ok(())
    }
}

pub const fn particle_material_id(material: ParticleMaterial) -> u8 {
    match material {
        ParticleMaterial::WaterLiquid => 0,
        ParticleMaterial::StoneSolid => 1,
        ParticleMaterial::StoneGranular => 2,
        ParticleMaterial::SoilSolid => 3,
        ParticleMaterial::SoilGranular => 4,
        ParticleMaterial::SandSolid => 5,
        ParticleMaterial::SandGranular => 6,
        ParticleMaterial::GrassSolid => 7,
        ParticleMaterial::GrassGranular => 8,
    }
}

pub const fn particle_material_from_id(material_id: u32) -> Option<ParticleMaterial> {
    match material_id as u8 {
        0 => Some(ParticleMaterial::WaterLiquid),
        1 => Some(ParticleMaterial::StoneSolid),
        2 => Some(ParticleMaterial::StoneGranular),
        3 => Some(ParticleMaterial::SoilSolid),
        4 => Some(ParticleMaterial::SoilGranular),
        5 => Some(ParticleMaterial::SandSolid),
        6 => Some(ParticleMaterial::SandGranular),
        7 => Some(ParticleMaterial::GrassSolid),
        8 => Some(ParticleMaterial::GrassGranular),
        _ => None,
    }
}

pub const fn terrain_material_id(material: TerrainMaterial) -> u16 {
    match material {
        TerrainMaterial::Stone => 1,
        TerrainMaterial::Soil => 2,
        TerrainMaterial::Sand => 3,
        TerrainMaterial::Grass => 4,
    }
}

pub const fn terrain_material_from_id(material_id: u16) -> Option<TerrainMaterial> {
    match material_id {
        1 => Some(TerrainMaterial::Stone),
        2 => Some(TerrainMaterial::Soil),
        3 => Some(TerrainMaterial::Sand),
        4 => Some(TerrainMaterial::Grass),
        _ => None,
    }
}

pub const fn pack_particle_home_slot(slot_id: u32, material: ParticleMaterial) -> u32 {
    ((particle_material_id(material) as u32) << PARTICLE_HOME_MATERIAL_SHIFT)
        | (slot_id & PARTICLE_HOME_SLOT_MASK)
}

pub const fn unpack_particle_home_slot(packed: u32) -> u32 {
    packed & PARTICLE_HOME_SLOT_MASK
}

pub const fn repack_particle_home_slot(packed: u32, slot_id: u32) -> u32 {
    (packed & !PARTICLE_HOME_SLOT_MASK) | (slot_id & PARTICLE_HOME_SLOT_MASK)
}

pub const fn unpack_particle_material_id(packed: u32) -> u32 {
    packed >> PARTICLE_HOME_MATERIAL_SHIFT
}
