use bevy::prelude::*;
use noise::{NoiseFn, OpenSimplex};

use super::material::TerrainMaterial;

pub const TERRAIN_GENERATOR_VERSION: u32 = 1;
pub const WORLD_SEED: u32 = 13_370;
pub const HEIGHT_NOISE_FREQ: f64 = 0.035;
pub const HEIGHT_NOISE_AMP_CELLS: i32 = 10;
pub const BASE_SURFACE_Y: i32 = -4;
pub const SOIL_DEPTH_CELLS: i32 = 4;

pub fn surface_y_for_world_x(world_x: i32) -> i32 {
    let noise = OpenSimplex::new(generator_seed());
    let n = noise.get([world_x as f64 * HEIGHT_NOISE_FREQ, TERRAIN_GENERATOR_VERSION as f64]);
    BASE_SURFACE_Y + (n * HEIGHT_NOISE_AMP_CELLS as f64).round() as i32
}

pub fn generated_material_for_world_cell(global_cell: IVec2) -> Option<TerrainMaterial> {
    let surface_y = surface_y_for_world_x(global_cell.x);
    generated_material_for_y(global_cell.y, surface_y)
}

pub fn generated_material_for_y(global_y: i32, surface_y: i32) -> Option<TerrainMaterial> {
    if global_y > surface_y {
        return None;
    }
    if global_y > surface_y - SOIL_DEPTH_CELLS {
        return Some(TerrainMaterial::Soil);
    }
    Some(TerrainMaterial::Stone)
}

fn generator_seed() -> u32 {
    WORLD_SEED ^ TERRAIN_GENERATOR_VERSION.wrapping_mul(0x9E37_79B9)
}
