use bevy::prelude::*;
use noise::{NoiseFn, OpenSimplex};
use std::sync::OnceLock;

use super::material::TerrainMaterial;
use super::world::constants::CELL_SIZE_M;

pub const TERRAIN_GENERATOR_VERSION: u32 = 1;
pub const WORLD_SEED: u32 = 13_370;
pub const HEIGHT_NOISE_FREQ: f64 = 0.035;
pub const HEIGHT_NOISE_AMP_CELLS: i32 = 10;
pub const BASE_SURFACE_Y: i32 = -4;
pub const SOIL_DEPTH_CELLS: i32 = 4;
const TERRAIN_FBM_OCTAVES: u32 = 6;
const TERRAIN_FBM_LACUNARITY: f64 = 2.0;
const TERRAIN_FBM_GAIN: f64 = 0.5;
const FBM_ATTENUATION_SCALE: f64 = 2.2;
const SURFACE_BLEND_BAND_BASE_CELLS: f32 = 0.35;
const SURFACE_BLEND_BAND_PER_FOOTPRINT: f32 = 0.75;
const SOIL_DEPTH_VARIATION_CELLS: f32 = 1.25;
const SOIL_DEPTH_NOISE_FREQ: f64 = HEIGHT_NOISE_FREQ * 0.65;

#[derive(Clone, Copy, Debug, Default)]
pub struct TerrainMaterialProbabilities {
    pub empty: f32,
    pub soil: f32,
    pub stone: f32,
}

impl TerrainMaterialProbabilities {
    fn normalize(self) -> Self {
        let empty = self.empty.max(0.0);
        let soil = self.soil.max(0.0);
        let stone = self.stone.max(0.0);
        let sum = (empty + soil + stone).max(1e-6);
        Self {
            empty: empty / sum,
            soil: soil / sum,
            stone: stone / sum,
        }
    }
}

pub fn surface_y_for_world_x(world_x: i32) -> i32 {
    let n = terrain_noise().get([
        world_x as f64 * HEIGHT_NOISE_FREQ,
        TERRAIN_GENERATOR_VERSION as f64,
    ]);
    BASE_SURFACE_Y + (n * HEIGHT_NOISE_AMP_CELLS as f64).round() as i32
}

pub fn generated_material_for_world_cell(global_cell: IVec2) -> Option<TerrainMaterial> {
    let surface_y = surface_y_for_world_x(global_cell.x);
    generated_material_for_y(global_cell.y, surface_y)
}

pub fn sample_material_probabilities(
    world_pos: Vec2,
    footprint_m: f32,
) -> TerrainMaterialProbabilities {
    let x_cells = world_pos.x / CELL_SIZE_M;
    let y_cells = world_pos.y / CELL_SIZE_M;
    let footprint_cells = (footprint_m / CELL_SIZE_M).max(1e-3);

    let surface_y = sample_surface_height_cells(x_cells as f64, footprint_cells as f64);
    let depth = surface_y - y_cells;
    let blend_band_cells = (SURFACE_BLEND_BAND_BASE_CELLS
        + footprint_cells * SURFACE_BLEND_BAND_PER_FOOTPRINT)
        .max(0.05);
    let solid_prob = smoothstep(-blend_band_cells, blend_band_cells, depth);

    let soil_depth = sample_soil_depth_cells(x_cells as f64, footprint_cells as f64);
    let stone_prob_within_solid = smoothstep(
        soil_depth - blend_band_cells,
        soil_depth + blend_band_cells,
        depth,
    );
    let stone = solid_prob * stone_prob_within_solid;
    let soil = (solid_prob - stone).max(0.0);
    let empty = (1.0 - solid_prob).max(0.0);
    TerrainMaterialProbabilities { empty, soil, stone }.normalize()
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

fn terrain_noise() -> &'static OpenSimplex {
    static NOISE: OnceLock<OpenSimplex> = OnceLock::new();
    NOISE.get_or_init(|| OpenSimplex::new(generator_seed()))
}

fn sample_surface_height_cells(x_cells: f64, footprint_cells: f64) -> f32 {
    let n = fbm_1d_filtered(
        x_cells,
        HEIGHT_NOISE_FREQ,
        TERRAIN_FBM_OCTAVES,
        TERRAIN_FBM_LACUNARITY,
        TERRAIN_FBM_GAIN,
        footprint_cells,
        TERRAIN_GENERATOR_VERSION as f64,
    );
    BASE_SURFACE_Y as f32 + (n as f32 * HEIGHT_NOISE_AMP_CELLS as f32)
}

fn sample_soil_depth_cells(x_cells: f64, footprint_cells: f64) -> f32 {
    let n = fbm_1d_filtered(
        x_cells,
        SOIL_DEPTH_NOISE_FREQ,
        4,
        TERRAIN_FBM_LACUNARITY,
        0.55,
        footprint_cells,
        TERRAIN_GENERATOR_VERSION as f64 + 17.0,
    );
    (SOIL_DEPTH_CELLS as f32 + n as f32 * SOIL_DEPTH_VARIATION_CELLS).max(1.0)
}

fn fbm_1d_filtered(
    x: f64,
    base_freq: f64,
    octaves: u32,
    lacunarity: f64,
    gain: f64,
    footprint_cells: f64,
    channel_offset: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = base_freq;
    let mut amp_sum = 0.0;
    let noise = terrain_noise();

    for octave in 0..octaves {
        let low_pass = octave_low_pass_weight(freq, footprint_cells);
        if low_pass < 1e-3 && octave > 1 {
            break;
        }
        let value = noise.get([x * freq, channel_offset + octave as f64 * 13.37]);
        let weighted_amp = amp * low_pass;
        sum += value * weighted_amp;
        amp_sum += weighted_amp;
        amp *= gain;
        freq *= lacunarity;
    }

    if amp_sum <= 1e-6 { 0.0 } else { sum / amp_sum }
}

fn octave_low_pass_weight(freq: f64, footprint_cells: f64) -> f64 {
    let footprint_freq = (freq * footprint_cells).max(0.0);
    1.0 / (1.0 + (footprint_freq * FBM_ATTENUATION_SCALE).powi(2))
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
