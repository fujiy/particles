use bevy::prelude::{IVec2, Vec2};

use crate::physics::material::TerrainMaterial;
use crate::physics::world::constants::CELL_SIZE_M;

use super::params::{
    BASE_SURFACE_Y, HEIGHT_NOISE_AMP_CELLS, HEIGHT_NOISE_DETAIL_AMP_CELLS,
    HEIGHT_NOISE_DETAIL_FREQ_MULT, HEIGHT_NOISE_DETAIL_OCTAVES, HEIGHT_NOISE_FREQ,
    SOIL_DEPTH_CELLS, SOIL_DEPTH_NOISE_FREQ, SOIL_DEPTH_VARIATION_CELLS,
    SURFACE_BLEND_BAND_BASE_CELLS, SURFACE_BLEND_BAND_PER_FOOTPRINT, TERRAIN_FBM_LACUNARITY,
    TERRAIN_GENERATOR_VERSION,
};
use super::random::{default_surface_fbm, fbm_1d_filtered};

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
    sample_surface_height_cells(world_x as f64, 0.0).round() as i32
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

fn sample_surface_height_cells(x_cells: f64, footprint_cells: f64) -> f32 {
    let macro_n = default_surface_fbm(x_cells, footprint_cells, HEIGHT_NOISE_FREQ);
    let detail_n = fbm_1d_filtered(
        x_cells,
        HEIGHT_NOISE_FREQ * HEIGHT_NOISE_DETAIL_FREQ_MULT,
        HEIGHT_NOISE_DETAIL_OCTAVES,
        TERRAIN_FBM_LACUNARITY,
        0.5,
        footprint_cells,
        TERRAIN_GENERATOR_VERSION as f64 + 101.0,
    );
    BASE_SURFACE_Y as f32
        + (macro_n as f32 * HEIGHT_NOISE_AMP_CELLS as f32)
        + (detail_n as f32 * HEIGHT_NOISE_DETAIL_AMP_CELLS)
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

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
