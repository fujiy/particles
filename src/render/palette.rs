use bevy::prelude::*;

use crate::physics::world::particle::ParticleMaterial;
use crate::physics::world::terrain::{TerrainCell, TerrainMaterial};

const PALETTE_COLOR_PROBS: [f32; 4] = [0.25, 0.25, 0.25, 0.25];

pub const STONE_BASE_PALETTE: [[u8; 4]; 4] = [
    [70, 67, 63, 255],
    [83, 79, 74, 255],
    [95, 90, 84, 255],
    [108, 103, 96, 255],
];

pub const SAND_BASE_PALETTE: [[u8; 4]; 4] = [
    [170, 150, 110, 255],
    [186, 166, 124, 255],
    [201, 181, 137, 255],
    [216, 196, 150, 255],
];

pub const SOIL_BASE_PALETTE: [[u8; 4]; 4] = [
    [105, 79, 56, 255],
    [119, 91, 67, 255],
    [133, 103, 78, 255],
    [147, 115, 88, 255],
];

#[derive(Clone, Copy)]
struct MaterialPalettes {
    stone: [[u8; 4]; 4],
    sand: [[u8; 4]; 4],
    soil: [[u8; 4]; 4],
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MaterialMoments {
    pub mean_premul: [f32; 4],
    pub second_premul: [f32; 4],
}

#[derive(Clone, Copy)]
struct MaterialPaletteMoments {
    stone: MaterialMoments,
    sand: MaterialMoments,
    soil: MaterialMoments,
}

#[derive(Resource)]
pub struct TerrainLodPaletteCache {
    levels: Vec<MaterialPalettes>,
    moments: Vec<MaterialPaletteMoments>,
}

impl Default for TerrainLodPaletteCache {
    fn default() -> Self {
        let mut levels = Vec::with_capacity(super::LOD_PRECOMPUTED_LEVELS as usize + 1);
        let mut moments = Vec::with_capacity(super::LOD_PRECOMPUTED_LEVELS as usize + 1);
        for lod_level in 0..=super::LOD_PRECOMPUTED_LEVELS {
            let stone = build_lod_palette(STONE_BASE_PALETTE, lod_level);
            let sand = build_lod_palette(SAND_BASE_PALETTE, lod_level);
            let soil = build_lod_palette(SOIL_BASE_PALETTE, lod_level);
            levels.push(MaterialPalettes { stone, sand, soil });
            moments.push(MaterialPaletteMoments {
                stone: compute_palette_moments(stone),
                sand: compute_palette_moments(sand),
                soil: compute_palette_moments(soil),
            });
        }
        Self { levels, moments }
    }
}

impl TerrainLodPaletteCache {
    pub(crate) fn palette_for(&self, material: TerrainMaterial, lod_level: u32) -> [[u8; 4]; 4] {
        let max_level = self.levels.len().saturating_sub(1) as u32;
        let level = lod_level.min(max_level) as usize;
        let palettes = self.levels[level];
        match material {
            TerrainMaterial::Stone => palettes.stone,
            TerrainMaterial::Sand => palettes.sand,
            TerrainMaterial::Soil => palettes.soil,
        }
    }

    pub(crate) fn moments_for(&self, material: TerrainMaterial, lod_level: u32) -> MaterialMoments {
        let max_level = self.moments.len().saturating_sub(1) as u32;
        let level = lod_level.min(max_level) as usize;
        let moments = self.moments[level];
        match material {
            TerrainMaterial::Stone => moments.stone,
            TerrainMaterial::Sand => moments.sand,
            TerrainMaterial::Soil => moments.soil,
        }
    }
}

fn build_lod_palette(base_palette: [[u8; 4]; 4], lod_level: u32) -> [[u8; 4]; 4] {
    if lod_level == 0 {
        return base_palette;
    }

    let scale = 1.0 / (super::LOD_LEVEL_RESOLUTION_DIV.max(2) as f32).powi(lod_level as i32);
    let mut mean = Vec3::ZERO;
    for (i, color) in base_palette.iter().enumerate() {
        let prob = PALETTE_COLOR_PROBS[i];
        mean += srgb8_to_linear_vec3([color[0], color[1], color[2]]) * prob;
    }

    let mut lod_palette = [[0_u8; 4]; 4];
    for (i, color) in base_palette.iter().enumerate() {
        let linear = srgb8_to_linear_vec3([color[0], color[1], color[2]]);
        let remapped = mean + (linear - mean) * scale;
        let rgb = linear_vec3_to_srgb8(remapped);
        lod_palette[i] = [rgb[0], rgb[1], rgb[2], color[3]];
    }
    lod_palette
}

fn compute_palette_moments(palette: [[u8; 4]; 4]) -> MaterialMoments {
    let mut mean = [0.0_f32; 4];
    let mut second = [0.0_f32; 4];
    for color in palette {
        let alpha = color[3] as f32 / 255.0;
        let sample = [
            color[0] as f32 * alpha,
            color[1] as f32 * alpha,
            color[2] as f32 * alpha,
            color[3] as f32,
        ];
        for channel in 0..4 {
            mean[channel] += sample[channel] * 0.25;
            second[channel] += sample[channel] * sample[channel] * 0.25;
        }
    }
    MaterialMoments {
        mean_premul: mean,
        second_premul: second,
    }
}

fn srgb8_to_linear_vec3(rgb: [u8; 3]) -> Vec3 {
    Vec3::new(
        srgb_channel_to_linear(rgb[0]),
        srgb_channel_to_linear(rgb[1]),
        srgb_channel_to_linear(rgb[2]),
    )
}

fn linear_vec3_to_srgb8(rgb: Vec3) -> [u8; 3] {
    [
        linear_channel_to_srgb8(rgb.x),
        linear_channel_to_srgb8(rgb.y),
        linear_channel_to_srgb8(rgb.z),
    ]
}

fn srgb_channel_to_linear(channel: u8) -> f32 {
    let x = channel as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_channel_to_srgb8(channel: f32) -> u8 {
    let x = channel.clamp(0.0, 1.0);
    let srgb = if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round().clamp(0.0, 255.0) as u8
}

pub fn cell_palette(cell: TerrainCell) -> Option<[[u8; 4]; 4]> {
    match cell {
        TerrainCell::Empty => None,
        TerrainCell::Solid {
            material: TerrainMaterial::Stone,
            ..
        } => Some(STONE_BASE_PALETTE),
        TerrainCell::Solid {
            material: TerrainMaterial::Sand,
            ..
        } => Some(SAND_BASE_PALETTE),
        TerrainCell::Solid {
            material: TerrainMaterial::Soil,
            ..
        } => Some(SOIL_BASE_PALETTE),
    }
}

pub fn lod_particle_color(material: ParticleMaterial) -> [u8; 4] {
    let palette = cell_palette_for_particle(material);
    let mut color = palette[0];
    color[3] = 220;
    color
}

pub fn deterministic_palette_index(x: i32, y: i32) -> usize {
    let mut state = (x as u32).wrapping_mul(0x45d9f3b);
    state ^= (y as u32).wrapping_mul(0x27d4eb2d);
    state ^= 0xa53c_9e4d;
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    (state & 0b11) as usize
}

pub fn cell_palette_for_particle(material: ParticleMaterial) -> [[u8; 4]; 4] {
    match material {
        ParticleMaterial::WaterLiquid => [
            [42, 120, 202, 235],
            [52, 136, 218, 240],
            [65, 152, 228, 245],
            [78, 167, 238, 250],
        ],
        ParticleMaterial::StoneSolid | ParticleMaterial::StoneGranular => STONE_BASE_PALETTE,
        ParticleMaterial::SoilSolid | ParticleMaterial::SoilGranular => SOIL_BASE_PALETTE,
        ParticleMaterial::SandSolid | ParticleMaterial::SandGranular => SAND_BASE_PALETTE,
    }
}
