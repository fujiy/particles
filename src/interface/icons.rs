use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use super::*;
use crate::params::interface::{InterfaceIconPaletteParams, UiColor8};

impl WorldToolIconSet {
    pub(super) fn icon_for(&self, tool: WorldTool) -> Handle<Image> {
        match tool {
            WorldTool::WaterLiquid => self.water_liquid.clone(),
            WorldTool::StoneSolid => self.stone_solid.clone(),
            WorldTool::StoneGranular => self.stone_granular.clone(),
            WorldTool::SoilSolid => self.soil_solid.clone(),
            WorldTool::SoilGranular => self.soil_granular.clone(),
            WorldTool::SandSolid => self.sand_solid.clone(),
            WorldTool::SandGranular => self.sand_granular.clone(),
            WorldTool::GrassSolid => self.grass_solid.clone(),
            WorldTool::GrassGranular => self.grass_granular.clone(),
            WorldTool::Break => self.break_icon.clone(),
            WorldTool::Delete => self.delete.clone(),
        }
    }
}

pub(super) fn create_world_tool_icon_set(
    images: &mut Assets<Image>,
    icon_palette: &InterfaceIconPaletteParams,
) -> WorldToolIconSet {
    let water_palette = to_material_palette(icon_palette.water);
    let stone_palette = to_material_palette(icon_palette.stone);
    let sand_palette = to_material_palette(icon_palette.sand);
    let soil_palette = to_material_palette(icon_palette.soil);
    let grass_palette = to_material_palette(icon_palette.grass);

    let water_liquid = images.add(build_material_icon_image(
        water_palette,
        &MATERIAL_PATTERN_LIQUID,
        0x4e23_1f91,
    ));
    let stone_solid = images.add(build_material_icon_image(
        stone_palette,
        &MATERIAL_PATTERN_SOLID,
        0x8a52_d9b7,
    ));
    let stone_granular = images.add(build_material_icon_image(
        stone_palette,
        &MATERIAL_PATTERN_GRANULAR,
        0x77bc_26f1,
    ));
    let soil_solid = images.add(build_material_icon_image(
        soil_palette,
        &MATERIAL_PATTERN_SOLID,
        0x68a2_1bd4,
    ));
    let soil_granular = images.add(build_material_icon_image(
        soil_palette,
        &MATERIAL_PATTERN_GRANULAR,
        0xb86a_c921,
    ));
    let sand_solid = images.add(build_material_icon_image(
        sand_palette,
        &MATERIAL_PATTERN_SOLID,
        0x2f9a_43ce,
    ));
    let sand_granular = images.add(build_material_icon_image(
        sand_palette,
        &MATERIAL_PATTERN_GRANULAR,
        0x9133_257e,
    ));
    let grass_solid = images.add(build_material_icon_image(
        grass_palette,
        &MATERIAL_PATTERN_SOLID,
        0x8cf1_3157,
    ));
    let grass_granular = images.add(build_material_icon_image(
        grass_palette,
        &MATERIAL_PATTERN_GRANULAR,
        0x2ca7_b5de,
    ));
    let break_icon = images.add(build_break_icon_image());
    let delete = images.add(build_delete_icon_image());
    WorldToolIconSet {
        water_liquid,
        stone_solid,
        stone_granular,
        soil_solid,
        soil_granular,
        sand_solid,
        sand_granular,
        grass_solid,
        grass_granular,
        break_icon,
        delete,
    }
}

fn to_material_palette(colors: [UiColor8; 4]) -> [[u8; 4]; 4] {
    colors.map(UiColor8::as_rgba)
}

fn build_material_icon_image(
    palette: [[u8; 4]; 4],
    pattern: &MaterialPattern8,
    seed: u32,
) -> Image {
    let width = TOOLBAR_ICON_SIZE_PX;
    let height = TOOLBAR_ICON_SIZE_PX;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for (gy, row) in pattern.iter().enumerate() {
        for (gx, enabled) in row.iter().copied().enumerate() {
            if !enabled {
                continue;
            }
            let palette_index = material_icon_palette_index(gx as u32, gy as u32, seed);
            let color = palette[palette_index];
            let start_x = gx as u32 * TOOLBAR_ICON_DOT_PX;
            let start_y = gy as u32 * TOOLBAR_ICON_DOT_PX;
            for py in start_y..(start_y + TOOLBAR_ICON_DOT_PX) {
                for px in start_x..(start_x + TOOLBAR_ICON_DOT_PX) {
                    if px >= width || py >= height {
                        continue;
                    }
                    let idx = ((py * width + px) * 4) as usize;
                    pixels[idx..idx + 4].copy_from_slice(&color);
                }
            }
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    image
}

fn material_icon_palette_index(x: u32, y: u32, seed: u32) -> usize {
    let mut state = x.wrapping_mul(0x45d9f3b);
    state ^= y.wrapping_mul(0x27d4eb2d);
    state ^= seed;
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    (state & 0b11) as usize
}

fn build_delete_icon_image() -> Image {
    let width = TOOLBAR_ICON_SIZE_PX;
    let height = TOOLBAR_ICON_SIZE_PX;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let red = [220u8, 52u8, 52u8, 255u8];
    let thickness = 3i32;
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let d1 = (x - y).abs();
            let d2 = ((width as i32 - 1 - x) - y).abs();
            if d1 <= thickness || d2 <= thickness {
                let idx = (((y as u32) * width + (x as u32)) * 4) as usize;
                pixels[idx..idx + 4].copy_from_slice(&red);
            }
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    image
}

fn build_break_icon_image() -> Image {
    let width = TOOLBAR_ICON_SIZE_PX;
    let height = TOOLBAR_ICON_SIZE_PX;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let amber = [246u8, 170u8, 38u8, 255u8];
    let dark = [115u8, 63u8, 8u8, 255u8];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            let border = x < 2 || y < 2 || x >= width - 2 || y >= height - 2;
            if border {
                pixels[idx..idx + 4].copy_from_slice(&dark);
                continue;
            }
            let diagonal =
                (x as i32 - y as i32).abs() <= 1 || ((width - 1 - x) as i32 - y as i32).abs() <= 1;
            if diagonal {
                pixels[idx..idx + 4].copy_from_slice(&amber);
            }
        }
    }
    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    image
}
