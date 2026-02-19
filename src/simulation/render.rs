use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use super::terrain::{
    CELL_PIXEL_SIZE, CHUNK_PIXEL_SIZE, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, TerrainCell,
    TerrainChunk, TerrainMaterial, TerrainWorld,
};

const WORLD_MIN_CHUNK_X: i32 = -2;
const WORLD_MAX_CHUNK_X: i32 = 1;
const WORLD_MIN_CHUNK_Y: i32 = -2;
const WORLD_MAX_CHUNK_Y: i32 = 1;

#[derive(Resource, Default)]
pub struct TerrainRenderState {
    chunk_sprites: HashMap<IVec2, TerrainChunkSprite>,
}

struct TerrainChunkSprite {
    entity: Entity,
    image: Handle<Image>,
}

pub fn bootstrap_terrain_chunks(mut terrain_world: ResMut<TerrainWorld>) {
    for chunk_y in WORLD_MIN_CHUNK_Y..=WORLD_MAX_CHUNK_Y {
        for chunk_x in WORLD_MIN_CHUNK_X..=WORLD_MAX_CHUNK_X {
            terrain_world.ensure_chunk_loaded(IVec2::new(chunk_x, chunk_y));
        }
    }

    let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32;
    let max_cell_x = (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1;
    let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32;
    let max_cell_y = (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1;

    terrain_world.fill_rect(
        IVec2::new(min_cell_x, min_cell_y),
        IVec2::new(min_cell_x, max_cell_y),
        TerrainCell::rock(),
    );
    terrain_world.fill_rect(
        IVec2::new(max_cell_x, min_cell_y),
        IVec2::new(max_cell_x, max_cell_y),
        TerrainCell::rock(),
    );
}

pub fn sync_dirty_terrain_chunks_to_render(
    mut commands: Commands,
    mut terrain_world: ResMut<TerrainWorld>,
    mut render_state: ResMut<TerrainRenderState>,
    mut images: ResMut<Assets<Image>>,
) {
    let dirty_chunks = terrain_world.take_dirty_chunks();

    for chunk_coord in dirty_chunks {
        let Some(chunk) = terrain_world.chunk(chunk_coord) else {
            continue;
        };

        if chunk.is_empty() {
            if let Some(entry) = render_state.chunk_sprites.remove(&chunk_coord) {
                commands.entity(entry.entity).despawn();
                images.remove(entry.image.id());
            }
            continue;
        }

        let image = build_chunk_image(chunk_coord, chunk);

        if let Some(entry) = render_state.chunk_sprites.get_mut(&chunk_coord) {
            if images.insert(entry.image.id(), image).is_err() {
                entry.image = images.add(build_chunk_image(chunk_coord, chunk));
                let mut sprite = Sprite::from_image(entry.image.clone());
                sprite.custom_size = Some(Vec2::splat(CHUNK_WORLD_SIZE_M));
                commands.entity(entry.entity).insert(sprite);
            }
            continue;
        }

        let image_handle = images.add(image);
        let mut sprite = Sprite::from_image(image_handle.clone());
        sprite.custom_size = Some(Vec2::splat(CHUNK_WORLD_SIZE_M));

        let center = chunk_to_world_center(chunk_coord);
        let entity = commands
            .spawn((sprite, Transform::from_xyz(center.x, center.y, 0.0)))
            .id();

        render_state.chunk_sprites.insert(
            chunk_coord,
            TerrainChunkSprite {
                entity,
                image: image_handle,
            },
        );
    }
}

fn build_chunk_image(chunk_coord: IVec2, chunk: &TerrainChunk) -> Image {
    let mut pixels = vec![0_u8; (CHUNK_PIXEL_SIZE * CHUNK_PIXEL_SIZE * 4) as usize];

    for local_y in 0..CHUNK_SIZE {
        for local_x in 0..CHUNK_SIZE {
            let cell = chunk.get(IVec2::new(local_x as i32, local_y as i32));
            write_cell_pixels(
                &mut pixels,
                chunk_coord,
                local_x as u32,
                local_y as u32,
                cell,
            );
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width: CHUNK_PIXEL_SIZE,
            height: CHUNK_PIXEL_SIZE,
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

fn cell_palette(cell: TerrainCell) -> Option<[[u8; 4]; 4]> {
    match cell {
        TerrainCell::Empty => None,
        TerrainCell::Solid {
            material: TerrainMaterial::Rock,
            ..
        } => Some([
            [70, 67, 63, 255],
            [83, 79, 74, 255],
            [95, 90, 84, 255],
            [108, 103, 96, 255],
        ]),
    }
}

fn write_cell_pixels(
    buffer: &mut [u8],
    chunk_coord: IVec2,
    local_x: u32,
    local_y: u32,
    cell: TerrainCell,
) {
    let Some(palette) = cell_palette(cell) else {
        return;
    };

    let world_px_y = (CHUNK_SIZE as u32 - 1 - local_y) * CELL_PIXEL_SIZE;
    let world_px_x = local_x * CELL_PIXEL_SIZE;
    let chunk_px_x = chunk_coord.x * CHUNK_PIXEL_SIZE as i32;
    let chunk_px_y = chunk_coord.y * CHUNK_PIXEL_SIZE as i32;

    for py in 0..CELL_PIXEL_SIZE {
        for px in 0..CELL_PIXEL_SIZE {
            let x = world_px_x + px;
            let y = world_px_y + py;
            let offset = ((y * CHUNK_PIXEL_SIZE + x) * 4) as usize;

            let world_x = chunk_px_x + (local_x * CELL_PIXEL_SIZE + px) as i32;
            let world_y = chunk_px_y + (local_y * CELL_PIXEL_SIZE + py) as i32;
            let palette_index = deterministic_palette_index(world_x, world_y);

            buffer[offset..offset + 4].copy_from_slice(&palette[palette_index]);
        }
    }
}

fn deterministic_palette_index(x: i32, y: i32) -> usize {
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

fn chunk_to_world_center(chunk_coord: IVec2) -> Vec2 {
    (chunk_coord.as_vec2() + Vec2::splat(0.5)) * CHUNK_WORLD_SIZE_M
}
