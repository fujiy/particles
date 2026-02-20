use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use super::particle::{
    ParticleMaterial, ParticleWorld, REST_DENSITY, STONE_PARTICLE_RADIUS_M,
    TERRAIN_BOUNDARY_RADIUS_M, nominal_particle_draw_radius_m,
};
use super::terrain::{
    CELL_PIXEL_SIZE, CELL_SIZE_M, CHUNK_PIXEL_SIZE, CHUNK_SIZE, CHUNK_WORLD_SIZE_M, TerrainCell,
    TerrainChunk, TerrainMaterial, TerrainWorld, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y,
    WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y, world_to_cell,
};

#[derive(Resource, Default)]
pub struct TerrainRenderState {
    chunk_sprites: HashMap<IVec2, TerrainChunkSprite>,
}

#[derive(Resource, Default)]
pub struct WaterRenderState {
    image: Option<Handle<Image>>,
    entity: Option<Entity>,
    density: Vec<f32>,
    blur_tmp: Vec<f32>,
    blurred_density: Vec<f32>,
}

struct TerrainChunkSprite {
    entity: Entity,
    image: Handle<Image>,
}

const WORLD_CHUNK_WIDTH: u32 = (WORLD_MAX_CHUNK_X - WORLD_MIN_CHUNK_X + 1) as u32;
const WORLD_CHUNK_HEIGHT: u32 = (WORLD_MAX_CHUNK_Y - WORLD_MIN_CHUNK_Y + 1) as u32;
const WATER_IMAGE_WIDTH: u32 = WORLD_CHUNK_WIDTH * CHUNK_PIXEL_SIZE;
const WATER_IMAGE_HEIGHT: u32 = WORLD_CHUNK_HEIGHT * CHUNK_PIXEL_SIZE;
const WATER_DOT_THRESHOLD_REST_DENSITY_RATIO: f32 = 0.4;
const WATER_DOT_SMOOTH_WIDTH: f32 = 1.0;
const WATER_BLUR_RADIUS_DOTS: i32 = 2;
const WATER_DOT_SCALE: f32 = CELL_PIXEL_SIZE as f32 / CELL_SIZE_M;
const WATER_RENDER_Z: f32 = 6.0;
const STONE_DOT_ALPHA: u8 = 255;

pub fn bootstrap_terrain_chunks(mut terrain_world: ResMut<TerrainWorld>) {
    terrain_world.reset_fixed_world();
    terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
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

pub fn sync_water_dots_to_render(
    mut commands: Commands,
    particles: Res<ParticleWorld>,
    terrain: Res<TerrainWorld>,
    mut water_state: ResMut<WaterRenderState>,
    mut images: ResMut<Assets<Image>>,
) {
    let center = world_center();
    let world_size = world_size();
    let image_handle = if let Some(handle) = &water_state.image {
        handle.clone()
    } else {
        let image = blank_water_image();
        let handle = images.add(image);
        let mut sprite = Sprite::from_image(handle.clone());
        sprite.custom_size = Some(world_size);
        let entity = commands
            .spawn((
                sprite,
                Transform::from_xyz(center.x, center.y, WATER_RENDER_Z),
            ))
            .id();
        water_state.entity = Some(entity);
        water_state.image = Some(handle.clone());
        handle
    };

    let mut pixels = vec![0_u8; (WATER_IMAGE_WIDTH * WATER_IMAGE_HEIGHT * 4) as usize];
    water_state
        .density
        .resize((WATER_IMAGE_WIDTH * WATER_IMAGE_HEIGHT) as usize, 0.0);
    water_state.density.fill(0.0);
    water_state
        .blur_tmp
        .resize((WATER_IMAGE_WIDTH * WATER_IMAGE_HEIGHT) as usize, 0.0);
    water_state
        .blurred_density
        .resize((WATER_IMAGE_WIDTH * WATER_IMAGE_HEIGHT) as usize, 0.0);

    let draw_radius = nominal_particle_draw_radius_m();
    let draw_radius_px = (draw_radius * WATER_DOT_SCALE).ceil() as i32;
    let dot_threshold = REST_DENSITY * WATER_DOT_THRESHOLD_REST_DENSITY_RATIO;

    for (particle_index, &pos) in particles.positions().iter().enumerate() {
        if !matches!(
            particles.materials()[particle_index],
            ParticleMaterial::Water
        ) {
            continue;
        }
        let particle_mass = particles.masses()[particle_index];
        let px_center = world_to_water_pixel(pos);
        for py in (px_center.y - draw_radius_px)..=(px_center.y + draw_radius_px) {
            if py < 0 || py >= WATER_IMAGE_HEIGHT as i32 {
                continue;
            }
            for px in (px_center.x - draw_radius_px)..=(px_center.x + draw_radius_px) {
                if px < 0 || px >= WATER_IMAGE_WIDTH as i32 {
                    continue;
                }
                let sample_world = water_pixel_to_world(IVec2::new(px, py));
                if is_solid_terrain_at_world(&terrain, sample_world) {
                    continue;
                }
                let r2 = (sample_world - pos).length_squared();
                let w = kernel_poly6_for_render(r2, draw_radius);
                if w == 0.0 {
                    continue;
                }
                let idx = water_density_index(px, py);
                water_state.density[idx] += particle_mass * w;
            }
        }
    }

    let density = std::mem::take(&mut water_state.density);
    let WaterRenderState {
        blur_tmp,
        blurred_density,
        ..
    } = &mut *water_state;
    blur_density_gaussian_separable(
        &density,
        blur_tmp,
        blurred_density,
        WATER_IMAGE_WIDTH,
        WATER_IMAGE_HEIGHT,
        WATER_BLUR_RADIUS_DOTS,
    );
    water_state.density = density;

    for py in 0..WATER_IMAGE_HEIGHT {
        for px in 0..WATER_IMAGE_WIDTH {
            let idx = water_density_index(px as i32, py as i32);
            let blurred = water_state.blurred_density[idx];
            let coverage = smoothstep(
                dot_threshold - WATER_DOT_SMOOTH_WIDTH,
                dot_threshold + WATER_DOT_SMOOTH_WIDTH,
                blurred,
            );
            if coverage <= 0.0 {
                continue;
            }
            let sample_world = water_pixel_to_world(IVec2::new(px as i32, py as i32));
            if is_solid_terrain_at_world(&terrain, sample_world) {
                continue;
            }
            let mut color = water_palette_color(px as i32, py as i32, blurred, dot_threshold);
            color[3] = ((color[3] as f32) * coverage).round().clamp(0.0, 255.0) as u8;
            let offset = idx * 4;
            pixels[offset..offset + 4].copy_from_slice(&color);
        }
    }

    let stone_radius_px = (STONE_PARTICLE_RADIUS_M * WATER_DOT_SCALE).ceil() as i32;
    let stone_radius_px2 = (stone_radius_px * stone_radius_px) as f32;
    for (particle_index, &pos) in particles.positions().iter().enumerate() {
        if !matches!(
            particles.materials()[particle_index],
            ParticleMaterial::Stone
        ) {
            continue;
        }
        let center_px = world_to_water_pixel(pos);
        for py in (center_px.y - stone_radius_px)..=(center_px.y + stone_radius_px) {
            if py < 0 || py >= WATER_IMAGE_HEIGHT as i32 {
                continue;
            }
            for px in (center_px.x - stone_radius_px)..=(center_px.x + stone_radius_px) {
                if px < 0 || px >= WATER_IMAGE_WIDTH as i32 {
                    continue;
                }
                let dx = (px - center_px.x) as f32;
                let dy = (py - center_px.y) as f32;
                if dx * dx + dy * dy > stone_radius_px2 {
                    continue;
                }
                let idx = water_density_index(px, py);
                let offset = idx * 4;
                let palette = [
                    [70, 67, 63, STONE_DOT_ALPHA],
                    [83, 79, 74, STONE_DOT_ALPHA],
                    [95, 90, 84, STONE_DOT_ALPHA],
                    [108, 103, 96, STONE_DOT_ALPHA],
                ];
                let pidx = deterministic_palette_index(px, py);
                pixels[offset..offset + 4].copy_from_slice(&palette[pidx]);
            }
        }
    }

    if let Some(image) = images.get_mut(&image_handle) {
        image.data = Some(pixels);
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

fn blank_water_image() -> Image {
    Image::new_fill(
        Extent3d {
            width: WATER_IMAGE_WIDTH,
            height: WATER_IMAGE_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn world_size() -> Vec2 {
    Vec2::new(
        WORLD_CHUNK_WIDTH as f32 * CHUNK_WORLD_SIZE_M,
        WORLD_CHUNK_HEIGHT as f32 * CHUNK_WORLD_SIZE_M,
    )
}

fn world_center() -> Vec2 {
    let min = Vec2::new(
        WORLD_MIN_CHUNK_X as f32 * CHUNK_WORLD_SIZE_M,
        WORLD_MIN_CHUNK_Y as f32 * CHUNK_WORLD_SIZE_M,
    );
    min + world_size() * 0.5
}

fn world_to_water_pixel(world: Vec2) -> IVec2 {
    let min = Vec2::new(
        WORLD_MIN_CHUNK_X as f32 * CHUNK_WORLD_SIZE_M,
        WORLD_MIN_CHUNK_Y as f32 * CHUNK_WORLD_SIZE_M,
    );
    let rel = (world - min) * WATER_DOT_SCALE;
    IVec2::new(rel.x.floor() as i32, rel.y.floor() as i32)
}

fn water_pixel_to_world(pixel: IVec2) -> Vec2 {
    let min = Vec2::new(
        WORLD_MIN_CHUNK_X as f32 * CHUNK_WORLD_SIZE_M,
        WORLD_MIN_CHUNK_Y as f32 * CHUNK_WORLD_SIZE_M,
    );
    min + (pixel.as_vec2() + Vec2::splat(0.5)) / WATER_DOT_SCALE
}

fn water_density_index(px: i32, py: i32) -> usize {
    let image_y = WATER_IMAGE_HEIGHT as i32 - 1 - py;
    (image_y as u32 * WATER_IMAGE_WIDTH + px as u32) as usize
}

fn is_solid_terrain_at_world(terrain: &TerrainWorld, world_pos: Vec2) -> bool {
    matches!(
        terrain.get_loaded_cell_or_empty(world_to_cell(world_pos)),
        TerrainCell::Solid { .. }
    )
}

fn kernel_poly6_for_render(r2: f32, support_radius: f32) -> f32 {
    let h2 = support_radius * support_radius;
    if !(0.0..h2).contains(&r2) {
        return 0.0;
    }
    let x = h2 - r2;
    let coeff = 4.0 / (std::f32::consts::PI * h2.powi(4));
    coeff * x.powi(3)
}

fn blur_density_gaussian_separable(
    src: &[f32],
    tmp: &mut [f32],
    dst: &mut [f32],
    width: u32,
    height: u32,
    radius_dots: i32,
) {
    if radius_dots <= 0 {
        tmp.copy_from_slice(src);
        dst.copy_from_slice(src);
        return;
    }

    let sigma = (radius_dots as f32).max(1.0) * 0.6;
    let mut kernel = Vec::with_capacity((radius_dots * 2 + 1) as usize);
    let mut kernel_sum = 0.0_f32;
    for offset in -radius_dots..=radius_dots {
        let x = offset as f32;
        let weight = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(weight);
        kernel_sum += weight;
    }
    for weight in &mut kernel {
        *weight /= kernel_sum;
    }

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut accum = 0.0;
            for offset in -radius_dots..=radius_dots {
                let sx = (x + offset).clamp(0, width as i32 - 1);
                let weight = kernel[(offset + radius_dots) as usize];
                let idx = water_density_index(sx, y);
                accum += src[idx] * weight;
            }
            let idx = water_density_index(x, y);
            tmp[idx] = accum;
        }
    }

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut accum = 0.0;
            for offset in -radius_dots..=radius_dots {
                let sy = (y + offset).clamp(0, height as i32 - 1);
                let weight = kernel[(offset + radius_dots) as usize];
                let idx = water_density_index(x, sy);
                accum += tmp[idx] * weight;
            }
            let idx = water_density_index(x, y);
            dst[idx] = accum;
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn water_palette_color(x: i32, y: i32, density: f32, dot_threshold: f32) -> [u8; 4] {
    let palette = [
        [42, 120, 202, 235],
        [52, 136, 218, 240],
        [65, 152, 228, 245],
        [78, 167, 238, 250],
    ];
    let base = deterministic_palette_index(x, y);
    let boost = if density > dot_threshold * 1.6 { 1 } else { 0 };
    palette[(base + boost).min(3)]
}
