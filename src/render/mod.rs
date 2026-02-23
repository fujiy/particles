use std::collections::{HashMap, HashSet};

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::physics::material::{DEFAULT_MATERIAL_PARAMS, particle_properties};
use crate::physics::state::{SimUpdateSet, TerrainStreamingSettings};
use crate::physics::world::object::{ObjectData, ObjectId, ObjectWorld};
use crate::physics::world::particle::{ParticleMaterial, ParticleWorld, nominal_particle_draw_radius_m};
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, TerrainCell, TerrainChunk,
    TerrainWorld,
    world_to_cell,
};
use crate::physics::material::{MaterialParams, terrain_boundary_radius_m};

mod constants;
mod palette;
pub(crate) use constants::*;
pub(crate) use palette::*;

#[derive(Resource, Default)]
pub struct TerrainRenderState {
    chunk_sprites: HashMap<IVec2, TerrainChunkSprite>,
}

#[derive(Resource, Default)]
pub struct TerrainLodRenderState {
    chunk_sprites: HashMap<LodTileKey, TerrainLodTileSprite>,
    tiles_with_particles: HashSet<LodTileKey>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LodTileKey {
    origin_chunk: IVec2,
    span_chunks: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LodRenderTile {
    pub origin_chunk: IVec2,
    pub span_chunks: i32,
}

#[derive(Resource, Default)]
pub struct TerrainRenderDiagnostics {
    pub terrain_updated_chunk_highlight_frames: HashMap<IVec2, u8>,
    pub particle_updated_chunk_highlight_frames: HashMap<IVec2, u8>,
    pub lod_visible_tiles: Vec<LodRenderTile>,
}

const UPDATED_CHUNK_HIGHLIGHT_FRAMES: u8 = 60;
const REST_DENSITY: f32 = DEFAULT_MATERIAL_PARAMS.rest_density;

#[derive(Resource, Default)]
pub struct WaterRenderState {
    chunk_sprites: HashMap<IVec2, OverlayChunkSprite>,
    density: Vec<f32>,
    blur_tmp: Vec<f32>,
    blurred_density: Vec<f32>,
}

#[derive(Resource, Default)]
pub struct ObjectRenderState {
    object_sprites: HashMap<ObjectId, ObjectSprite>,
}

#[derive(Resource, Default)]
pub struct FreeParticleRenderState {
    chunk_sprites: HashMap<IVec2, OverlayChunkSprite>,
}

#[derive(Resource, Default)]
pub struct ParticleRenderChunkCache {
    all_by_chunk: HashMap<IVec2, Vec<usize>>,
    water_by_chunk: HashMap<IVec2, Vec<usize>>,
    free_by_chunk: HashMap<IVec2, Vec<usize>>,
}

struct TerrainChunkSprite {
    entity: Entity,
    image: Handle<Image>,
}

struct TerrainLodTileSprite {
    entity: Entity,
    image: Handle<Image>,
    terrain_pixels: Vec<u8>,
    has_particles: bool,
}

struct ObjectSprite {
    entity: Entity,
    image: Handle<Image>,
    world_size: Vec2,
}

struct OverlayChunkSprite {
    entity: Entity,
    image: Handle<Image>,
}

#[derive(Component, Clone, Copy)]
pub struct ObjectRenderHandle {
    #[allow(dead_code)]
    pub object_id: ObjectId,
}

mod free_particles;
mod lod;
mod object_sprites;
mod water;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainRenderState>()
            .init_resource::<TerrainLodPaletteCache>()
            .init_resource::<TerrainLodRenderState>()
            .init_resource::<TerrainRenderDiagnostics>()
            .init_resource::<WaterRenderState>()
            .init_resource::<ObjectRenderState>()
            .init_resource::<FreeParticleRenderState>()
            .init_resource::<ParticleRenderChunkCache>()
            .add_systems(Startup, bootstrap_terrain_chunks)
            .add_systems(
                Update,
                (
                    refresh_particle_render_chunk_cache,
                    sync_lod_chunks_to_render,
                    sync_dirty_terrain_chunks_to_render,
                    sync_water_dots_to_render,
                    sync_free_particles_to_render,
                    sync_object_sprites_to_render,
                )
                    .chain()
                    .in_set(SimUpdateSet::Rendering),
            );
    }
}

pub fn bootstrap_terrain_chunks(
    mut terrain_world: ResMut<TerrainWorld>,
    material_params: Res<MaterialParams>,
) {
    terrain_world.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(*material_params));
}

fn refresh_particle_render_chunk_cache(
    particles: Res<ParticleWorld>,
    object_world: Res<ObjectWorld>,
    mut cache: ResMut<ParticleRenderChunkCache>,
) {
    let _span = tracing::info_span!("render::refresh_particle_render_chunk_cache").entered();
    if !particles.is_changed() && !object_world.is_changed() {
        return;
    }

    cache.all_by_chunk.clear();
    cache.water_by_chunk.clear();
    cache.free_by_chunk.clear();

    for (idx, &pos) in particles.positions().iter().enumerate() {
        let chunk = chunk_coord_from_world(pos);
        cache.all_by_chunk.entry(chunk).or_default().push(idx);

        let material = particles.materials()[idx];
        if matches!(material, ParticleMaterial::WaterLiquid) {
            cache.water_by_chunk.entry(chunk).or_default().push(idx);
            continue;
        }
        if object_world.object_of_particle(idx).is_some() {
            continue;
        }
        cache.free_by_chunk.entry(chunk).or_default().push(idx);
    }
}

pub use lod::sync_lod_chunks_to_render;

pub fn sync_dirty_terrain_chunks_to_render(
    mut commands: Commands,
    mut terrain_world: ResMut<TerrainWorld>,
    mut render_state: ResMut<TerrainRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let _span = tracing::info_span!("render::sync_dirty_terrain_chunks_to_render").entered();
    let stale_chunks: Vec<_> = render_state
        .chunk_sprites
        .keys()
        .copied()
        .filter(|chunk_coord| terrain_world.chunk(*chunk_coord).is_none())
        .collect();
    for chunk_coord in stale_chunks {
        if let Some(entry) = render_state.chunk_sprites.remove(&chunk_coord) {
            commands.entity(entry.entity).despawn();
            images.remove(entry.image.id());
        }
    }

    let dirty_chunks = terrain_world.take_dirty_chunks();
    render_diagnostics
        .terrain_updated_chunk_highlight_frames
        .retain(|_, frames_left| {
            if *frames_left > 0 {
                *frames_left -= 1;
            }
            *frames_left > 0
        });
    render_diagnostics
        .particle_updated_chunk_highlight_frames
        .retain(|_, frames_left| {
            if *frames_left > 0 {
                *frames_left -= 1;
            }
            *frames_left > 0
        });

    for chunk_coord in dirty_chunks {
        render_diagnostics
            .terrain_updated_chunk_highlight_frames
            .insert(chunk_coord, UPDATED_CHUNK_HIGHLIGHT_FRAMES);
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

pub use water::sync_water_dots_to_render;

pub use free_particles::sync_free_particles_to_render;

pub use object_sprites::sync_object_sprites_to_render;

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

    image_from_rgba_pixels(pixels)
}

fn image_from_rgba_pixels(pixels: Vec<u8>) -> Image {
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

fn blend_rgba8_over(dst: &mut [u8], src: [u8; 4]) {
    debug_assert!(dst.len() >= 4);
    let src_a = src[3] as f32 / 255.0;
    if src_a <= 0.0 {
        return;
    }
    let inv_src_a = 1.0 - src_a;
    dst[0] = ((dst[0] as f32 * inv_src_a) + (src[0] as f32 * src_a))
        .round()
        .clamp(0.0, 255.0) as u8;
    dst[1] = ((dst[1] as f32 * inv_src_a) + (src[1] as f32 * src_a))
        .round()
        .clamp(0.0, 255.0) as u8;
    dst[2] = ((dst[2] as f32 * inv_src_a) + (src[2] as f32 * src_a))
        .round()
        .clamp(0.0, 255.0) as u8;
    dst[3] = dst[3].max(src[3]);
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

fn chunk_to_world_center(chunk_coord: IVec2) -> Vec2 {
    (chunk_coord.as_vec2() + Vec2::splat(0.5)) * CHUNK_WORLD_SIZE_M
}

fn chunk_coord_from_world(world_pos: Vec2) -> IVec2 {
    let cell = world_to_cell(world_pos);
    IVec2::new(
        cell.x.div_euclid(CHUNK_SIZE_I32),
        cell.y.div_euclid(CHUNK_SIZE_I32),
    )
}

fn mark_particle_chunk_updated(render_diagnostics: &mut TerrainRenderDiagnostics, chunk: IVec2) {
    render_diagnostics
        .particle_updated_chunk_highlight_frames
        .insert(chunk, UPDATED_CHUNK_HIGHLIGHT_FRAMES);
}

fn blank_water_image(width: u32, height: u32) -> Image {
    Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn world_to_chunk_pixel(world: Vec2, chunk_coord: IVec2) -> IVec2 {
    let min = chunk_coord.as_vec2() * CHUNK_WORLD_SIZE_M;
    let rel = (world - min) * WATER_DOT_SCALE;
    IVec2::new(rel.x.floor() as i32, rel.y.floor() as i32)
}

fn chunk_pixel_to_world(pixel: IVec2, chunk_coord: IVec2) -> Vec2 {
    let min = chunk_coord.as_vec2() * CHUNK_WORLD_SIZE_M;
    min + (pixel.as_vec2() + Vec2::splat(0.5)) / WATER_DOT_SCALE
}

fn chunk_density_index(px: i32, py: i32, width: u32, height: u32) -> usize {
    let image_y = height as i32 - 1 - py;
    (image_y as u32 * width + px as u32) as usize
}

fn is_solid_terrain_at_world(terrain: &TerrainWorld, world_pos: Vec2) -> bool {
    matches!(
        terrain.get_loaded_cell_or_empty(world_to_cell(world_pos)),
        TerrainCell::Solid { .. }
    )
}
