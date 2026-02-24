use std::collections::{HashMap, HashSet};

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::physics::material::{
    DEFAULT_MATERIAL_PARAMS, MaterialParams, particle_properties, terrain_boundary_radius_m,
};
use crate::physics::state::SimUpdateSet;
use crate::physics::world::object::{ObjectData, ObjectId, ObjectWorld};
use crate::physics::world::particle::{
    ParticleMaterial, ParticleWorld, nominal_particle_draw_radius_m,
};
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, TerrainCell, TerrainWorld,
    world_to_cell,
};

mod constants;
mod palette;
pub(crate) use constants::*;
pub(crate) use palette::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RenderTileKey {
    origin_chunk: IVec2,
    span_chunks: i32,
}

#[derive(Component, Clone, Copy, Debug)]
struct RenderTile {
    key: RenderTileKey,
    lod_level: u32,
}

#[derive(Component)]
struct RenderTileVisual {
    image: Handle<Image>,
}

#[derive(Component, Default)]
struct RenderTileFlags {
    terrain_dirty: bool,
    needs_upload: bool,
}

#[derive(Component, Default)]
struct RenderTileTerrainCache {
    pixels: Vec<u8>,
    rebuilt_this_frame: bool,
}

#[derive(Component, Default)]
struct RenderTileParticleState {
    indices: Vec<usize>,
    deferred: Vec<DeferredRenderParticle>,
    has_particles: bool,
    had_particles_last_compose: bool,
    changed_this_frame: bool,
}

#[derive(Component, Default)]
struct RenderTileUploadBuffer {
    pixels: Vec<u8>,
    pending: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderVisibleTile {
    pub origin_chunk: IVec2,
    pub span_chunks: i32,
    pub lod_level: u32,
}

#[derive(Resource, Default)]
pub struct TerrainTileRenderState {
    entities_by_key: HashMap<RenderTileKey, Entity>,
    target_tiles: HashSet<RenderTileKey>,
    target_lod_level: u32,
    target_span_chunks: i32,
    evict_frames_outside_required: HashMap<RenderTileKey, u16>,
}

#[derive(Resource, Default)]
pub struct TerrainRenderDiagnostics {
    pub terrain_updated_chunk_highlight_frames: HashMap<IVec2, u8>,
    pub particle_updated_chunk_highlight_frames: HashMap<IVec2, u8>,
    pub particle_updated_sub_block_highlight_frames: HashMap<IVec2, u8>,
    pub last_particle_dirty_frame: u64,
    pub visible_tiles: Vec<RenderVisibleTile>,
}

const UPDATED_CHUNK_HIGHLIGHT_FRAMES: u8 = 60;
const REST_DENSITY: f32 = DEFAULT_MATERIAL_PARAMS.rest_density;

#[derive(Resource, Default)]
pub struct WaterRenderState {
    chunk_sprites: HashMap<IVec2, OverlayChunkSprite>,
    density: Vec<f32>,
    blur_tmp: Vec<f32>,
    blurred_density: Vec<f32>,
    had_any_water: bool,
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
    deferred_by_chunk: HashMap<IVec2, Vec<DeferredRenderParticle>>,
    water_by_chunk: HashMap<IVec2, Vec<usize>>,
    free_by_chunk: HashMap<IVec2, Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct DeferredRenderParticle {
    position: Vec2,
    material: ParticleMaterial,
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
mod object_sprites;
mod tiles;
mod water;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainTileRenderState>()
            .init_resource::<TerrainLodPaletteCache>()
            .init_resource::<TerrainRenderDiagnostics>()
            .init_resource::<WaterRenderState>()
            .init_resource::<ObjectRenderState>()
            .init_resource::<FreeParticleRenderState>()
            .init_resource::<ParticleRenderChunkCache>()
            .add_systems(Startup, bootstrap_terrain_chunks)
            .add_systems(
                Update,
                (
                    refresh_particle_render_chunk_cache.in_set(SimUpdateSet::Rendering),
                    tiles::sync_required_render_tiles
                        .in_set(SimUpdateSet::Rendering)
                        .after(refresh_particle_render_chunk_cache),
                    tiles::mark_dirty_render_tiles_from_world
                        .in_set(SimUpdateSet::Rendering)
                        .after(tiles::sync_required_render_tiles),
                    tiles::refresh_render_tile_particle_state
                        .in_set(SimUpdateSet::Rendering)
                        .after(tiles::mark_dirty_render_tiles_from_world),
                    tiles::refresh_render_tile_terrain_cache
                        .in_set(SimUpdateSet::Rendering)
                        .after(tiles::mark_dirty_render_tiles_from_world),
                    tiles::compose_render_tile_upload_buffers
                        .in_set(SimUpdateSet::Rendering)
                        .after(tiles::refresh_render_tile_particle_state)
                        .after(tiles::refresh_render_tile_terrain_cache),
                    tiles::apply_render_tile_uploads
                        .in_set(SimUpdateSet::Rendering)
                        .after(tiles::compose_render_tile_upload_buffers),
                    sync_water_dots_to_render
                        .in_set(SimUpdateSet::Rendering)
                        .after(refresh_particle_render_chunk_cache),
                    sync_free_particles_to_render
                        .in_set(SimUpdateSet::Rendering)
                        .after(refresh_particle_render_chunk_cache),
                    sync_object_sprites_to_render.in_set(SimUpdateSet::Rendering),
                ),
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
    cache.deferred_by_chunk.clear();
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

    particles.for_each_deferred_particle(|chunk, position, material| {
        cache
            .deferred_by_chunk
            .entry(chunk)
            .or_default()
            .push(DeferredRenderParticle { position, material });
    });
}

pub use water::sync_water_dots_to_render;

pub use free_particles::sync_free_particles_to_render;

pub use object_sprites::sync_object_sprites_to_render;

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
