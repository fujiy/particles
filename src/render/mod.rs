use std::collections::{HashMap, HashSet};

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::PrimaryWindow;

use crate::physics::material::particle_properties;
use crate::physics::object::{ObjectData, ObjectId, ObjectWorld};
use crate::physics::particle::{
    ParticleMaterial, ParticleWorld, REST_DENSITY, TERRAIN_BOUNDARY_RADIUS_M,
    nominal_particle_draw_radius_m,
};
use crate::physics::state::{SimUpdateSet, TerrainStreamingSettings};
use crate::physics::terrain::{
    CELL_PIXEL_SIZE, CELL_SIZE_M, CHUNK_PIXEL_SIZE, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M,
    TerrainCell, TerrainChunk, TerrainMaterial, TerrainWorld, generated_cell_for_world,
    world_to_cell,
};

#[derive(Resource, Default)]
pub struct TerrainRenderState {
    chunk_sprites: HashMap<IVec2, TerrainChunkSprite>,
}

#[derive(Resource)]
pub struct TerrainLodPaletteCache {
    levels: Vec<MaterialPalettes>,
}

#[derive(Resource, Default)]
pub struct TerrainLodRenderState {
    chunk_sprites: HashMap<LodTileKey, TerrainChunkSprite>,
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

struct TerrainChunkSprite {
    entity: Entity,
    image: Handle<Image>,
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

#[derive(Clone, Copy)]
struct MaterialPalettes {
    stone: [[u8; 4]; 4],
    sand: [[u8; 4]; 4],
    soil: [[u8; 4]; 4],
}

#[derive(Component, Clone, Copy)]
pub struct ObjectRenderHandle {
    #[allow(dead_code)]
    pub object_id: ObjectId,
}

const WATER_DOT_THRESHOLD_REST_DENSITY_RATIO: f32 = 0.4;
const WATER_DOT_SMOOTH_WIDTH: f32 = 1.0;
const WATER_BLUR_RADIUS_DOTS: i32 = 2;
const WATER_DOT_SCALE: f32 = CELL_PIXEL_SIZE as f32 / CELL_SIZE_M;
const WATER_RENDER_Z: f32 = 6.0;
const FREE_PARTICLE_RENDER_Z: f32 = 6.5;
const OBJECT_DOT_SCALE: f32 = CELL_PIXEL_SIZE as f32 / CELL_SIZE_M;
const OBJECT_RENDER_Z: f32 = 7.0;
const OBJECT_PADDING_PX: u32 = 2;
const LOD_RENDER_Z: f32 = -0.5;
const LOD_LEVEL_RESOLUTION_DIV: i32 = 4;
const LOD_LEVEL_DISTANCE_STEP_CHUNKS: i32 = 6;
const LOD_PRECOMPUTED_LEVELS: u32 = 8;
const PALETTE_COLOR_PROBS: [f32; 4] = [0.25, 0.25, 0.25, 0.25];
const STONE_BASE_PALETTE: [[u8; 4]; 4] = [
    [70, 67, 63, 255],
    [83, 79, 74, 255],
    [95, 90, 84, 255],
    [108, 103, 96, 255],
];
const SAND_BASE_PALETTE: [[u8; 4]; 4] = [
    [170, 150, 110, 255],
    [186, 166, 124, 255],
    [201, 181, 137, 255],
    [216, 196, 150, 255],
];
const SOIL_BASE_PALETTE: [[u8; 4]; 4] = [
    [105, 79, 56, 255],
    [119, 91, 67, 255],
    [133, 103, 78, 255],
    [147, 115, 88, 255],
];

impl Default for TerrainLodPaletteCache {
    fn default() -> Self {
        let mut levels = Vec::with_capacity(LOD_PRECOMPUTED_LEVELS as usize + 1);
        for lod_level in 0..=LOD_PRECOMPUTED_LEVELS {
            levels.push(MaterialPalettes {
                stone: build_lod_palette(STONE_BASE_PALETTE, lod_level),
                sand: build_lod_palette(SAND_BASE_PALETTE, lod_level),
                soil: build_lod_palette(SOIL_BASE_PALETTE, lod_level),
            });
        }
        Self { levels }
    }
}

impl TerrainLodPaletteCache {
    fn palette_for(&self, material: TerrainMaterial, lod_level: u32) -> [[u8; 4]; 4] {
        let max_level = self.levels.len().saturating_sub(1) as u32;
        let level = lod_level.min(max_level) as usize;
        let palettes = self.levels[level];
        match material {
            TerrainMaterial::Stone => palettes.stone,
            TerrainMaterial::Sand => palettes.sand,
            TerrainMaterial::Soil => palettes.soil,
        }
    }
}

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
            .add_systems(Startup, bootstrap_terrain_chunks)
            .add_systems(
                Update,
                (
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

pub fn bootstrap_terrain_chunks(mut terrain_world: ResMut<TerrainWorld>) {
    terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
}

pub fn sync_lod_chunks_to_render(
    mut commands: Commands,
    streaming_settings: Res<TerrainStreamingSettings>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    lod_palette_cache: Res<TerrainLodPaletteCache>,
    mut lod_state: ResMut<TerrainLodRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some((min_chunk, max_chunk, center_chunk)) = visible_chunk_bounds(&windows, &cameras)
    else {
        let stale_tiles: Vec<_> = lod_state.chunk_sprites.keys().copied().collect();
        for tile in stale_tiles {
            if let Some(entry) = lod_state.chunk_sprites.remove(&tile) {
                commands.entity(entry.entity).despawn();
                images.remove(entry.image.id());
            }
        }
        render_diagnostics.lod_visible_tiles.clear();
        return;
    };

    let load_radius = streaming_settings.load_radius_chunks.max(0);
    let mut target_tiles = HashSet::new();
    for y in min_chunk.y..=max_chunk.y {
        for x in min_chunk.x..=max_chunk.x {
            let chunk_coord = IVec2::new(x, y);
            let dx = (chunk_coord.x - center_chunk.x).abs();
            let dy = (chunk_coord.y - center_chunk.y).abs();
            let chebyshev_distance = dx.max(dy);
            if chebyshev_distance <= load_radius {
                continue;
            }
            let outside_distance = chebyshev_distance - load_radius;
            let lod_level = lod_level_for_outside_distance(outside_distance);
            let span_chunks = lod_span_chunks_for_level(lod_level);
            let origin_chunk = align_chunk_to_lod_tile(chunk_coord, span_chunks);
            target_tiles.insert(LodTileKey {
                origin_chunk,
                span_chunks,
            });
        }
    }

    let stale_tiles: Vec<_> = lod_state
        .chunk_sprites
        .keys()
        .copied()
        .filter(|tile| !target_tiles.contains(tile))
        .collect();
    for tile in stale_tiles {
        if let Some(entry) = lod_state.chunk_sprites.remove(&tile) {
            commands.entity(entry.entity).despawn();
            images.remove(entry.image.id());
        }
    }

    for &tile in &target_tiles {
        if lod_state.chunk_sprites.contains_key(&tile) {
            continue;
        }
        let lod_level = lod_level_for_span(tile.span_chunks);
        let image_handle = images.add(build_lod_tile_image(tile, lod_level, &lod_palette_cache));
        let mut sprite = Sprite::from_image(image_handle.clone());
        sprite.custom_size = Some(Vec2::splat(CHUNK_WORLD_SIZE_M * tile.span_chunks as f32));
        let center = lod_tile_to_world_center(tile);
        let z = LOD_RENDER_Z - lod_level as f32 * 0.001;
        let entity = commands
            .spawn((sprite, Transform::from_xyz(center.x, center.y, z)))
            .id();
        lod_state
            .chunk_sprites
            .insert(tile, TerrainChunkSprite { entity, image: image_handle });
    }

    let mut diagnostic_tiles: Vec<_> = target_tiles
        .iter()
        .map(|tile| LodRenderTile {
            origin_chunk: tile.origin_chunk,
            span_chunks: tile.span_chunks,
        })
        .collect();
    diagnostic_tiles.sort_by_key(|tile| (tile.span_chunks, tile.origin_chunk.y, tile.origin_chunk.x));
    render_diagnostics.lod_visible_tiles = diagnostic_tiles;
}

pub fn sync_dirty_terrain_chunks_to_render(
    mut commands: Commands,
    mut terrain_world: ResMut<TerrainWorld>,
    mut render_state: ResMut<TerrainRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
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

pub fn sync_water_dots_to_render(
    mut commands: Commands,
    particles: Res<ParticleWorld>,
    terrain: Res<TerrainWorld>,
    mut water_state: ResMut<WaterRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let loaded_chunks = terrain.loaded_chunk_coords();
    let stale_chunks: Vec<_> = water_state
        .chunk_sprites
        .keys()
        .copied()
        .filter(|chunk_coord| terrain.chunk(*chunk_coord).is_none())
        .collect();
    for chunk_coord in stale_chunks {
        if let Some(entry) = water_state.chunk_sprites.remove(&chunk_coord) {
            commands.entity(entry.entity).despawn();
            images.remove(entry.image.id());
        }
    }

    let mut created_chunks = false;
    for &chunk_coord in &loaded_chunks {
        if water_state.chunk_sprites.contains_key(&chunk_coord) {
            continue;
        }
        let image_handle = images.add(blank_water_image(CHUNK_PIXEL_SIZE, CHUNK_PIXEL_SIZE));
        let mut sprite = Sprite::from_image(image_handle.clone());
        sprite.custom_size = Some(Vec2::splat(CHUNK_WORLD_SIZE_M));
        let center = chunk_to_world_center(chunk_coord);
        let entity = commands
            .spawn((
                sprite,
                Transform::from_xyz(center.x, center.y, WATER_RENDER_Z),
            ))
            .id();
        water_state.chunk_sprites.insert(
            chunk_coord,
            OverlayChunkSprite {
                entity,
                image: image_handle,
            },
        );
        created_chunks = true;
    }

    let should_update = created_chunks || particles.is_changed() || terrain.is_changed();
    if !should_update {
        return;
    }

    let mut water_by_chunk: HashMap<IVec2, Vec<usize>> = HashMap::new();
    for (idx, &pos) in particles.positions().iter().enumerate() {
        if !matches!(particles.materials()[idx], ParticleMaterial::WaterLiquid) {
            continue;
        }
        water_by_chunk
            .entry(chunk_coord_from_world(pos))
            .or_default()
            .push(idx);
    }

    let draw_radius = nominal_particle_draw_radius_m();
    let draw_radius_px = (draw_radius * WATER_DOT_SCALE).ceil() as i32;
    let dot_threshold = REST_DENSITY * WATER_DOT_THRESHOLD_REST_DENSITY_RATIO;
    let chunk_width = CHUNK_PIXEL_SIZE;
    let chunk_height = CHUNK_PIXEL_SIZE;

    water_state
        .density
        .resize((chunk_width * chunk_height) as usize, 0.0);
    water_state
        .blur_tmp
        .resize((chunk_width * chunk_height) as usize, 0.0);
    water_state
        .blurred_density
        .resize((chunk_width * chunk_height) as usize, 0.0);

    let mut updated_chunks = HashSet::new();
    for &chunk_coord in &loaded_chunks {
        let mut candidate_indices = Vec::new();
        for y in (chunk_coord.y - 1)..=(chunk_coord.y + 1) {
            for x in (chunk_coord.x - 1)..=(chunk_coord.x + 1) {
                if let Some(indices) = water_by_chunk.get(&IVec2::new(x, y)) {
                    candidate_indices.extend(indices.iter().copied());
                }
            }
        }

        water_state.density.fill(0.0);
        let mut pixels = vec![0_u8; (chunk_width * chunk_height * 4) as usize];

        for particle_index in candidate_indices {
            let pos = particles.positions()[particle_index];
            updated_chunks.insert(chunk_coord_from_world(pos));
            let particle_mass = particles.masses()[particle_index];
            let px_center = world_to_chunk_pixel(pos, chunk_coord);
            for py in (px_center.y - draw_radius_px)..=(px_center.y + draw_radius_px) {
                if py < 0 || py >= chunk_height as i32 {
                    continue;
                }
                for px in (px_center.x - draw_radius_px)..=(px_center.x + draw_radius_px) {
                    if px < 0 || px >= chunk_width as i32 {
                        continue;
                    }
                    let sample_world = chunk_pixel_to_world(IVec2::new(px, py), chunk_coord);
                    if is_solid_terrain_at_world(&terrain, sample_world) {
                        continue;
                    }
                    let r2 = (sample_world - pos).length_squared();
                    let w = kernel_poly6_for_render(r2, draw_radius);
                    if w == 0.0 {
                        continue;
                    }
                    let idx = chunk_density_index(px, py, chunk_width, chunk_height);
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
            chunk_width,
            chunk_height,
            WATER_BLUR_RADIUS_DOTS,
        );
        water_state.density = density;

        for py in 0..chunk_height {
            for px in 0..chunk_width {
                let idx = chunk_density_index(px as i32, py as i32, chunk_width, chunk_height);
                let blurred = water_state.blurred_density[idx];
                let coverage = smoothstep(
                    dot_threshold - WATER_DOT_SMOOTH_WIDTH,
                    dot_threshold + WATER_DOT_SMOOTH_WIDTH,
                    blurred,
                );
                if coverage <= 0.0 {
                    continue;
                }
                let sample_world =
                    chunk_pixel_to_world(IVec2::new(px as i32, py as i32), chunk_coord);
                if is_solid_terrain_at_world(&terrain, sample_world) {
                    continue;
                }
                let mut color = water_palette_color(px as i32, py as i32, blurred, dot_threshold);
                color[3] = ((color[3] as f32) * coverage).round().clamp(0.0, 255.0) as u8;
                let offset = idx * 4;
                pixels[offset..offset + 4].copy_from_slice(&color);
            }
        }

        if let Some(entry) = water_state.chunk_sprites.get(&chunk_coord) {
            if let Some(image) = images.get_mut(&entry.image) {
                image.data = Some(pixels);
            }
        }
    }

    for chunk in updated_chunks {
        mark_particle_chunk_updated(&mut render_diagnostics, chunk);
    }
}

pub fn sync_free_particles_to_render(
    mut commands: Commands,
    particles: Res<ParticleWorld>,
    object_world: Res<ObjectWorld>,
    terrain: Res<TerrainWorld>,
    mut free_state: ResMut<FreeParticleRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let loaded_chunks = terrain.loaded_chunk_coords();
    let stale_chunks: Vec<_> = free_state
        .chunk_sprites
        .keys()
        .copied()
        .filter(|chunk_coord| terrain.chunk(*chunk_coord).is_none())
        .collect();
    for chunk_coord in stale_chunks {
        if let Some(entry) = free_state.chunk_sprites.remove(&chunk_coord) {
            commands.entity(entry.entity).despawn();
            images.remove(entry.image.id());
        }
    }

    let mut created_chunks = false;
    for &chunk_coord in &loaded_chunks {
        if free_state.chunk_sprites.contains_key(&chunk_coord) {
            continue;
        }
        let image_handle = images.add(blank_water_image(CHUNK_PIXEL_SIZE, CHUNK_PIXEL_SIZE));
        let mut sprite = Sprite::from_image(image_handle.clone());
        sprite.custom_size = Some(Vec2::splat(CHUNK_WORLD_SIZE_M));
        let center = chunk_to_world_center(chunk_coord);
        let entity = commands
            .spawn((
                sprite,
                Transform::from_xyz(center.x, center.y, FREE_PARTICLE_RENDER_Z),
            ))
            .id();
        free_state.chunk_sprites.insert(
            chunk_coord,
            OverlayChunkSprite {
                entity,
                image: image_handle,
            },
        );
        created_chunks = true;
    }

    let should_update = created_chunks
        || particles.is_changed()
        || object_world.is_changed()
        || terrain.is_changed();
    if !should_update {
        return;
    }

    let mut free_particles_by_chunk: HashMap<IVec2, Vec<usize>> = HashMap::new();
    for (idx, &pos) in particles.positions().iter().enumerate() {
        let material = particles.materials()[idx];
        if matches!(material, ParticleMaterial::WaterLiquid) {
            continue;
        }
        if object_world.object_of_particle(idx).is_some() {
            continue;
        }
        free_particles_by_chunk
            .entry(chunk_coord_from_world(pos))
            .or_default()
            .push(idx);
    }

    let mut updated_chunks = HashSet::new();
    let chunk_width = CHUNK_PIXEL_SIZE;
    let chunk_height = CHUNK_PIXEL_SIZE;
    for &chunk_coord in &loaded_chunks {
        let mut pixels = vec![0_u8; (chunk_width * chunk_height * 4) as usize];
        let mut candidate_indices = Vec::new();
        for y in (chunk_coord.y - 1)..=(chunk_coord.y + 1) {
            for x in (chunk_coord.x - 1)..=(chunk_coord.x + 1) {
                if let Some(indices) = free_particles_by_chunk.get(&IVec2::new(x, y)) {
                    candidate_indices.extend(indices.iter().copied());
                }
            }
        }

        for particle_index in candidate_indices {
            let pos = particles.positions()[particle_index];
            let material = particles.materials()[particle_index];
            updated_chunks.insert(chunk_coord_from_world(pos));

            let radius_m = particle_properties(material).radius_m;
            let radius_px = (radius_m * WATER_DOT_SCALE).ceil().max(1.0) as i32;
            let center_px = world_to_chunk_pixel(pos, chunk_coord);
            let palette = cell_palette_for_particle(material);
            for py in (center_px.y - radius_px)..=(center_px.y + radius_px) {
                if py < 0 || py >= chunk_height as i32 {
                    continue;
                }
                for px in (center_px.x - radius_px)..=(center_px.x + radius_px) {
                    if px < 0 || px >= chunk_width as i32 {
                        continue;
                    }
                    let sample_world = chunk_pixel_to_world(IVec2::new(px, py), chunk_coord);
                    if is_solid_terrain_at_world(&terrain, sample_world) {
                        continue;
                    }
                    if (sample_world - pos).length_squared() > radius_m * radius_m {
                        continue;
                    }
                    let offset = chunk_density_index(px, py, chunk_width, chunk_height) * 4;
                    let palette_index = deterministic_palette_index(px, py);
                    pixels[offset..offset + 4].copy_from_slice(&palette[palette_index]);
                }
            }
        }

        if let Some(entry) = free_state.chunk_sprites.get(&chunk_coord) {
            if let Some(image) = images.get_mut(&entry.image) {
                image.data = Some(pixels);
            }
        }
    }

    for chunk in updated_chunks {
        mark_particle_chunk_updated(&mut render_diagnostics, chunk);
    }
}

pub fn sync_object_sprites_to_render(
    mut commands: Commands,
    mut object_world: ResMut<ObjectWorld>,
    particles: Res<ParticleWorld>,
    mut render_state: ResMut<ObjectRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    if !particles.is_changed() && !object_world.is_changed() {
        return;
    }

    let positions = particles.positions();
    let masses = particles.masses();
    let materials = particles.materials();
    let mut live_ids = Vec::new();
    let mut updated_chunks = HashSet::new();

    for object in object_world.objects_mut() {
        let Some((center, theta)) = object_pose(object, positions, masses) else {
            continue;
        };
        live_ids.push(object.id);
        for &index in &object.particle_indices {
            if index < positions.len() {
                updated_chunks.insert(chunk_coord_from_world(positions[index]));
            }
        }

        let transform = Transform::from_xyz(center.x, center.y, OBJECT_RENDER_Z)
            .with_rotation(Quat::from_rotation_z(theta));
        if let Some(entry) = render_state.object_sprites.get_mut(&object.id) {
            if object.shape_dirty {
                let (image, world_size) = rasterize_object_image(object, materials);
                if images.insert(entry.image.id(), image).is_err() {
                    let (new_image, _) = rasterize_object_image(object, materials);
                    entry.image = images.add(new_image);
                }
                entry.world_size = world_size;
                let mut sprite = Sprite::from_image(entry.image.clone());
                sprite.custom_size = Some(entry.world_size);
                commands.entity(entry.entity).insert(sprite);
                object.shape_dirty = false;
            }
            commands.entity(entry.entity).insert(transform);
            continue;
        }

        let (image, world_size) = rasterize_object_image(object, materials);
        let image_handle = images.add(image);
        let mut sprite = Sprite::from_image(image_handle.clone());
        sprite.custom_size = Some(world_size);
        let entity = commands
            .spawn((
                sprite,
                transform,
                ObjectRenderHandle {
                    object_id: object.id,
                },
            ))
            .id();
        render_state.object_sprites.insert(
            object.id,
            ObjectSprite {
                entity,
                image: image_handle,
                world_size,
            },
        );
        object.shape_dirty = false;
    }

    let stale_ids: Vec<_> = render_state
        .object_sprites
        .keys()
        .copied()
        .filter(|id| !live_ids.contains(id))
        .collect();
    for object_id in stale_ids {
        if let Some(entry) = render_state.object_sprites.remove(&object_id) {
            commands.entity(entry.entity).despawn();
            images.remove(entry.image.id());
        }
    }
    for chunk in updated_chunks {
        mark_particle_chunk_updated(&mut render_diagnostics, chunk);
    }
}

fn visible_chunk_bounds(
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) -> Option<(IVec2, IVec2, IVec2)> {
    let window = windows.iter().next()?;
    let (camera, camera_transform) = cameras.iter().next()?;

    let viewport_corners = [
        Vec2::new(0.0, 0.0),
        Vec2::new(window.width(), 0.0),
        Vec2::new(0.0, window.height()),
        Vec2::new(window.width(), window.height()),
    ];
    let mut min_cell = IVec2::new(i32::MAX, i32::MAX);
    let mut max_cell = IVec2::new(i32::MIN, i32::MIN);
    for corner in viewport_corners {
        let world = camera.viewport_to_world_2d(camera_transform, corner).ok()?;
        let cell = world_to_cell(world);
        min_cell.x = min_cell.x.min(cell.x);
        min_cell.y = min_cell.y.min(cell.y);
        max_cell.x = max_cell.x.max(cell.x);
        max_cell.y = max_cell.y.max(cell.y);
    }

    let min_chunk = IVec2::new(
        min_cell.x.div_euclid(CHUNK_SIZE_I32),
        min_cell.y.div_euclid(CHUNK_SIZE_I32),
    );
    let max_chunk = IVec2::new(
        max_cell.x.div_euclid(CHUNK_SIZE_I32),
        max_cell.y.div_euclid(CHUNK_SIZE_I32),
    );
    let center_cell = world_to_cell(camera_transform.translation().truncate());
    let center_chunk = IVec2::new(
        center_cell.x.div_euclid(CHUNK_SIZE_I32),
        center_cell.y.div_euclid(CHUNK_SIZE_I32),
    );
    Some((min_chunk, max_chunk, center_chunk))
}

fn lod_level_for_outside_distance(outside_distance_chunks: i32) -> u32 {
    let distance_step = LOD_LEVEL_DISTANCE_STEP_CHUNKS.max(1);
    let adjusted = outside_distance_chunks.max(1) - 1;
    1 + adjusted.div_euclid(distance_step) as u32
}

fn lod_span_chunks_for_level(level: u32) -> i32 {
    let mut span = 1_i32;
    let div = LOD_LEVEL_RESOLUTION_DIV.max(2);
    for _ in 0..level {
        span = span.saturating_mul(div);
    }
    span.max(1)
}

fn lod_level_for_span(span_chunks: i32) -> u32 {
    if span_chunks <= 1 {
        return 0;
    }
    let div = LOD_LEVEL_RESOLUTION_DIV.max(2);
    let mut span = 1_i32;
    let mut level = 0_u32;
    while span < span_chunks {
        span = span.saturating_mul(div);
        level = level.saturating_add(1);
    }
    level
}

fn align_chunk_to_lod_tile(chunk_coord: IVec2, span_chunks: i32) -> IVec2 {
    IVec2::new(
        chunk_coord.x.div_euclid(span_chunks) * span_chunks,
        chunk_coord.y.div_euclid(span_chunks) * span_chunks,
    )
}

fn lod_tile_to_world_center(tile: LodTileKey) -> Vec2 {
    let span = tile.span_chunks as f32;
    (tile.origin_chunk.as_vec2() + Vec2::splat(span * 0.5)) * CHUNK_WORLD_SIZE_M
}

fn build_lod_tile_image(
    tile: LodTileKey,
    lod_level: u32,
    lod_palette_cache: &TerrainLodPaletteCache,
) -> Image {
    let mut pixels = vec![0_u8; (CHUNK_PIXEL_SIZE * CHUNK_PIXEL_SIZE * 4) as usize];
    let tile_min_world = tile.origin_chunk.as_vec2() * CHUNK_WORLD_SIZE_M;
    let tile_size_world = CHUNK_WORLD_SIZE_M * tile.span_chunks as f32;
    let tex_size = CHUNK_PIXEL_SIZE as f32;

    for py in 0..CHUNK_PIXEL_SIZE {
        for px in 0..CHUNK_PIXEL_SIZE {
            let sample_world = Vec2::new(
                tile_min_world.x + ((px as f32 + 0.5) / tex_size) * tile_size_world,
                tile_min_world.y + ((py as f32 + 0.5) / tex_size) * tile_size_world,
            );
            let sample_cell = world_to_cell(sample_world);
            let sample_terrain = generated_cell_for_world(sample_cell);
            let TerrainCell::Solid { material, .. } = sample_terrain else {
                continue;
            };
            let palette = lod_palette_cache.palette_for(material, lod_level);

            let palette_index = deterministic_palette_index(sample_cell.x, sample_cell.y);
            let image_y = CHUNK_PIXEL_SIZE - 1 - py;
            let offset = ((image_y * CHUNK_PIXEL_SIZE + px) * 4) as usize;
            pixels[offset..offset + 4].copy_from_slice(&palette[palette_index]);
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

fn build_lod_palette(base_palette: [[u8; 4]; 4], lod_level: u32) -> [[u8; 4]; 4] {
    if lod_level == 0 {
        return base_palette;
    }

    let scale = 1.0 / (LOD_LEVEL_RESOLUTION_DIV.max(2) as f32).powi(lod_level as i32);
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

fn cell_palette(cell: TerrainCell) -> Option<[[u8; 4]; 4]> {
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
                let idx = chunk_density_index(sx, y, width, height);
                accum += src[idx] * weight;
            }
            let idx = chunk_density_index(x, y, width, height);
            tmp[idx] = accum;
        }
    }

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut accum = 0.0;
            for offset in -radius_dots..=radius_dots {
                let sy = (y + offset).clamp(0, height as i32 - 1);
                let weight = kernel[(offset + radius_dots) as usize];
                let idx = chunk_density_index(x, sy, width, height);
                accum += tmp[idx] * weight;
            }
            let idx = chunk_density_index(x, y, width, height);
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

fn object_pose(object: &ObjectData, positions: &[Vec2], masses: &[f32]) -> Option<(Vec2, f32)> {
    if object.particle_indices.is_empty() || object.rest_local.is_empty() {
        return None;
    }

    let mut mass_sum = 0.0;
    let mut center = Vec2::ZERO;
    for &index in &object.particle_indices {
        if index >= positions.len() || index >= masses.len() {
            continue;
        }
        let mass = masses[index];
        mass_sum += mass;
        center += positions[index] * mass;
    }
    if mass_sum <= 1e-6 {
        return None;
    }
    center /= mass_sum;

    let mut a00 = 0.0;
    let mut a01 = 0.0;
    let mut a10 = 0.0;
    let mut a11 = 0.0;
    for (slot, &index) in object.particle_indices.iter().enumerate() {
        if slot >= object.rest_local.len() || index >= positions.len() || index >= masses.len() {
            continue;
        }
        let mass = masses[index];
        let p = positions[index] - center;
        let q = object.rest_local[slot];
        a00 += mass * p.x * q.x;
        a01 += mass * p.x * q.y;
        a10 += mass * p.y * q.x;
        a11 += mass * p.y * q.y;
    }
    let theta = (a10 - a01).atan2(a00 + a11);
    Some((center, theta))
}

fn rasterize_object_image(object: &ObjectData, materials: &[ParticleMaterial]) -> (Image, Vec2) {
    let half_cell = CELL_SIZE_M * 0.5;
    let mut extent = Vec2::splat(half_cell);
    for q in &object.rest_local {
        extent.x = extent.x.max(q.x.abs() + half_cell);
        extent.y = extent.y.max(q.y.abs() + half_cell);
    }

    let world_size = extent * 2.0;
    let width = ((world_size.x * OBJECT_DOT_SCALE).ceil() as u32 + OBJECT_PADDING_PX * 2).max(2);
    let height = ((world_size.y * OBJECT_DOT_SCALE).ceil() as u32 + OBJECT_PADDING_PX * 2).max(2);
    let mut pixels = vec![0_u8; (width * height * 4) as usize];
    let pad = OBJECT_PADDING_PX as f32;
    let seed_material = object
        .particle_indices
        .first()
        .and_then(|&index| materials.get(index))
        .copied()
        .unwrap_or(ParticleMaterial::StoneSolid);
    let palette = cell_palette_for_particle(seed_material);

    for q in &object.rest_local {
        let min_world = *q - Vec2::splat(half_cell);
        let max_world = *q + Vec2::splat(half_cell);
        let min_px = Vec2::new(
            (min_world.x + extent.x) * OBJECT_DOT_SCALE + pad,
            (min_world.y + extent.y) * OBJECT_DOT_SCALE + pad,
        );
        let max_px = Vec2::new(
            (max_world.x + extent.x) * OBJECT_DOT_SCALE + pad,
            (max_world.y + extent.y) * OBJECT_DOT_SCALE + pad,
        );
        let start_x = min_px.x.floor() as i32;
        let end_x = max_px.x.ceil() as i32 - 1;
        let start_y = min_px.y.floor() as i32;
        let end_y = max_px.y.ceil() as i32 - 1;
        let cell_index_x = (q.x / CELL_SIZE_M).round() as i32;
        let cell_index_y = (q.y / CELL_SIZE_M).round() as i32;

        for py in start_y..=end_y {
            if py < 0 || py >= height as i32 {
                continue;
            }
            for px in start_x..=end_x {
                if px < 0 || px >= width as i32 {
                    continue;
                }
                let iy = height as i32 - 1 - py;
                let offset = ((iy as u32 * width + px as u32) * 4) as usize;
                let pattern_x = cell_index_x * CELL_PIXEL_SIZE as i32 + (px - start_x);
                let pattern_y = cell_index_y * CELL_PIXEL_SIZE as i32 + (py - start_y);
                let palette_index = deterministic_palette_index(pattern_x, pattern_y);
                pixels[offset..offset + 4].copy_from_slice(&palette[palette_index]);
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
    let texture_world_size = Vec2::new(
        width as f32 / OBJECT_DOT_SCALE,
        height as f32 / OBJECT_DOT_SCALE,
    );
    (image, texture_world_size)
}

fn cell_palette_for_particle(material: ParticleMaterial) -> [[u8; 4]; 4] {
    match material {
        ParticleMaterial::WaterLiquid => [
            [42, 120, 202, 235],
            [52, 136, 218, 240],
            [65, 152, 228, 245],
            [78, 167, 238, 250],
        ],
        ParticleMaterial::StoneSolid | ParticleMaterial::StoneGranular => [
            [70, 67, 63, 255],
            [83, 79, 74, 255],
            [95, 90, 84, 255],
            [108, 103, 96, 255],
        ],
        ParticleMaterial::SoilSolid | ParticleMaterial::SoilGranular => [
            [105, 79, 56, 255],
            [119, 91, 67, 255],
            [133, 103, 78, 255],
            [147, 115, 88, 255],
        ],
        ParticleMaterial::SandSolid | ParticleMaterial::SandGranular => [
            [170, 150, 110, 255],
            [186, 166, 124, 255],
            [201, 181, 137, 255],
            [216, 196, 150, 255],
        ],
    }
}
