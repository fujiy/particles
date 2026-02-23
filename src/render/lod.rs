use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use super::*;

pub fn sync_lod_chunks_to_render(
    mut commands: Commands,
    terrain_world: Res<TerrainWorld>,
    particles: Res<ParticleWorld>,
    particle_chunk_cache: Res<ParticleRenderChunkCache>,
    streaming_settings: Res<TerrainStreamingSettings>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    lod_palette_cache: Res<TerrainLodPaletteCache>,
    mut lod_state: ResMut<TerrainLodRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let _span = tracing::info_span!("render::sync_lod_chunks_to_render").entered();
    let Some((min_chunk, max_chunk, center_chunk)) = visible_chunk_bounds(&windows, &cameras)
    else {
        let stale_tiles: Vec<_> = lod_state.chunk_sprites.keys().copied().collect();
        for tile in stale_tiles {
            if let Some(entry) = lod_state.chunk_sprites.remove(&tile) {
                commands.entity(entry.entity).despawn();
                images.remove(entry.image.id());
            }
        }
        lod_state.tiles_with_particles.clear();
        render_diagnostics.lod_visible_tiles.clear();
        return;
    };

    let load_radius = streaming_settings.load_radius_chunks.max(0);
    let visible_chunk_w = (max_chunk.x - min_chunk.x + 1).max(0) as usize;
    let visible_chunk_h = (max_chunk.y - min_chunk.y + 1).max(0) as usize;
    let mut target_tiles = HashSet::with_capacity(visible_chunk_w.saturating_mul(visible_chunk_h));
    {
        let _span = tracing::info_span!("render::lod_target_tiles").entered();
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
    }

    {
        let _span = tracing::info_span!("render::lod_remove_stale_tiles").entered();
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
            lod_state.tiles_with_particles.remove(&tile);
        }
    }

    let terrain_changed = terrain_world.is_changed();
    let particles_changed = particles.is_changed();
    let has_new_tiles = target_tiles
        .iter()
        .any(|tile| !lod_state.chunk_sprites.contains_key(tile));

    let mut terrain_dirty_tiles = HashSet::new();
    if terrain_changed {
        let _span = tracing::info_span!("render::lod_collect_terrain_dirty_tiles").entered();
        for chunk in terrain_world.dirty_chunk_coords() {
            if chunk.x < min_chunk.x
                || chunk.x > max_chunk.x
                || chunk.y < min_chunk.y
                || chunk.y > max_chunk.y
            {
                continue;
            }
            let dx = (chunk.x - center_chunk.x).abs();
            let dy = (chunk.y - center_chunk.y).abs();
            let chebyshev_distance = dx.max(dy);
            if chebyshev_distance <= load_radius {
                continue;
            }
            let outside_distance = chebyshev_distance - load_radius;
            let lod_level = lod_level_for_outside_distance(outside_distance);
            let span_chunks = lod_span_chunks_for_level(lod_level);
            let tile = LodTileKey {
                origin_chunk: align_chunk_to_lod_tile(chunk, span_chunks),
                span_chunks,
            };
            if target_tiles.contains(&tile) {
                terrain_dirty_tiles.insert(tile);
            }
        }
    }

    let terrain_refresh_with_particles = terrain_dirty_tiles.iter().any(|tile| {
        lod_state
            .chunk_sprites
            .get(tile)
            .map(|entry| entry.has_particles)
            .unwrap_or(false)
    });
    let need_particle_collection =
        particles_changed || has_new_tiles || terrain_refresh_with_particles;

    let mut particles_by_tile = HashMap::new();
    let mut particle_tiles_collected = false;
    if need_particle_collection && !target_tiles.is_empty() {
        let _span = tracing::info_span!("render::lod_collect_particles_by_tile").entered();
        particles_by_tile = collect_lod_particles_by_tile(
            &particle_chunk_cache,
            &target_tiles,
            center_chunk,
            load_radius,
            min_chunk,
            max_chunk,
        );
        particle_tiles_collected = true;
    }

    {
        let _span = tracing::info_span!("render::lod_update_tiles").entered();
        let mut tiles_to_update = HashSet::new();
        if terrain_changed {
            tiles_to_update.extend(terrain_dirty_tiles.iter().copied());
        }
        if particles_changed {
            tiles_to_update.extend(particles_by_tile.keys().copied());
            tiles_to_update.extend(lod_state.tiles_with_particles.iter().copied());
        }
        tiles_to_update.retain(|tile| target_tiles.contains(tile));

        {
            let _span = tracing::info_span!("render::lod_update_existing_tiles").entered();
            for tile in tiles_to_update {
                let lod_level = lod_level_for_span(tile.span_chunks);
                let Some(entry) = lod_state.chunk_sprites.get_mut(&tile) else {
                    continue;
                };
                let tile_particle_indices = particles_by_tile
                    .get(&tile)
                    .map(|indices| indices.as_slice())
                    .unwrap_or(&[]);
                let has_particles_now = if particle_tiles_collected {
                    !tile_particle_indices.is_empty()
                } else {
                    entry.has_particles
                };
                let needs_terrain_refresh = terrain_changed && terrain_dirty_tiles.contains(&tile);
                let needs_particle_refresh = if particles_changed {
                    has_particles_now || entry.has_particles
                } else {
                    needs_terrain_refresh && has_particles_now
                };
                if !needs_terrain_refresh && !needs_particle_refresh {
                    continue;
                }

                if needs_terrain_refresh {
                    entry.terrain_pixels = build_lod_tile_terrain_pixels(
                        &terrain_world,
                        tile,
                        lod_level,
                        &lod_palette_cache,
                    );
                }
                let composed = if has_particles_now {
                    compose_lod_tile_pixels(
                        &entry.terrain_pixels,
                        &particles,
                        tile_particle_indices,
                        tile.origin_chunk,
                        tile.span_chunks,
                    )
                } else {
                    entry.terrain_pixels.clone()
                };
                if images
                    .insert(entry.image.id(), image_from_rgba_pixels(composed))
                    .is_err()
                {
                    let composed = if has_particles_now {
                        compose_lod_tile_pixels(
                            &entry.terrain_pixels,
                            &particles,
                            tile_particle_indices,
                            tile.origin_chunk,
                            tile.span_chunks,
                        )
                    } else {
                        entry.terrain_pixels.clone()
                    };
                    entry.image = images.add(image_from_rgba_pixels(composed));
                    let mut sprite = Sprite::from_image(entry.image.clone());
                    sprite.custom_size =
                        Some(Vec2::splat(CHUNK_WORLD_SIZE_M * tile.span_chunks as f32));
                    commands.entity(entry.entity).insert(sprite);
                }
                entry.has_particles = has_particles_now;
                if has_particles_now {
                    lod_state.tiles_with_particles.insert(tile);
                } else {
                    lod_state.tiles_with_particles.remove(&tile);
                }
            }
        }

        if has_new_tiles {
            let _span = tracing::info_span!("render::lod_create_new_tiles").entered();
            for &tile in &target_tiles {
                if lod_state.chunk_sprites.contains_key(&tile) {
                    continue;
                }

                let lod_level = lod_level_for_span(tile.span_chunks);
                let tile_particle_indices = particles_by_tile
                    .get(&tile)
                    .map(|indices| indices.as_slice())
                    .unwrap_or(&[]);
                let has_particles_now = !tile_particle_indices.is_empty();
                let terrain_pixels = build_lod_tile_terrain_pixels(
                    &terrain_world,
                    tile,
                    lod_level,
                    &lod_palette_cache,
                );
                let composed = if has_particles_now {
                    compose_lod_tile_pixels(
                        &terrain_pixels,
                        &particles,
                        tile_particle_indices,
                        tile.origin_chunk,
                        tile.span_chunks,
                    )
                } else {
                    terrain_pixels.clone()
                };
                let image_handle = images.add(image_from_rgba_pixels(composed));
                let mut sprite = Sprite::from_image(image_handle.clone());
                sprite.custom_size =
                    Some(Vec2::splat(CHUNK_WORLD_SIZE_M * tile.span_chunks as f32));
                let center = lod_tile_to_world_center(tile);
                let z = LOD_RENDER_Z - lod_level as f32 * 0.001;
                let entity = commands
                    .spawn((sprite, Transform::from_xyz(center.x, center.y, z)))
                    .id();
                lod_state.chunk_sprites.insert(
                    tile,
                    TerrainLodTileSprite {
                        entity,
                        image: image_handle,
                        terrain_pixels,
                        has_particles: has_particles_now,
                    },
                );
                if has_particles_now {
                    lod_state.tiles_with_particles.insert(tile);
                }
            }
        }
    }

    let mut diagnostic_tiles: Vec<_> = target_tiles
        .iter()
        .map(|tile| LodRenderTile {
            origin_chunk: tile.origin_chunk,
            span_chunks: tile.span_chunks,
        })
        .collect();
    diagnostic_tiles
        .sort_by_key(|tile| (tile.span_chunks, tile.origin_chunk.y, tile.origin_chunk.x));
    render_diagnostics.lod_visible_tiles = diagnostic_tiles;
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

fn collect_lod_particles_by_tile(
    particle_chunk_cache: &ParticleRenderChunkCache,
    target_tiles: &HashSet<LodTileKey>,
    center_chunk: IVec2,
    load_radius: i32,
    min_chunk: IVec2,
    max_chunk: IVec2,
) -> HashMap<LodTileKey, Vec<usize>> {
    let mut by_tile: HashMap<LodTileKey, Vec<usize>> = HashMap::with_capacity(target_tiles.len());
    for (&chunk, indices) in &particle_chunk_cache.all_by_chunk {
        if chunk.x < min_chunk.x
            || chunk.x > max_chunk.x
            || chunk.y < min_chunk.y
            || chunk.y > max_chunk.y
        {
            continue;
        }
        let dx = (chunk.x - center_chunk.x).abs();
        let dy = (chunk.y - center_chunk.y).abs();
        let chebyshev_distance = dx.max(dy);
        if chebyshev_distance <= load_radius {
            continue;
        }
        let outside_distance = chebyshev_distance - load_radius;
        let lod_level = lod_level_for_outside_distance(outside_distance);
        let span_chunks = lod_span_chunks_for_level(lod_level);
        let tile = LodTileKey {
            origin_chunk: align_chunk_to_lod_tile(chunk, span_chunks),
            span_chunks,
        };
        if !target_tiles.contains(&tile) {
            continue;
        }
        by_tile
            .entry(tile)
            .or_default()
            .extend(indices.iter().copied());
    }
    by_tile
}

fn build_lod_tile_terrain_pixels(
    terrain_world: &TerrainWorld,
    tile: LodTileKey,
    lod_level: u32,
    lod_palette_cache: &TerrainLodPaletteCache,
) -> Vec<u8> {
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
            let sample_terrain = terrain_world.get_cell_or_generated(sample_cell);
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

    pixels
}

fn compose_lod_tile_pixels(
    terrain_pixels: &[u8],
    particles: &ParticleWorld,
    particle_indices: &[usize],
    tile_origin_chunk: IVec2,
    tile_span_chunks: i32,
) -> Vec<u8> {
    let mut pixels = terrain_pixels.to_vec();
    rasterize_lod_particles_into_pixels(
        &mut pixels,
        particles,
        particle_indices,
        tile_origin_chunk,
        tile_span_chunks,
    );
    pixels
}

fn rasterize_lod_particles_into_pixels(
    pixels: &mut [u8],
    particles: &ParticleWorld,
    particle_indices: &[usize],
    tile_origin_chunk: IVec2,
    tile_span_chunks: i32,
) {
    let tile_min_world = tile_origin_chunk.as_vec2() * CHUNK_WORLD_SIZE_M;
    let tile_size_world = CHUNK_WORLD_SIZE_M * tile_span_chunks.max(1) as f32;
    let texture_size = CHUNK_PIXEL_SIZE as f32;

    for &index in particle_indices {
        let Some(&pos) = particles.positions().get(index) else {
            continue;
        };
        let local = (pos - tile_min_world) / tile_size_world;
        if local.x < 0.0 || local.x >= 1.0 || local.y < 0.0 || local.y >= 1.0 {
            continue;
        }
        let px = (local.x * texture_size)
            .floor()
            .clamp(0.0, texture_size - 1.0) as u32;
        let py = (local.y * texture_size)
            .floor()
            .clamp(0.0, texture_size - 1.0) as u32;
        let image_y = CHUNK_PIXEL_SIZE - 1 - py;
        let offset = ((image_y * CHUNK_PIXEL_SIZE + px) * 4) as usize;
        let src = lod_particle_color(particles.materials()[index]);
        blend_rgba8_over(&mut pixels[offset..offset + 4], src);
    }
}
