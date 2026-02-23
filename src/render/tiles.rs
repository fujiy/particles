use std::collections::HashSet;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::physics::generation::{TerrainMaterialProbabilities, sample_material_probabilities};
use crate::physics::material::TerrainMaterial;
use crate::physics::world::terrain::generated_cell_for_world;

use super::*;

pub(super) fn sync_required_render_tiles(
    mut commands: Commands,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform, &Projection), With<Camera2d>>,
    mut tile_state: ResMut<TerrainTileRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
    existing_tiles: Query<&RenderTileVisual>,
    mut tile_runtime_query: Query<(&mut Visibility, &mut RenderTileFlags, &mut Transform)>,
) {
    let _span = tracing::info_span!("render::sync_required_render_tiles").entered();

    let Some((min_chunk, max_chunk, zoom_scale)) = visible_chunk_bounds(&windows, &cameras) else {
        let stale_keys: Vec<_> = tile_state.entities_by_key.keys().copied().collect();
        for key in stale_keys {
            if let Some(entity) = tile_state.entities_by_key.remove(&key) {
                if let Ok(visual) = existing_tiles.get(entity) {
                    images.remove(visual.image.id());
                }
                commands.entity(entity).despawn();
            }
        }
        tile_state.target_tiles.clear();
        tile_state.target_lod_level = 0;
        tile_state.target_span_chunks = 1;
        tile_state.evict_frames_outside_required.clear();
        render_diagnostics.visible_tiles.clear();
        return;
    };

    let target_lod_level = lod_level_for_zoom_scale(zoom_scale);
    let target_span_chunks = lod_span_chunks_for_level(target_lod_level);
    let previous_target_tiles = tile_state.target_tiles.clone();
    let target_tiles = {
        let _span = tracing::info_span!("render::required_tile_calc").entered();
        required_tiles_for_view(
            min_chunk,
            max_chunk,
            target_span_chunks,
            &previous_target_tiles,
        )
    };

    {
        let _span = tracing::info_span!("render::tile_reconcile").entered();
        reconcile_tiles_with_hysteresis(
            &mut commands,
            &mut images,
            &existing_tiles,
            &mut tile_runtime_query,
            &mut tile_state,
            &target_tiles,
        );
    }

    tile_state.target_tiles = target_tiles;
    tile_state.target_lod_level = target_lod_level;
    tile_state.target_span_chunks = target_span_chunks;

    let mut visible_tiles: Vec<_> = tile_state
        .target_tiles
        .iter()
        .map(|tile| RenderVisibleTile {
            origin_chunk: tile.origin_chunk,
            span_chunks: tile.span_chunks,
            lod_level: lod_level_for_span(tile.span_chunks),
        })
        .collect();
    visible_tiles.sort_by_key(|tile| (tile.span_chunks, tile.origin_chunk.y, tile.origin_chunk.x));
    render_diagnostics.visible_tiles = visible_tiles;
}

pub(super) fn mark_dirty_render_tiles_from_world(
    mut terrain_world: ResMut<TerrainWorld>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    tile_state: Res<TerrainTileRenderState>,
    mut tile_query: Query<(&RenderTile, &mut RenderTileFlags)>,
) {
    let _span = tracing::info_span!("render::reflect_world_to_tiles").entered();

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

    let dirty_chunks = terrain_world.take_dirty_chunks();
    if dirty_chunks.is_empty() {
        return;
    }

    for &chunk in &dirty_chunks {
        render_diagnostics
            .terrain_updated_chunk_highlight_frames
            .insert(chunk, UPDATED_CHUNK_HIGHLIGHT_FRAMES);
    }

    if tile_state.target_tiles.is_empty() {
        return;
    }
    let dirty_tiles: HashSet<_> = tile_state
        .target_tiles
        .iter()
        .copied()
        .filter(|key| {
            dirty_chunks
                .iter()
                .any(|&chunk| tile_contains_chunk(*key, chunk))
        })
        .collect();

    if dirty_tiles.is_empty() {
        return;
    }

    for (tile, mut flags) in &mut tile_query {
        if dirty_tiles.contains(&tile.key) {
            flags.terrain_dirty = true;
        }
    }
}

pub(super) fn refresh_render_tile_particle_state(
    particles: Res<ParticleWorld>,
    particle_chunk_cache: Res<ParticleRenderChunkCache>,
    tile_state: Res<TerrainTileRenderState>,
    mut tile_query: Query<(&RenderTile, &RenderTileFlags, &mut RenderTileParticleState)>,
) {
    let _span = tracing::info_span!("render::refresh_tile_particle_state").entered();

    if tile_state.target_tiles.is_empty() {
        return;
    }

    let particles_changed = particles.is_changed();
    for (tile, flags, mut particle_state) in &mut tile_query {
        if !tile_state.target_tiles.contains(&tile.key) {
            continue;
        }

        particle_state.changed_this_frame = false;

        let should_resample = particles_changed
            || flags.needs_upload
            || flags.terrain_dirty
            || particle_state.has_particles
            || particle_state.had_particles_last_compose;
        if !should_resample {
            continue;
        }

        let (indices, deferred) = collect_particles_for_tile(&particle_chunk_cache, tile.key);
        // Full-resolution tiles use dedicated particle render passes for live particles.
        // Keep deferred particles composited so particles remain visible after chunk unload.
        let live_indices = if tile.key.span_chunks <= 1 {
            Vec::new()
        } else {
            indices
        };
        let has_particles_now = !live_indices.is_empty() || !deferred.is_empty();
        particle_state.indices = live_indices;
        particle_state.deferred = deferred;
        particle_state.has_particles = has_particles_now;
        particle_state.changed_this_frame =
            particles_changed && (has_particles_now || particle_state.had_particles_last_compose);
    }
}

pub(super) fn refresh_render_tile_terrain_cache(
    terrain_world: Res<TerrainWorld>,
    lod_palette_cache: Res<TerrainLodPaletteCache>,
    tile_state: Res<TerrainTileRenderState>,
    mut tile_query: Query<(&RenderTile, &RenderTileFlags, &mut RenderTileTerrainCache)>,
) {
    let _span = tracing::info_span!("render::refresh_tile_terrain_cache").entered();

    if tile_state.target_tiles.is_empty() {
        return;
    }

    for (tile, flags, mut terrain_cache) in &mut tile_query {
        if !tile_state.target_tiles.contains(&tile.key) {
            continue;
        }

        terrain_cache.rebuilt_this_frame = false;
        if !flags.terrain_dirty && !flags.needs_upload {
            continue;
        }

        terrain_cache.pixels =
            build_tile_terrain_pixels(&terrain_world, tile.key, tile.lod_level, &lod_palette_cache);
        terrain_cache.rebuilt_this_frame = true;
    }
}

pub(super) fn compose_render_tile_upload_buffers(
    particles: Res<ParticleWorld>,
    tile_state: Res<TerrainTileRenderState>,
    mut tile_query: Query<(
        &RenderTile,
        &mut RenderTileFlags,
        &mut RenderTileTerrainCache,
        &mut RenderTileParticleState,
        &mut RenderTileUploadBuffer,
    )>,
) {
    let _span = tracing::info_span!("render::compose_tile_upload_buffers").entered();

    if tile_state.target_tiles.is_empty() {
        return;
    }

    for (tile, mut flags, mut terrain_cache, mut particle_state, mut upload_buffer) in
        &mut tile_query
    {
        if !tile_state.target_tiles.contains(&tile.key) {
            continue;
        }

        let needs_particle_refresh = particle_state.changed_this_frame;
        let needs_upload =
            flags.needs_upload || terrain_cache.rebuilt_this_frame || needs_particle_refresh;
        if !needs_upload {
            continue;
        }

        let composed = if particle_state.has_particles {
            compose_lod_tile_pixels(
                &terrain_cache.pixels,
                &particles,
                &particle_state.indices,
                &particle_state.deferred,
                tile.key.origin_chunk,
                tile.key.span_chunks,
            )
        } else {
            terrain_cache.pixels.clone()
        };

        upload_buffer.pixels = composed;
        upload_buffer.pending = true;

        flags.terrain_dirty = false;
        flags.needs_upload = false;
        terrain_cache.rebuilt_this_frame = false;
        particle_state.had_particles_last_compose = particle_state.has_particles;
        particle_state.changed_this_frame = false;
    }
}

pub(super) fn apply_render_tile_uploads(
    mut images: ResMut<Assets<Image>>,
    mut tile_query: Query<(
        &mut RenderTileVisual,
        &mut Sprite,
        &mut RenderTileUploadBuffer,
    )>,
) {
    let _span = tracing::info_span!("render::texture_upload").entered();
    for (mut visual, mut sprite, mut upload_buffer) in &mut tile_query {
        if !upload_buffer.pending {
            continue;
        }

        let pixels = std::mem::take(&mut upload_buffer.pixels);
        let fallback_pixels = pixels.clone();
        let tile_image = image_from_rgba_pixels(pixels);
        if images.insert(visual.image.id(), tile_image).is_err() {
            let replacement = images.add(image_from_rgba_pixels(fallback_pixels));
            visual.image = replacement.clone();
            sprite.image = replacement;
        }
        upload_buffer.pending = false;
    }
}

fn reconcile_tiles_with_hysteresis(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    existing_tiles: &Query<&RenderTileVisual>,
    tile_runtime_query: &mut Query<(&mut Visibility, &mut RenderTileFlags, &mut Transform)>,
    tile_state: &mut TerrainTileRenderState,
    target_tiles: &HashSet<RenderTileKey>,
) {
    for &key in target_tiles {
        let was_outside_required = tile_state
            .evict_frames_outside_required
            .remove(&key)
            .is_some();
        if let Some(&entity) = tile_state.entities_by_key.get(&key) {
            if let Ok((mut visibility, mut flags, mut transform)) =
                tile_runtime_query.get_mut(entity)
            {
                *visibility = Visibility::Inherited;
                transform.translation.z = render_tile_z(key, lod_level_for_span(key.span_chunks));
                if was_outside_required {
                    flags.terrain_dirty = true;
                    flags.needs_upload = true;
                }
            }
            continue;
        }

        let lod_level = lod_level_for_span(key.span_chunks);
        let image_handle = images.add(blank_water_image(CHUNK_PIXEL_SIZE, CHUNK_PIXEL_SIZE));
        let mut sprite = Sprite::from_image(image_handle.clone());
        sprite.custom_size = Some(Vec2::splat(CHUNK_WORLD_SIZE_M * key.span_chunks as f32));
        let center = tile_to_world_center(key);
        let z = render_tile_z(key, lod_level);
        let entity = commands
            .spawn((
                sprite,
                Transform::from_xyz(center.x, center.y, z),
                RenderTile { key, lod_level },
                RenderTileVisual {
                    image: image_handle,
                },
                RenderTileFlags {
                    terrain_dirty: true,
                    needs_upload: true,
                },
                Visibility::Inherited,
                RenderTileTerrainCache::default(),
                RenderTileParticleState::default(),
                RenderTileUploadBuffer::default(),
            ))
            .id();
        tile_state.entities_by_key.insert(key, entity);
    }

    let mut remove_keys = Vec::new();
    let existing_keys: Vec<_> = tile_state.entities_by_key.keys().copied().collect();
    for key in existing_keys {
        if target_tiles.contains(&key) {
            continue;
        }
        if let Some(&entity) = tile_state.entities_by_key.get(&key) {
            if let Ok((mut visibility, _, _)) = tile_runtime_query.get_mut(entity) {
                *visibility = Visibility::Hidden;
            }
        }
        let next = tile_state
            .evict_frames_outside_required
            .get(&key)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        if next >= RENDER_TILE_EVICT_GRACE_FRAMES {
            remove_keys.push(key);
        } else {
            tile_state.evict_frames_outside_required.insert(key, next);
        }
    }

    for key in remove_keys {
        tile_state.evict_frames_outside_required.remove(&key);
        if let Some(entity) = tile_state.entities_by_key.remove(&key) {
            if let Ok(visual) = existing_tiles.get(entity) {
                images.remove(visual.image.id());
            }
            commands.entity(entity).despawn();
        }
    }
}

fn visible_chunk_bounds(
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &GlobalTransform, &Projection), With<Camera2d>>,
) -> Option<(IVec2, IVec2, f32)> {
    let window = windows.iter().next()?;
    let (camera, camera_transform, projection) = cameras.iter().next()?;

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
    Some((min_chunk, max_chunk, camera_zoom_scale(projection)))
}

fn camera_zoom_scale(projection: &Projection) -> f32 {
    match projection {
        Projection::Orthographic(ortho) => ortho.scale.max(0.0001),
        _ => 1.0,
    }
}

fn required_tiles_for_view(
    min_chunk: IVec2,
    max_chunk: IVec2,
    span_chunks: i32,
    previous_target_tiles: &HashSet<RenderTileKey>,
) -> HashSet<RenderTileKey> {
    let span_chunks = span_chunks.max(1);
    let visible_chunk_w = (max_chunk.x - min_chunk.x + 1).max(0) as usize;
    let visible_chunk_h = (max_chunk.y - min_chunk.y + 1).max(0) as usize;
    let mut target_tiles = HashSet::with_capacity(visible_chunk_w.saturating_mul(visible_chunk_h));
    let mut covered_chunks =
        HashSet::with_capacity(visible_chunk_w.saturating_mul(visible_chunk_h));

    let mut reusable_tiles: Vec<_> = previous_target_tiles
        .iter()
        .copied()
        .filter(|key| {
            key.span_chunks <= span_chunks && tile_intersects_view(*key, min_chunk, max_chunk)
        })
        .collect();
    reusable_tiles.sort_by_key(|key| (key.span_chunks, key.origin_chunk.y, key.origin_chunk.x));

    for key in reusable_tiles {
        if !tile_has_uncovered_view_chunk(key, min_chunk, max_chunk, &covered_chunks) {
            continue;
        }
        target_tiles.insert(key);
        mark_view_chunks_covered(key, min_chunk, max_chunk, &mut covered_chunks);
    }

    for y in min_chunk.y..=max_chunk.y {
        for x in min_chunk.x..=max_chunk.x {
            let chunk = IVec2::new(x, y);
            if covered_chunks.contains(&chunk) {
                continue;
            }
            let key = tile_key_for_chunk(chunk, span_chunks);
            if target_tiles.insert(key) {
                mark_view_chunks_covered(key, min_chunk, max_chunk, &mut covered_chunks);
            }
        }
    }
    target_tiles
}

fn tile_key_for_chunk(chunk_coord: IVec2, span_chunks: i32) -> RenderTileKey {
    let span_chunks = span_chunks.max(1);
    RenderTileKey {
        origin_chunk: align_chunk_to_lod_tile(chunk_coord, span_chunks),
        span_chunks,
    }
}

fn tile_intersects_view(tile: RenderTileKey, min_chunk: IVec2, max_chunk: IVec2) -> bool {
    let span = tile.span_chunks.max(1);
    let tile_min = tile.origin_chunk;
    let tile_max = tile.origin_chunk + IVec2::splat(span - 1);
    tile_max.x >= min_chunk.x
        && tile_min.x <= max_chunk.x
        && tile_max.y >= min_chunk.y
        && tile_min.y <= max_chunk.y
}

fn tile_contains_chunk(tile: RenderTileKey, chunk: IVec2) -> bool {
    let span = tile.span_chunks.max(1);
    chunk.x >= tile.origin_chunk.x
        && chunk.x < tile.origin_chunk.x + span
        && chunk.y >= tile.origin_chunk.y
        && chunk.y < tile.origin_chunk.y + span
}

fn tile_has_uncovered_view_chunk(
    tile: RenderTileKey,
    min_chunk: IVec2,
    max_chunk: IVec2,
    covered_chunks: &HashSet<IVec2>,
) -> bool {
    let span = tile.span_chunks.max(1);
    let min_x = tile.origin_chunk.x.max(min_chunk.x);
    let max_x = (tile.origin_chunk.x + span - 1).min(max_chunk.x);
    let min_y = tile.origin_chunk.y.max(min_chunk.y);
    let max_y = (tile.origin_chunk.y + span - 1).min(max_chunk.y);
    if min_x > max_x || min_y > max_y {
        return false;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if !covered_chunks.contains(&IVec2::new(x, y)) {
                return true;
            }
        }
    }
    false
}

fn mark_view_chunks_covered(
    tile: RenderTileKey,
    min_chunk: IVec2,
    max_chunk: IVec2,
    covered_chunks: &mut HashSet<IVec2>,
) {
    let span = tile.span_chunks.max(1);
    let min_x = tile.origin_chunk.x.max(min_chunk.x);
    let max_x = (tile.origin_chunk.x + span - 1).min(max_chunk.x);
    let min_y = tile.origin_chunk.y.max(min_chunk.y);
    let max_y = (tile.origin_chunk.y + span - 1).min(max_chunk.y);
    if min_x > max_x || min_y > max_y {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            covered_chunks.insert(IVec2::new(x, y));
        }
    }
}

fn lod_level_for_zoom_scale(zoom_scale: f32) -> u32 {
    if !zoom_scale.is_finite() {
        return 0;
    }

    let div = LOD_LEVEL_RESOLUTION_DIV.max(2) as f32;
    let mut level = 0_u32;
    let mut threshold = 1.0_f32;
    let zoom = zoom_scale.max(1.0);
    while level < LOD_PRECOMPUTED_LEVELS && zoom >= threshold * div {
        threshold *= div;
        level = level.saturating_add(1);
    }
    level
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

fn tile_to_world_center(tile: RenderTileKey) -> Vec2 {
    let span = tile.span_chunks as f32;
    (tile.origin_chunk.as_vec2() + Vec2::splat(span * 0.5)) * CHUNK_WORLD_SIZE_M
}

fn render_tile_z(key: RenderTileKey, lod_level: u32) -> f32 {
    if key.span_chunks <= 1 {
        0.0
    } else {
        LOD_RENDER_Z - lod_level as f32 * 0.001
    }
}

fn collect_particles_for_tile(
    particle_chunk_cache: &ParticleRenderChunkCache,
    tile: RenderTileKey,
) -> (Vec<usize>, Vec<DeferredRenderParticle>) {
    let span = tile.span_chunks.max(1);
    let mut indices = Vec::new();
    let mut deferred = Vec::new();
    for y in tile.origin_chunk.y..(tile.origin_chunk.y + span) {
        for x in tile.origin_chunk.x..(tile.origin_chunk.x + span) {
            if let Some(chunk_indices) = particle_chunk_cache.all_by_chunk.get(&IVec2::new(x, y)) {
                indices.extend(chunk_indices.iter().copied());
            }
            if let Some(chunk_deferred) = particle_chunk_cache
                .deferred_by_chunk
                .get(&IVec2::new(x, y))
            {
                deferred.extend(chunk_deferred.iter().copied());
            }
        }
    }
    (indices, deferred)
}

fn build_tile_terrain_pixels(
    terrain_world: &TerrainWorld,
    tile: RenderTileKey,
    lod_level: u32,
    lod_palette_cache: &TerrainLodPaletteCache,
) -> Vec<u8> {
    if tile.span_chunks <= 1 {
        return build_full_chunk_terrain_pixels(terrain_world, tile.origin_chunk);
    }

    let mut pixels = vec![0_u8; (CHUNK_PIXEL_SIZE * CHUNK_PIXEL_SIZE * 4) as usize];
    let tile_min_world = tile.origin_chunk.as_vec2() * CHUNK_WORLD_SIZE_M;
    let tile_size_world = CHUNK_WORLD_SIZE_M * tile.span_chunks as f32;
    let tex_size = CHUNK_PIXEL_SIZE as f32;
    let sample_footprint_m = tile_size_world / tex_size;
    let use_probabilistic_sampling = tile.span_chunks > CELL_PIXEL_SIZE as i32;

    for py in 0..CHUNK_PIXEL_SIZE {
        for px in 0..CHUNK_PIXEL_SIZE {
            let sample_world = Vec2::new(
                tile_min_world.x + ((px as f32 + 0.5) / tex_size) * tile_size_world,
                tile_min_world.y + ((py as f32 + 0.5) / tex_size) * tile_size_world,
            );
            let sample_cell = world_to_cell(sample_world);
            let image_y = CHUNK_PIXEL_SIZE - 1 - py;
            let offset = ((image_y * CHUNK_PIXEL_SIZE + px) * 4) as usize;
            let sample_terrain = terrain_world.get_cell_or_generated(sample_cell);
            let generated_terrain = generated_cell_for_world(sample_cell);

            if sample_terrain != generated_terrain {
                let TerrainCell::Solid { material, .. } = sample_terrain else {
                    continue;
                };
                let palette = lod_palette_cache.palette_for(material, lod_level);
                let palette_index = lod_palette_index_from_world_dot(sample_world);
                pixels[offset..offset + 4].copy_from_slice(&palette[palette_index]);
                continue;
            }

            // Keep deterministic dot-sampled palette up to CELL_PIXEL_SIZE× LOD span.
            if !use_probabilistic_sampling {
                let TerrainCell::Solid { material, .. } = generated_terrain else {
                    continue;
                };
                let palette = lod_palette_cache.palette_for(material, lod_level);
                let palette_index = lod_palette_index_from_world_dot(sample_world);
                pixels[offset..offset + 4].copy_from_slice(&palette[palette_index]);
                continue;
            }

            let probabilities = sample_material_probabilities(sample_world, sample_footprint_m);
            let color = sample_lod_terrain_color_from_probabilities(
                probabilities,
                sample_cell,
                lod_level,
                sample_footprint_m,
                lod_palette_cache,
            );
            if color[3] == 0 {
                continue;
            }
            pixels[offset..offset + 4].copy_from_slice(&color);
        }
    }

    pixels
}

fn lod_palette_index_from_world_dot(world_pos: Vec2) -> usize {
    let dots_per_meter = CELL_PIXEL_SIZE as f32 / CELL_SIZE_M;
    let world_dot_x = (world_pos.x * dots_per_meter).floor() as i32;
    let world_dot_y = (world_pos.y * dots_per_meter).floor() as i32;
    deterministic_palette_index(world_dot_x, world_dot_y)
}

fn sample_lod_terrain_color_from_probabilities(
    probabilities: TerrainMaterialProbabilities,
    sample_cell: IVec2,
    lod_level: u32,
    sample_footprint_m: f32,
    lod_palette_cache: &TerrainLodPaletteCache,
) -> [u8; 4] {
    let sampled_color =
        deterministic_sampled_color(probabilities, sample_cell, lod_level, lod_palette_cache);
    let (expected_color, stderr_per_sample) =
        expected_color_from_probabilities(probabilities, lod_level, lod_palette_cache);
    let expected_blend = expected_color_blend_factor(sample_footprint_m, stderr_per_sample);
    mix_rgba8(sampled_color, expected_color, expected_blend)
}

fn deterministic_sampled_color(
    probabilities: TerrainMaterialProbabilities,
    sample_cell: IVec2,
    lod_level: u32,
    lod_palette_cache: &TerrainLodPaletteCache,
) -> [u8; 4] {
    let sample = deterministic_unit_float(sample_cell.x, sample_cell.y, 0x73a9_f2d1);
    let material = if sample < probabilities.empty {
        None
    } else if sample < probabilities.empty + probabilities.soil {
        Some(TerrainMaterial::Soil)
    } else {
        Some(TerrainMaterial::Stone)
    };
    let Some(material) = material else {
        return [0, 0, 0, 0];
    };
    let palette = lod_palette_cache.palette_for(material, lod_level);
    let palette_index = deterministic_palette_index(sample_cell.x, sample_cell.y);
    palette[palette_index]
}

fn expected_color_from_probabilities(
    probabilities: TerrainMaterialProbabilities,
    lod_level: u32,
    lod_palette_cache: &TerrainLodPaletteCache,
) -> ([u8; 4], f32) {
    let stone = lod_palette_cache.moments_for(TerrainMaterial::Stone, lod_level);
    let soil = lod_palette_cache.moments_for(TerrainMaterial::Soil, lod_level);
    let mut mean = [0.0_f32; 4];
    let mut second = [0.0_f32; 4];
    for channel in 0..4 {
        mean[channel] = probabilities.stone * stone.mean_premul[channel]
            + probabilities.soil * soil.mean_premul[channel];
        second[channel] = probabilities.stone * stone.second_premul[channel]
            + probabilities.soil * soil.second_premul[channel];
    }

    let mut max_var = 0.0_f32;
    for channel in 0..4 {
        let var = (second[channel] - mean[channel] * mean[channel]).max(0.0);
        max_var = max_var.max(var);
    }
    let stderr_per_sample = max_var.sqrt() / 255.0;

    let alpha = mean[3].clamp(0.0, 255.0);
    if alpha <= 1e-5 {
        return ([0, 0, 0, 0], stderr_per_sample);
    }
    let alpha_norm = alpha / 255.0;
    let r = (mean[0] / alpha_norm).clamp(0.0, 255.0);
    let g = (mean[1] / alpha_norm).clamp(0.0, 255.0);
    let b = (mean[2] / alpha_norm).clamp(0.0, 255.0);
    (
        [
            r.round() as u8,
            g.round() as u8,
            b.round() as u8,
            alpha.round() as u8,
        ],
        stderr_per_sample,
    )
}

fn expected_color_blend_factor(sample_footprint_m: f32, stderr_per_sample: f32) -> f32 {
    let footprint_cells = (sample_footprint_m / CELL_SIZE_M).max(0.0);
    if footprint_cells <= 1.0 {
        return 0.0;
    }

    let sample_count = (footprint_cells * footprint_cells).max(1.0);
    let corr_cells = LOD_EXPECTED_CORRELATION_CELLS.max(1e-3);
    let n_eff = (sample_count / (corr_cells * corr_cells)).max(1.0);
    let stderr_norm = stderr_per_sample / n_eff.sqrt();

    if stderr_norm <= LOD_EXPECTED_STDERR_BLEND_LOW {
        return 1.0;
    }
    if stderr_norm >= LOD_EXPECTED_STDERR_BLEND_HIGH {
        return 0.0;
    }
    1.0 - smoothstep(
        LOD_EXPECTED_STDERR_BLEND_LOW,
        LOD_EXPECTED_STDERR_BLEND_HIGH,
        stderr_norm,
    )
}

fn deterministic_unit_float(x: i32, y: i32, salt: u32) -> f32 {
    let mut state = (x as u32).wrapping_mul(0x45d9f3b);
    state ^= (y as u32).wrapping_mul(0x27d4eb2d);
    state ^= salt;
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    state as f32 / u32::MAX as f32
}

fn mix_rgba8(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    [
        ((a[0] as f32 * (1.0 - t)) + (b[0] as f32 * t))
            .round()
            .clamp(0.0, 255.0) as u8,
        ((a[1] as f32 * (1.0 - t)) + (b[1] as f32 * t))
            .round()
            .clamp(0.0, 255.0) as u8,
        ((a[2] as f32 * (1.0 - t)) + (b[2] as f32 * t))
            .round()
            .clamp(0.0, 255.0) as u8,
        ((a[3] as f32 * (1.0 - t)) + (b[3] as f32 * t))
            .round()
            .clamp(0.0, 255.0) as u8,
    ]
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn build_full_chunk_terrain_pixels(terrain_world: &TerrainWorld, chunk_coord: IVec2) -> Vec<u8> {
    let mut pixels = vec![0_u8; (CHUNK_PIXEL_SIZE * CHUNK_PIXEL_SIZE * 4) as usize];
    let chunk_min_cell = chunk_coord * CHUNK_SIZE_I32;

    for local_y in 0..CHUNK_SIZE {
        for local_x in 0..CHUNK_SIZE {
            let local = IVec2::new(local_x as i32, local_y as i32);
            let global_cell = chunk_min_cell + local;
            let cell = terrain_world.get_cell_or_generated(global_cell);
            write_cell_pixels(
                &mut pixels,
                chunk_coord,
                local_x as u32,
                local_y as u32,
                cell,
            );
        }
    }

    pixels
}

fn compose_lod_tile_pixels(
    terrain_pixels: &[u8],
    particles: &ParticleWorld,
    particle_indices: &[usize],
    deferred_particles: &[DeferredRenderParticle],
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
    rasterize_lod_deferred_particles_into_pixels(
        &mut pixels,
        deferred_particles,
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
        let material = particles.materials()[index];
        if !rasterize_lod_particle_disc(
            pixels,
            pos,
            material,
            tile_min_world,
            tile_size_world,
            texture_size,
        ) {
            continue;
        }
    }
}

fn rasterize_lod_deferred_particles_into_pixels(
    pixels: &mut [u8],
    deferred_particles: &[DeferredRenderParticle],
    tile_origin_chunk: IVec2,
    tile_span_chunks: i32,
) {
    let tile_min_world = tile_origin_chunk.as_vec2() * CHUNK_WORLD_SIZE_M;
    let tile_size_world = CHUNK_WORLD_SIZE_M * tile_span_chunks.max(1) as f32;
    let texture_size = CHUNK_PIXEL_SIZE as f32;

    for particle in deferred_particles {
        let _ = rasterize_lod_particle_disc(
            pixels,
            particle.position,
            particle.material,
            tile_min_world,
            tile_size_world,
            texture_size,
        );
    }
}

fn rasterize_lod_particle_disc(
    pixels: &mut [u8],
    world_pos: Vec2,
    material: ParticleMaterial,
    tile_min_world: Vec2,
    tile_size_world: f32,
    texture_size: f32,
) -> bool {
    let local = (world_pos - tile_min_world) / tile_size_world;
    if local.x < 0.0 || local.x >= 1.0 || local.y < 0.0 || local.y >= 1.0 {
        return false;
    }

    let center_x = local.x * texture_size;
    let center_y = local.y * texture_size;
    let radius_world = particle_properties(material).radius_m;
    let radius_px = ((radius_world / tile_size_world) * texture_size).max(0.75);
    let radius2 = radius_px * radius_px;
    let src = lod_particle_color(material);

    let min_x = (center_x - radius_px).floor().max(0.0) as i32;
    let max_x = (center_x + radius_px).ceil().min(texture_size - 1.0) as i32;
    let min_y = (center_y - radius_px).floor().max(0.0) as i32;
    let max_y = (center_y + radius_px).ceil().min(texture_size - 1.0) as i32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = (x as f32 + 0.5) - center_x;
            let dy = (y as f32 + 0.5) - center_y;
            if dx * dx + dy * dy > radius2 {
                continue;
            }
            let image_y = CHUNK_PIXEL_SIZE as i32 - 1 - y;
            let offset = ((image_y as u32 * CHUNK_PIXEL_SIZE + x as u32) * 4) as usize;
            blend_rgba8_over(&mut pixels[offset..offset + 4], src);
        }
    }

    true
}
