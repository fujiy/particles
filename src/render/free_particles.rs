use std::collections::HashSet;

use bevy::prelude::*;

use super::*;

pub fn sync_free_particles_to_render(
    mut commands: Commands,
    particles: Res<ParticleWorld>,
    terrain: Res<TerrainWorld>,
    particle_chunk_cache: Res<ParticleRenderChunkCache>,
    mut free_state: ResMut<FreeParticleRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let _span = tracing::info_span!("render::sync_free_particles_to_render").entered();
    let loaded_chunks = terrain.loaded_chunk_coords();
    let created_chunks = {
        let _span = tracing::info_span!("render::free_chunk_sprite_maintenance").entered();
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
        created_chunks
    };

    let should_update = created_chunks || particle_chunk_cache.is_changed() || terrain.is_changed();
    if !should_update {
        return;
    }

    let mut updated_chunks = HashSet::new();
    let chunk_width = CHUNK_PIXEL_SIZE;
    let chunk_height = CHUNK_PIXEL_SIZE;
    {
        let _span = tracing::info_span!("render::free_rasterize_chunks").entered();
        for &chunk_coord in &loaded_chunks {
            let mut pixels = vec![0_u8; (chunk_width * chunk_height * 4) as usize];
            let mut candidate_indices = Vec::new();
            for y in (chunk_coord.y - 1)..=(chunk_coord.y + 1) {
                for x in (chunk_coord.x - 1)..=(chunk_coord.x + 1) {
                    if let Some(indices) = particle_chunk_cache.free_by_chunk.get(&IVec2::new(x, y))
                    {
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
    }

    for chunk in updated_chunks {
        mark_particle_chunk_updated(&mut render_diagnostics, chunk);
    }
}
