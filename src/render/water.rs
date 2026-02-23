use std::collections::HashSet;

use bevy::prelude::*;

use super::*;

pub fn sync_water_dots_to_render(
    mut commands: Commands,
    particles: Res<ParticleWorld>,
    terrain: Res<TerrainWorld>,
    particle_chunk_cache: Res<ParticleRenderChunkCache>,
    mut water_state: ResMut<WaterRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let _span = tracing::info_span!("render::sync_water_dots_to_render").entered();
    let loaded_chunks = terrain.loaded_chunk_coords();
    {
        let _span = tracing::info_span!("render::water_chunk_sprite_maintenance").entered();
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
        if !(created_chunks || particle_chunk_cache.is_changed() || terrain.is_changed()) {
            return;
        }
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
    {
        let _span = tracing::info_span!("render::water_rasterize_chunks").entered();
        for &chunk_coord in &loaded_chunks {
            let mut candidate_indices = Vec::new();
            for y in (chunk_coord.y - 1)..=(chunk_coord.y + 1) {
                for x in (chunk_coord.x - 1)..=(chunk_coord.x + 1) {
                    if let Some(indices) =
                        particle_chunk_cache.water_by_chunk.get(&IVec2::new(x, y))
                    {
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
                    let mut color =
                        water_palette_color(px as i32, py as i32, blurred, dot_threshold);
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
    }

    for chunk in updated_chunks {
        mark_particle_chunk_updated(&mut render_diagnostics, chunk);
    }
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
