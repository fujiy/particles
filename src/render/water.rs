use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use super::*;
use crate::physics::material::particle_properties;
use crate::physics::world::constants::CELL_SIZE_M;
use crate::physics::world::grid::GridHierarchy;
use crate::physics::world::particle::ParticleMaterial;

pub fn sync_water_dots_to_render(
    mut commands: Commands,
    terrain: Res<TerrainWorld>,
    grid_hierarchy: Res<GridHierarchy>,
    particle_chunk_cache: Res<ParticleRenderChunkCache>,
    mut water_state: ResMut<WaterRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let _span = tracing::info_span!("render::sync_water_dots_to_render").entered();
    let has_any_water = !particle_chunk_cache.water_by_chunk.is_empty();
    let target_render_chunks = collect_water_target_render_chunks(&terrain, &particle_chunk_cache);
    let target_changed = target_render_chunks != water_state.rendered_chunks_last_frame;
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
        for &chunk_coord in &target_render_chunks {
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
        if !has_any_water {
            if water_state.had_any_water {
                let _span = tracing::info_span!("render::water_clear_when_empty").entered();
                let blank = vec![0_u8; (CHUNK_PIXEL_SIZE * CHUNK_PIXEL_SIZE * 4) as usize];
                for chunk in &water_state.rendered_chunks_last_frame {
                    if let Some(entry) = water_state.chunk_sprites.get(chunk) {
                        if let Some(image) = images.get_mut(&entry.image) {
                            image.data = Some(blank.clone());
                        }
                    }
                }
                water_state.had_any_water = false;
            }
            water_state.rendered_chunks_last_frame.clear();
            return;
        }

        if !(created_chunks
            || grid_hierarchy.is_changed()
            || particle_chunk_cache.is_changed()
            || terrain.is_changed()
            || target_changed)
        {
            return;
        }
    }
    water_state.had_any_water = true;
    let rendered_chunks_prev = water_state.rendered_chunks_last_frame.clone();

    if grid_hierarchy.blocks().is_empty() {
        return;
    }
    let dot_threshold = REST_DENSITY * WATER_GRID_DENSITY_THRESHOLD_REST_DENSITY_RATIO;
    let chunk_width = CHUNK_PIXEL_SIZE;
    let chunk_height = CHUNK_PIXEL_SIZE;
    let splash_points_by_chunk = collect_splash_points_by_chunk(&grid_hierarchy);

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
        let mut render_chunks: Vec<_> = target_render_chunks.iter().copied().collect();
        render_chunks.sort_by_key(|chunk| (chunk.y, chunk.x));
        for chunk_coord in render_chunks {
            water_state.density.fill(0.0);
            let mut pixels = vec![0_u8; (chunk_width * chunk_height * 4) as usize];

            for py in 0..chunk_height as i32 {
                for px in 0..chunk_width as i32 {
                    let sample_world = chunk_pixel_to_world(IVec2::new(px, py), chunk_coord);
                    if is_solid_terrain_at_world(&terrain, sample_world) {
                        continue;
                    }
                    let idx = chunk_density_index(px, py, chunk_width, chunk_height);
                    water_state.density[idx] =
                        sample_grid_density_at_world(&grid_hierarchy, sample_world);
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
            if let Some(points) = splash_points_by_chunk.get(&chunk_coord) {
                draw_splash_points_for_chunk(
                    &terrain,
                    chunk_coord,
                    &mut pixels,
                    points,
                    dot_threshold,
                );
            }

            if let Some(entry) = water_state.chunk_sprites.get(&chunk_coord) {
                if let Some(image) = images.get_mut(&entry.image) {
                    image.data = Some(pixels);
                }
            }
            updated_chunks.insert(chunk_coord);
        }
    }

    let blank = vec![0_u8; (CHUNK_PIXEL_SIZE * CHUNK_PIXEL_SIZE * 4) as usize];
    for chunk in rendered_chunks_prev.difference(&target_render_chunks) {
        if let Some(entry) = water_state.chunk_sprites.get(chunk) {
            if let Some(image) = images.get_mut(&entry.image) {
                image.data = Some(blank.clone());
            }
        }
    }
    water_state.rendered_chunks_last_frame = target_render_chunks;

    for chunk in updated_chunks {
        mark_particle_chunk_updated(&mut render_diagnostics, chunk);
    }
}

fn collect_water_target_render_chunks(
    terrain: &TerrainWorld,
    particle_chunk_cache: &ParticleRenderChunkCache,
) -> HashSet<IVec2> {
    let mut target = HashSet::new();
    for &chunk in particle_chunk_cache.water_by_chunk.keys() {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let candidate = chunk + IVec2::new(dx, dy);
                if terrain.chunk(candidate).is_some() {
                    target.insert(candidate);
                }
            }
        }
    }
    target
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

fn node_scale_for_h(h: f32) -> i32 {
    ((h.max(1e-6) / CELL_SIZE_M).round() as i32).max(1)
}

fn sample_grid_density_at_world(grid_hierarchy: &GridHierarchy, world_pos: Vec2) -> f32 {
    let (h, scale) = match grid_hierarchy
        .block_index_for_position(world_pos)
        .and_then(|block_index| grid_hierarchy.blocks().get(block_index))
    {
        Some(block) => (block.h_b.max(1e-6), node_scale_for_h(block.h_b)),
        None => {
            let Some(block) = grid_hierarchy.blocks().first() else {
                return 0.0;
            };
            (block.h_b.max(1e-6), node_scale_for_h(block.h_b))
        }
    };
    let inv_h = 1.0 / h.max(1e-6);
    let node_f = world_pos * inv_h;
    let x0 = node_f.x.floor() as i32;
    let y0 = node_f.y.floor() as i32;
    let tx = node_f.x - x0 as f32;
    let ty = node_f.y - y0 as f32;

    let m00 = sample_grid_node_mass(grid_hierarchy, IVec2::new(x0, y0) * scale);
    let m10 = sample_grid_node_mass(grid_hierarchy, IVec2::new(x0 + 1, y0) * scale);
    let m01 = sample_grid_node_mass(grid_hierarchy, IVec2::new(x0, y0 + 1) * scale);
    let m11 = sample_grid_node_mass(grid_hierarchy, IVec2::new(x0 + 1, y0 + 1) * scale);

    let mx0 = m00 + (m10 - m00) * tx;
    let mx1 = m01 + (m11 - m01) * tx;
    let m = mx0 + (mx1 - mx0) * ty;
    m.max(0.0) / (h * h).max(1e-6)
}

fn sample_grid_node_mass(grid_hierarchy: &GridHierarchy, world_node: IVec2) -> f32 {
    grid_hierarchy
        .node_by_world(world_node)
        .map(|node| node.m)
        .unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug)]
struct RenderSplashPoint {
    world_pos: Vec2,
    equivalent_particle_count: f32,
}

fn collect_splash_points_by_chunk(
    grid_hierarchy: &GridHierarchy,
) -> HashMap<IVec2, Vec<RenderSplashPoint>> {
    let mut points_by_chunk = HashMap::<IVec2, Vec<RenderSplashPoint>>::new();
    let water_particle_mass = particle_properties(ParticleMaterial::WaterLiquid)
        .mass
        .max(1e-6);
    for block in grid_hierarchy.blocks() {
        let width = block.node_dims.x as usize;
        let height = block.node_dims.y as usize;
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let node = block.nodes()[idx];
                let render_mass = node.render_mass_sum;
                if render_mass <= 1e-6 {
                    continue;
                }
                let equivalent_particle_count = render_mass / water_particle_mass;
                if equivalent_particle_count > WATER_SPLASH_MAX_EQUIV_PARTICLE_COUNT {
                    continue;
                }

                let world_node = block.origin_node + IVec2::new(x as i32, y as i32);
                let default_pos = world_node.as_vec2() * block.h_b;
                let world_pos = if node.render_mass_pos_sum.is_finite() && render_mass > 1e-6 {
                    node.render_mass_pos_sum / render_mass
                } else {
                    default_pos
                };
                if !world_pos.is_finite() {
                    continue;
                }
                let chunk = chunk_coord_from_world(world_pos);
                points_by_chunk
                    .entry(chunk)
                    .or_default()
                    .push(RenderSplashPoint {
                        world_pos,
                        equivalent_particle_count,
                    });
            }
        }
    }
    points_by_chunk
}

fn draw_splash_points_for_chunk(
    terrain: &TerrainWorld,
    chunk_coord: IVec2,
    pixels: &mut [u8],
    points: &[RenderSplashPoint],
    dot_threshold: f32,
) {
    for point in points {
        let radius = if point.equivalent_particle_count <= WATER_SPLASH_SINGLE_EQUIV_PARTICLE_COUNT
        {
            WATER_ISOLATED_SPLASH_RADIUS_DOTS
        } else {
            ((WATER_ISOLATED_SPLASH_RADIUS_DOTS as f32) * point.equivalent_particle_count.sqrt())
                .ceil()
                .max(WATER_ISOLATED_SPLASH_RADIUS_DOTS as f32) as i32
        };
        draw_splash_point(
            terrain,
            chunk_coord,
            point.world_pos,
            pixels,
            radius,
            dot_threshold,
        );
    }
}

fn draw_splash_point(
    terrain: &TerrainWorld,
    chunk_coord: IVec2,
    world_pos: Vec2,
    pixels: &mut [u8],
    radius_px: i32,
    dot_threshold: f32,
) {
    if radius_px <= 0 {
        return;
    }
    let center_px = world_to_chunk_pixel(world_pos, chunk_coord);
    let radius_sq = radius_px * radius_px;
    for py in (center_px.y - radius_px)..=(center_px.y + radius_px) {
        if py < 0 || py >= CHUNK_PIXEL_SIZE as i32 {
            continue;
        }
        for px in (center_px.x - radius_px)..=(center_px.x + radius_px) {
            if px < 0 || px >= CHUNK_PIXEL_SIZE as i32 {
                continue;
            }
            let d = IVec2::new(px - center_px.x, py - center_px.y);
            if d.length_squared() > radius_sq {
                continue;
            }
            let sample_world = chunk_pixel_to_world(IVec2::new(px, py), chunk_coord);
            if is_solid_terrain_at_world(terrain, sample_world) {
                continue;
            }
            let mut splash_color = water_palette_color(px, py, dot_threshold * 2.0, dot_threshold);
            splash_color[3] = ((splash_color[3] as f32) * 0.92).round().clamp(0.0, 255.0) as u8;
            let idx = chunk_density_index(px, py, CHUNK_PIXEL_SIZE, CHUNK_PIXEL_SIZE) * 4;
            blend_rgba8_over(&mut pixels[idx..idx + 4], splash_color);
        }
    }
}
