use bevy::prelude::*;

use super::*;

pub(super) fn draw_tile_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<TileOverlayState>,
    render_diagnostics: Res<TerrainRenderDiagnostics>,
) {
    if !overlay_state.enabled {
        return;
    }

    for tile in &render_diagnostics.visible_tiles {
        let color = if tile.span_chunks <= 1 {
            GRID_FULL_TILE_COLOR
        } else {
            GRID_LOD_CHUNK_COLOR
        };
        draw_chunk_rect_outline(&mut gizmos, tile.origin_chunk, tile.span_chunks, color);
    }
}

pub(super) fn draw_physics_area_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<PhysicsAreaOverlayState>,
    active_region: Res<PhysicsActiveRegion>,
    region_settings: Res<PhysicsRegionSettings>,
    render_diagnostics: Res<TerrainRenderDiagnostics>,
    object_world: Res<ObjectWorld>,
    particle_world: Res<ParticleWorld>,
) {
    if !overlay_state.enabled {
        return;
    }

    let (Some(min_chunk), Some(max_chunk)) = (active_region.chunk_min, active_region.chunk_max)
    else {
        return;
    };
    let min_x = min_chunk.x as f32 * CHUNK_WORLD_SIZE_M;
    let max_x = (max_chunk.x + 1) as f32 * CHUNK_WORLD_SIZE_M;
    let min_y = min_chunk.y as f32 * CHUNK_WORLD_SIZE_M;
    let max_y = (max_chunk.y + 1) as f32 * CHUNK_WORLD_SIZE_M;

    let neighbor_step = WATER_KERNEL_RADIUS_M;
    let min_neighbor_x = (min_x / neighbor_step).floor() as i32;
    let max_neighbor_x = (max_x / neighbor_step).ceil() as i32;
    let min_neighbor_y = (min_y / neighbor_step).floor() as i32;
    let max_neighbor_y = (max_y / neighbor_step).ceil() as i32;

    for gx in min_neighbor_x..=max_neighbor_x {
        let x = gx as f32 * neighbor_step;
        gizmos.line_2d(
            Vec2::new(x, min_y),
            Vec2::new(x, max_y),
            GRID_NEIGHBOR_COLOR,
        );
    }
    for gy in min_neighbor_y..=max_neighbor_y {
        let y = gy as f32 * neighbor_step;
        gizmos.line_2d(
            Vec2::new(min_x, y),
            Vec2::new(max_x, y),
            GRID_NEIGHBOR_COLOR,
        );
    }

    for &chunk in &active_region.active_chunks {
        draw_chunk_outline(&mut gizmos, chunk, GRID_ACTIVE_CHUNK_COLOR);
    }

    let halo_chunks = region_settings.active_halo_chunks.max(0);
    if halo_chunks > 0 {
        let halo_min = min_chunk - IVec2::splat(halo_chunks);
        let halo_max = max_chunk + IVec2::splat(halo_chunks);
        for y in halo_min.y..=halo_max.y {
            for x in halo_min.x..=halo_max.x {
                let chunk = IVec2::new(x, y);
                if x >= min_chunk.x && x <= max_chunk.x && y >= min_chunk.y && y <= max_chunk.y {
                    continue;
                }
                draw_chunk_outline(&mut gizmos, chunk, GRID_HALO_CHUNK_COLOR);
            }
        }
    }

    for &chunk in render_diagnostics
        .terrain_updated_chunk_highlight_frames
        .keys()
    {
        if chunk.x < min_chunk.x
            || chunk.x > max_chunk.x
            || chunk.y < min_chunk.y
            || chunk.y > max_chunk.y
        {
            continue;
        }
        draw_chunk_outline(&mut gizmos, chunk, GRID_TERRAIN_UPDATED_COLOR);
    }
    for &chunk in render_diagnostics
        .particle_updated_chunk_highlight_frames
        .keys()
    {
        if chunk.x < min_chunk.x
            || chunk.x > max_chunk.x
            || chunk.y < min_chunk.y
            || chunk.y > max_chunk.y
        {
            continue;
        }
        draw_chunk_outline(&mut gizmos, chunk, GRID_PARTICLE_UPDATED_COLOR);
    }

    draw_rect_outline(
        &mut gizmos,
        Vec2::new(min_x, min_y),
        Vec2::new(max_x, max_y),
        GRID_PHYSICS_REGION_COLOR,
    );

    let particle_positions = particle_world.positions();
    let particle_masses = particle_world.masses();
    for object in object_world.objects() {
        if let Some((center, theta)) =
            object_pose_for_overlay(object, particle_positions, particle_masses)
        {
            draw_object_grid_cells(&mut gizmos, object, center, theta);
            draw_object_pose_axes(&mut gizmos, center, theta);
        }
    }
}

fn draw_object_grid_cells(gizmos: &mut Gizmos, object: &ObjectData, center: Vec2, theta: f32) {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let half = CELL_SIZE_M * 0.5;

    for local_center in &object.rest_local {
        let corners = [
            Vec2::new(local_center.x - half, local_center.y - half),
            Vec2::new(local_center.x + half, local_center.y - half),
            Vec2::new(local_center.x + half, local_center.y + half),
            Vec2::new(local_center.x - half, local_center.y + half),
        ];
        let world = corners
            .map(|p| center + Vec2::new(cos_t * p.x - sin_t * p.y, sin_t * p.x + cos_t * p.y));
        gizmos.line_2d(world[0], world[1], GRID_OBJECT_COLOR);
        gizmos.line_2d(world[1], world[2], GRID_OBJECT_COLOR);
        gizmos.line_2d(world[2], world[3], GRID_OBJECT_COLOR);
        gizmos.line_2d(world[3], world[0], GRID_OBJECT_COLOR);
    }
}

fn draw_rect_outline(gizmos: &mut Gizmos, min: Vec2, max: Vec2, color: Color) {
    gizmos.line_2d(Vec2::new(min.x, min.y), Vec2::new(max.x, min.y), color);
    gizmos.line_2d(Vec2::new(max.x, min.y), Vec2::new(max.x, max.y), color);
    gizmos.line_2d(Vec2::new(max.x, max.y), Vec2::new(min.x, max.y), color);
    gizmos.line_2d(Vec2::new(min.x, max.y), Vec2::new(min.x, min.y), color);
}

fn draw_object_pose_axes(gizmos: &mut Gizmos, center: Vec2, theta: f32) {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let local_x = Vec2::new(cos_t, sin_t);
    let local_y = Vec2::new(-sin_t, cos_t);
    let x_end = center + local_x * GRID_OBJECT_AXIS_LENGTH_M;
    let y_end = center + local_y * GRID_OBJECT_AXIS_LENGTH_M;
    gizmos.line_2d(center, x_end, GRID_OBJECT_LOCAL_X_COLOR);
    gizmos.line_2d(center, y_end, GRID_OBJECT_LOCAL_Y_COLOR);
    gizmos.circle_2d(
        center,
        GRID_OBJECT_CENTER_RADIUS_M,
        GRID_OBJECT_CENTER_COLOR,
    );
}

fn object_pose_for_overlay(
    object: &ObjectData,
    positions: &[Vec2],
    masses: &[f32],
) -> Option<(Vec2, f32)> {
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
fn draw_chunk_outline(gizmos: &mut Gizmos, chunk: IVec2, color: Color) {
    draw_chunk_rect_outline(gizmos, chunk, 1, color);
}

fn draw_chunk_rect_outline(
    gizmos: &mut Gizmos,
    origin_chunk: IVec2,
    span_chunks: i32,
    color: Color,
) {
    let span = span_chunks.max(1) as f32;
    let x0 = origin_chunk.x as f32 * CHUNK_WORLD_SIZE_M;
    let x1 = (origin_chunk.x as f32 + span) * CHUNK_WORLD_SIZE_M;
    let y0 = origin_chunk.y as f32 * CHUNK_WORLD_SIZE_M;
    let y1 = (origin_chunk.y as f32 + span) * CHUNK_WORLD_SIZE_M;
    gizmos.line_2d(Vec2::new(x0, y0), Vec2::new(x1, y0), color);
    gizmos.line_2d(Vec2::new(x1, y0), Vec2::new(x1, y1), color);
    gizmos.line_2d(Vec2::new(x1, y1), Vec2::new(x0, y1), color);
    gizmos.line_2d(Vec2::new(x0, y1), Vec2::new(x0, y0), color);
}
