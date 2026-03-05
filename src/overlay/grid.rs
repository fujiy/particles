use std::collections::HashSet;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use super::*;
use crate::physics::gpu_mpm::gpu_resources::world_grid_layout;
use crate::physics::world::sub_block::{rate_level_from_divisor, sub_block_world_bounds};

#[derive(Component)]
pub(super) struct SdfOverlayNegativeFillCell;

pub(super) fn draw_tile_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<TileOverlayState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    terrain_world: Res<TerrainWorld>,
    generated_chunk_cache: Res<TerrainGeneratedChunkCache>,
) {
    if !overlay_state.enabled {
        return;
    }

    let Some(window) = windows.iter().next() else {
        return;
    };
    let Some((camera, camera_transform)) = camera_query.iter().next() else {
        return;
    };
    let Some(world_minmax) = camera_view_world_bounds(window, camera, camera_transform) else {
        return;
    };
    let (world_min, world_max) = world_minmax;
    let min_chunk = IVec2::new(
        (world_min.x / CHUNK_WORLD_SIZE_M).floor() as i32,
        (world_min.y / CHUNK_WORLD_SIZE_M).floor() as i32,
    );
    let max_chunk = IVec2::new(
        (world_max.x / CHUNK_WORLD_SIZE_M).floor() as i32,
        (world_max.y / CHUNK_WORLD_SIZE_M).floor() as i32,
    );

    let mut visible_chunks = HashSet::new();
    for y in min_chunk.y..=max_chunk.y {
        for x in min_chunk.x..=max_chunk.x {
            visible_chunks.insert(IVec2::new(x, y));
        }
    }

    for &chunk in &visible_chunks {
        draw_chunk_outline(&mut gizmos, chunk, GRID_CHUNK_BOUNDARY_COLOR);
    }

    for chunk in generated_chunk_cache.cached_chunk_coords() {
        if visible_chunks.contains(&chunk) {
            draw_chunk_outline(&mut gizmos, chunk, GRID_CACHED_CHUNK_COLOR);
        }
    }

    for chunk in terrain_world.override_chunk_coords() {
        if visible_chunks.contains(&chunk) {
            draw_chunk_outline(&mut gizmos, chunk, GRID_MODIFIED_CHUNK_COLOR);
        }
    }
}

fn camera_view_world_bounds(
    window: &Window,
    camera: &Camera,
    camera_transform: &GlobalTransform,
) -> Option<(Vec2, Vec2)> {
    let world_a = camera
        .viewport_to_world_2d(camera_transform, Vec2::new(0.0, 0.0))
        .ok()?;
    let world_b = camera
        .viewport_to_world_2d(camera_transform, Vec2::new(window.width(), window.height()))
        .ok()?;
    Some((
        Vec2::new(world_a.x.min(world_b.x), world_a.y.min(world_b.y)),
        Vec2::new(world_a.x.max(world_b.x), world_a.y.max(world_b.y)),
    ))
}

pub(super) fn draw_sdf_overlay(
    mut commands: Commands,
    mut gizmos: Gizmos,
    overlay_state: Res<SdfOverlayState>,
    active_region: Res<PhysicsActiveRegion>,
    region_settings: Res<PhysicsRegionSettings>,
    terrain_world: Res<TerrainWorld>,
    negative_fill_cells: Query<Entity, With<SdfOverlayNegativeFillCell>>,
) {
    let needs_rebuild = overlay_state.is_changed()
        || active_region.is_changed()
        || region_settings.is_changed()
        || terrain_world.is_changed();

    if !overlay_state.enabled {
        for entity in &negative_fill_cells {
            commands.entity(entity).despawn();
        }
        return;
    }

    let Some((world_min, world_max)) =
        sdf_overlay_world_bounds(&active_region, &region_settings, &terrain_world)
    else {
        for entity in &negative_fill_cells {
            commands.entity(entity).despawn();
        }
        return;
    };
    if needs_rebuild {
        for entity in &negative_fill_cells {
            commands.entity(entity).despawn();
        }
    }

    let step = TERRAIN_SDF_OVERLAY_STEP_M.max(1e-3);
    let x0 = (world_min.x / step).floor() as i32;
    let x1 = (world_max.x / step).ceil() as i32;
    let y0 = (world_min.y / step).floor() as i32;
    let y1 = (world_max.y / step).ceil() as i32;

    for gy in y0..=y1 {
        for gx in x0..=x1 {
            let center = Vec2::new((gx as f32 + 0.5) * step, (gy as f32 + 0.5) * step);
            let Some((sdf, _)) = terrain_world.sample_signed_distance_and_normal(center) else {
                continue;
            };
            if sdf <= 0.0 {
                if needs_rebuild {
                    let color = sdf_overlay_negative_fill_color(sdf);
                    if color.to_srgba().alpha > 1e-3 {
                        commands.spawn((
                            Sprite::from_color(color, Vec2::splat(step * 0.98)),
                            Transform::from_xyz(
                                center.x,
                                center.y,
                                TERRAIN_SDF_OVERLAY_NEGATIVE_FILL_Z,
                            ),
                            SdfOverlayNegativeFillCell,
                        ));
                    }
                }
            } else {
                let color = sdf_overlay_cell_color(sdf);
                if color.to_srgba().alpha <= 1e-3 {
                    continue;
                }
                gizmos
                    .circle_2d(center, TERRAIN_SDF_OVERLAY_CELL_RADIUS_M, color)
                    .resolution(4);
            }
        }
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

    let active_world_bounds = if let (Some(min_chunk), Some(max_chunk)) =
        (active_region.chunk_min, active_region.chunk_max)
    {
        let min_x = min_chunk.x as f32 * CHUNK_WORLD_SIZE_M;
        let max_x = (max_chunk.x + 1) as f32 * CHUNK_WORLD_SIZE_M;
        let min_y = min_chunk.y as f32 * CHUNK_WORLD_SIZE_M;
        let max_y = (max_chunk.y + 1) as f32 * CHUNK_WORLD_SIZE_M;
        Some((
            min_chunk,
            max_chunk,
            Vec2::new(min_x, min_y),
            Vec2::new(max_x, max_y),
        ))
    } else {
        None
    };

    if let Some((min_chunk, max_chunk, world_min, world_max)) = active_world_bounds {
        let min_x = world_min.x;
        let max_x = world_max.x;
        let min_y = world_min.y;
        let max_y = world_max.y;

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
                    if x >= min_chunk.x && x <= max_chunk.x && y >= min_chunk.y && y <= max_chunk.y
                    {
                        continue;
                    }
                    draw_chunk_outline(&mut gizmos, chunk, GRID_HALO_CHUNK_COLOR);
                }
            }
        }

        draw_sub_block_rate_digits(
            &mut gizmos,
            &particle_world,
            min_chunk - IVec2::splat(halo_chunks),
            max_chunk + IVec2::splat(halo_chunks),
        );

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

    draw_gpu_grid_overlay(
        &mut gizmos,
        active_world_bounds.map(|(_, _, min, max)| (min, max)),
    );
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

fn draw_gpu_grid_overlay(gizmos: &mut Gizmos, clip_rect: Option<(Vec2, Vec2)>) {
    let layout = world_grid_layout();
    if layout.dims.x < 2 || layout.dims.y < 2 {
        return;
    }

    let h = CELL_SIZE_M.max(1e-6);
    let min_node = layout.origin;
    let max_node = layout.origin + IVec2::new(layout.dims.x as i32 - 1, layout.dims.y as i32 - 1);
    let grid_min = min_node.as_vec2() * h;
    let grid_max = max_node.as_vec2() * h;
    let Some((draw_min, draw_max)) = clipped_rect(grid_min, grid_max, clip_rect) else {
        return;
    };

    let outline_color = Color::srgba(0.98, 0.48, 0.22, 0.88);
    let minor_line_color = Color::srgba(0.98, 0.48, 0.22, 0.14);
    let major_line_color = Color::srgba(0.98, 0.48, 0.22, 0.34);
    draw_rect_outline(gizmos, grid_min, grid_max, outline_color);

    let start_x =
        (((draw_min.x - grid_min.x) / h).floor() as i32).clamp(0, max_node.x - min_node.x);
    let end_x = (((draw_max.x - grid_min.x) / h).ceil() as i32).clamp(0, max_node.x - min_node.x);
    let start_y =
        (((draw_min.y - grid_min.y) / h).floor() as i32).clamp(0, max_node.y - min_node.y);
    let end_y = (((draw_max.y - grid_min.y) / h).ceil() as i32).clamp(0, max_node.y - min_node.y);

    for ix in start_x..=end_x {
        let x = grid_min.x + ix as f32 * h;
        let node_x = min_node.x + ix;
        let color = if node_x.rem_euclid(CHUNK_SIZE_I32) == 0 {
            major_line_color
        } else {
            minor_line_color
        };
        gizmos.line_2d(Vec2::new(x, draw_min.y), Vec2::new(x, draw_max.y), color);
    }
    for iy in start_y..=end_y {
        let y = grid_min.y + iy as f32 * h;
        let node_y = min_node.y + iy;
        let color = if node_y.rem_euclid(CHUNK_SIZE_I32) == 0 {
            major_line_color
        } else {
            minor_line_color
        };
        gizmos.line_2d(Vec2::new(draw_min.x, y), Vec2::new(draw_max.x, y), color);
    }
}

fn clipped_rect(min: Vec2, max: Vec2, clip_rect: Option<(Vec2, Vec2)>) -> Option<(Vec2, Vec2)> {
    let (clip_min, clip_max) = clip_rect.unwrap_or((min, max));
    let draw_min = Vec2::new(min.x.max(clip_min.x), min.y.max(clip_min.y));
    let draw_max = Vec2::new(max.x.min(clip_max.x), max.y.min(clip_max.y));
    if draw_min.x > draw_max.x || draw_min.y > draw_max.y {
        None
    } else {
        Some((draw_min, draw_max))
    }
}

fn sdf_overlay_world_bounds(
    active_region: &PhysicsActiveRegion,
    region_settings: &PhysicsRegionSettings,
    terrain_world: &TerrainWorld,
) -> Option<(Vec2, Vec2)> {
    if let (Some(min_chunk), Some(max_chunk)) = (active_region.chunk_min, active_region.chunk_max) {
        let halo_chunks = region_settings.active_halo_chunks.max(0);
        let overlay_min_chunk = min_chunk - IVec2::splat(halo_chunks);
        let overlay_max_chunk = max_chunk + IVec2::splat(halo_chunks);
        return Some((
            overlay_min_chunk.as_vec2() * CHUNK_WORLD_SIZE_M,
            (overlay_max_chunk + IVec2::ONE).as_vec2() * CHUNK_WORLD_SIZE_M,
        ));
    }
    let positions = terrain_world.static_particle_positions();
    if positions.is_empty() {
        return None;
    }
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for &p in positions {
        min = min.min(p);
        max = max.max(p);
    }
    let margin = Vec2::splat(CHUNK_WORLD_SIZE_M * 0.5);
    Some((min - margin, max + margin))
}

fn sdf_overlay_cell_color(sdf: f32) -> Color {
    let range = TERRAIN_SDF_OVERLAY_RANGE_M.max(1e-4);
    let signed = (sdf / range).clamp(-1.0, 1.0);
    if signed >= 1.0 {
        return Color::srgba(0.0, 0.0, 0.0, 0.0);
    }
    let near = 1.0 - signed;
    Color::srgba(0.10, 0.72, 0.95, 0.06 + 0.24 * near)
}

fn sdf_overlay_negative_fill_color(sdf: f32) -> Color {
    let range = TERRAIN_SDF_OVERLAY_RANGE_M.max(1e-4);
    let depth = (-sdf / range).clamp(0.0, 1.0);
    Color::srgba(
        0.92,
        0.22 + 0.30 * depth,
        0.10 + 0.12 * depth,
        0.24 + 0.54 * depth,
    )
}

fn blend_color(from: Color, to: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let from = from.to_srgba();
    let to = to.to_srgba();
    Color::srgba(
        from.red + (to.red - from.red) * t,
        from.green + (to.green - from.green) * t,
        from.blue + (to.blue - from.blue) * t,
        from.alpha + (to.alpha - from.alpha) * t,
    )
}

fn draw_sub_block_rate_digits(
    gizmos: &mut Gizmos,
    particle_world: &ParticleWorld,
    min_chunk: IVec2,
    max_chunk: IVec2,
) {
    let sub_block_size_cells = particle_world.sub_block_size_cells();
    let min_world = min_chunk.as_vec2() * CHUNK_WORLD_SIZE_M;
    let max_world = (max_chunk + IVec2::ONE).as_vec2() * CHUNK_WORLD_SIZE_M;
    for sample in particle_world.sub_block_overlay_samples() {
        let (min, max) = sub_block_world_bounds(sample.coord, sub_block_size_cells);
        if max.x < min_world.x || min.x > max_world.x || max.y < min_world.y || min.y > max_world.y
        {
            continue;
        }
        let center = (min + max) * 0.5;
        let size = ((max.x - min.x).min(max.y - min.y) * 0.28).max(0.06);
        let color = blend_color(
            GRID_SUB_BLOCK_LABEL_BASE_COLOR,
            GRID_SUB_BLOCK_DEBT_HOT_COLOR,
            sample.debt_ratio,
        );
        let level =
            rate_level_from_divisor(sample.rate_divisor, particle_world.sub_block_max_level());
        draw_number_stroke(gizmos, level, center, size, color);
    }
}

fn draw_number_stroke(gizmos: &mut Gizmos, value: u8, center: Vec2, size: f32, color: Color) {
    let text = value.to_string();
    let count = text.len() as f32;
    let advance = size * 1.15;
    let start_x = center.x - (count - 1.0) * advance * 0.5;
    for (index, ch) in text.chars().enumerate() {
        let Some(digit) = ch.to_digit(10) else {
            continue;
        };
        let digit_center = Vec2::new(start_x + index as f32 * advance, center.y);
        draw_digit_stroke(gizmos, digit as u8, digit_center, size, color);
    }
}

fn draw_digit_stroke(gizmos: &mut Gizmos, digit: u8, center: Vec2, size: f32, color: Color) {
    let x0 = center.x - size * 0.45;
    let x1 = center.x + size * 0.45;
    let y0 = center.y - size * 0.70;
    let y1 = center.y + size * 0.70;
    let ym = (y0 + y1) * 0.5;
    let stroke = |gizmos: &mut Gizmos, a: Vec2, b: Vec2| {
        gizmos.line_2d(a, b, color);
    };
    let draw_segments = |gizmos: &mut Gizmos, segments: [bool; 7]| {
        if segments[0] {
            stroke(gizmos, Vec2::new(x0, y1), Vec2::new(x1, y1));
        }
        if segments[1] {
            stroke(gizmos, Vec2::new(x1, y1), Vec2::new(x1, ym));
        }
        if segments[2] {
            stroke(gizmos, Vec2::new(x1, ym), Vec2::new(x1, y0));
        }
        if segments[3] {
            stroke(gizmos, Vec2::new(x0, y0), Vec2::new(x1, y0));
        }
        if segments[4] {
            stroke(gizmos, Vec2::new(x0, ym), Vec2::new(x0, y0));
        }
        if segments[5] {
            stroke(gizmos, Vec2::new(x0, y1), Vec2::new(x0, ym));
        }
        if segments[6] {
            stroke(gizmos, Vec2::new(x0, ym), Vec2::new(x1, ym));
        }
    };
    let segments = match digit {
        0 => [true, true, true, true, true, true, false],
        1 => [false, true, true, false, false, false, false],
        2 => [true, true, false, true, true, false, true],
        3 => [true, true, true, true, false, false, true],
        4 => [false, true, true, false, false, true, true],
        5 => [true, false, true, true, false, true, true],
        6 => [true, false, true, true, true, true, true],
        7 => [true, true, true, false, false, false, false],
        8 => [true, true, true, true, true, true, true],
        9 => [true, true, true, true, false, true, true],
        _ => [true, true, true, true, true, true, true],
    };
    draw_segments(gizmos, segments);
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
