use std::collections::HashSet;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use super::*;
use crate::params::ActiveOverlayParams;
use crate::physics::gpu_mpm::gpu_resources::world_grid_layout;
use crate::physics::world::terrain::CELL_SIZE_M;

#[derive(Component)]
pub(super) struct SdfOverlayNegativeFillCell;

pub(super) fn draw_tile_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<TileOverlayState>,
    overlay_params: Res<ActiveOverlayParams>,
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

    let colors = &overlay_params.0.colors;
    for &chunk in &visible_chunks {
        draw_chunk_outline(&mut gizmos, chunk, colors.grid_chunk_boundary.to_color());
    }

    for chunk in generated_chunk_cache.cached_chunk_coords() {
        if visible_chunks.contains(&chunk) {
            draw_chunk_outline(&mut gizmos, chunk, colors.grid_cached_chunk.to_color());
        }
    }

    for chunk in terrain_world.override_chunk_coords() {
        if visible_chunks.contains(&chunk) {
            draw_chunk_outline(&mut gizmos, chunk, colors.grid_modified_chunk.to_color());
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
    overlay_params: Res<ActiveOverlayParams>,
    terrain_world: Res<TerrainWorld>,
    negative_fill_cells: Query<Entity, With<SdfOverlayNegativeFillCell>>,
) {
    let needs_rebuild =
        overlay_state.is_changed() || terrain_world.is_changed() || overlay_params.is_changed();

    if !overlay_state.enabled {
        for entity in &negative_fill_cells {
            commands.entity(entity).despawn();
        }
        return;
    }

    let Some((world_min, world_max)) = sdf_overlay_world_bounds(&terrain_world) else {
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

    let sdf_params = &overlay_params.0.sdf;
    let step = (sdf_params.step_cell_scale * CELL_SIZE_M).max(1e-3);
    let radius_m = (sdf_params.cell_radius_scale * CELL_SIZE_M).max(1e-4);
    let range_m = (sdf_params.range_cell_scale * CELL_SIZE_M).max(1e-4);
    let negative_fill_z = sdf_params.negative_fill_z;
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
                    let color = sdf_overlay_negative_fill_color(sdf, range_m);
                    if color.to_srgba().alpha > 1e-3 {
                        commands.spawn((
                            Sprite::from_color(color, Vec2::splat(step * 0.98)),
                            Transform::from_xyz(center.x, center.y, negative_fill_z),
                            SdfOverlayNegativeFillCell,
                        ));
                    }
                }
            } else {
                let color = sdf_overlay_cell_color(sdf, range_m);
                if color.to_srgba().alpha <= 1e-3 {
                    continue;
                }
                gizmos.circle_2d(center, radius_m, color).resolution(4);
            }
        }
    }
}

pub(super) fn draw_physics_area_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<PhysicsAreaOverlayState>,
    overlay_params: Res<ActiveOverlayParams>,
    render_diagnostics: Res<TerrainRenderDiagnostics>,
) {
    if !overlay_state.enabled {
        return;
    }
    let colors = &overlay_params.0.colors;

    for &chunk in render_diagnostics
        .terrain_updated_chunk_highlight_frames
        .keys()
    {
        draw_chunk_outline(&mut gizmos, chunk, colors.grid_terrain_updated.to_color());
    }
    for &chunk in render_diagnostics
        .particle_updated_chunk_highlight_frames
        .keys()
    {
        draw_chunk_outline(&mut gizmos, chunk, colors.grid_particle_updated.to_color());
    }

    draw_gpu_grid_overlay(&mut gizmos, None);
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

fn sdf_overlay_world_bounds(terrain_world: &TerrainWorld) -> Option<(Vec2, Vec2)> {
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

fn sdf_overlay_cell_color(sdf: f32, range_m: f32) -> Color {
    let range = range_m.max(1e-4);
    let signed = (sdf / range).clamp(-1.0, 1.0);
    if signed >= 1.0 {
        return Color::srgba(0.0, 0.0, 0.0, 0.0);
    }
    let near = 1.0 - signed;
    Color::srgba(0.10, 0.72, 0.95, 0.06 + 0.24 * near)
}

fn sdf_overlay_negative_fill_color(sdf: f32, range_m: f32) -> Color {
    let range = range_m.max(1e-4);
    let depth = (-sdf / range).clamp(0.0, 1.0);
    Color::srgba(
        0.92,
        0.22 + 0.30 * depth,
        0.10 + 0.12 * depth,
        0.24 + 0.54 * depth,
    )
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
