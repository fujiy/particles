use bevy::prelude::*;

use super::*;
use crate::params::ActiveOverlayParams;
use crate::physics::world::terrain::CELL_SIZE_M;

#[derive(Component)]
pub(super) struct SdfOverlayNegativeFillCell;

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
