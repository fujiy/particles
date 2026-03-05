use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use super::*;
use crate::render::TerrainGeneratedChunkCache;

const TOOL_PREFETCH_RADIUS_CHUNKS: i32 = 1;

pub(super) fn handle_world_interactions(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    solver_params: Res<SolverParams>,
    material_params: Res<MaterialParams>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    ui_buttons: Query<&Interaction, With<Button>>,
    mut interaction_state: ResMut<WorldInteractionState>,
    save_load_ui_state: Res<SaveLoadUiState>,
    mut particle_world: ResMut<ParticleWorld>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut generated_chunk_cache: ResMut<TerrainGeneratedChunkCache>,
    mut object_world: ResMut<ObjectWorld>,
    mut prefetched_tool: Local<Option<WorldTool>>,
) {
    if save_load_ui_state.mode.is_some() {
        let selected_tool = interaction_state.selected_tool;
        finalize_stone_stroke(
            &mut interaction_state.stone_stroke,
            selected_tool,
            &mut particle_world,
            &mut object_world,
            &solver_params,
        );
        interaction_state.last_drag_world = None;
        interaction_state.last_water_spawn_cell = None;
        interaction_state.last_granular_spawn_cell = None;
        return;
    }

    if keyboard.just_pressed(KeyCode::Digit1) || keyboard.just_pressed(KeyCode::Numpad1) {
        select_world_tool(&mut interaction_state, Some(WorldTool::WaterLiquid));
    } else if keyboard.just_pressed(KeyCode::Digit2) || keyboard.just_pressed(KeyCode::Numpad2) {
        select_world_tool(&mut interaction_state, Some(WorldTool::StoneSolid));
    } else if keyboard.just_pressed(KeyCode::Digit3) || keyboard.just_pressed(KeyCode::Numpad3) {
        select_world_tool(&mut interaction_state, Some(WorldTool::StoneGranular));
    } else if keyboard.just_pressed(KeyCode::Digit4) || keyboard.just_pressed(KeyCode::Numpad4) {
        select_world_tool(&mut interaction_state, Some(WorldTool::SoilSolid));
    } else if keyboard.just_pressed(KeyCode::Digit5) || keyboard.just_pressed(KeyCode::Numpad5) {
        select_world_tool(&mut interaction_state, Some(WorldTool::SoilGranular));
    } else if keyboard.just_pressed(KeyCode::Digit6) || keyboard.just_pressed(KeyCode::Numpad6) {
        select_world_tool(&mut interaction_state, Some(WorldTool::SandSolid));
    } else if keyboard.just_pressed(KeyCode::Digit7) || keyboard.just_pressed(KeyCode::Numpad7) {
        select_world_tool(&mut interaction_state, Some(WorldTool::SandGranular));
    } else if keyboard.just_pressed(KeyCode::Digit8) || keyboard.just_pressed(KeyCode::Numpad8) {
        select_world_tool(&mut interaction_state, Some(WorldTool::Break));
    } else if keyboard.just_pressed(KeyCode::Digit9) || keyboard.just_pressed(KeyCode::Numpad9) {
        select_world_tool(&mut interaction_state, Some(WorldTool::Delete));
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        select_world_tool(&mut interaction_state, None);
    }

    let left_pressed = mouse_buttons.pressed(MouseButton::Left);
    let alt_pressed = keyboard.pressed(KeyCode::AltLeft) || keyboard.pressed(KeyCode::AltRight);
    let blocked_by_pan = alt_pressed || mouse_buttons.pressed(MouseButton::Middle);
    let blocked_by_ui = ui_buttons
        .iter()
        .any(|interaction| *interaction != Interaction::None);
    let cursor_world = cursor_world_position(&windows, &camera_query);
    let selected_tool = interaction_state.selected_tool;
    if selected_tool.is_none() {
        *prefetched_tool = None;
    }
    if let (Some(tool), Some(cursor_world)) = (selected_tool, cursor_world) {
        if Some(tool) != *prefetched_tool {
            if tool_uses_generated_chunk_data(tool) {
                let center_cell = world_to_cell(cursor_world);
                let center_chunk = IVec2::new(
                    center_cell.x.div_euclid(CHUNK_SIZE_I32),
                    center_cell.y.div_euclid(CHUNK_SIZE_I32),
                );
                generated_chunk_cache
                    .enqueue_prefetch_square(center_chunk, TOOL_PREFETCH_RADIUS_CHUNKS);
            }
            *prefetched_tool = Some(tool);
        }
    }

    if !left_pressed || blocked_by_pan || blocked_by_ui || cursor_world.is_none() {
        finalize_stone_stroke(
            &mut interaction_state.stone_stroke,
            selected_tool,
            &mut particle_world,
            &mut object_world,
            &solver_params,
        );
        interaction_state.last_drag_world = None;
        interaction_state.last_water_spawn_cell = None;
        interaction_state.last_granular_spawn_cell = None;
        return;
    }

    let cursor_world = cursor_world.unwrap_or(Vec2::ZERO);

    let previous_world = interaction_state.last_drag_world.unwrap_or(cursor_world);
    let dt = time.delta_secs().max(1e-4);
    let stroke_velocity = (cursor_world - previous_world) / dt;
    let mut terrain_changed = false;

    match selected_tool {
        Some(WorldTool::WaterLiquid) => {
            interaction_state.last_granular_spawn_cell = None;
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                selected_tool,
                &mut particle_world,
                &mut object_world,
                &solver_params,
            );
            let mut spawn_cells = Vec::new();
            let mut last_cell = interaction_state.last_water_spawn_cell;
            stroke_cells(previous_world, cursor_world, TOOL_STROKE_STEP_M, |cell| {
                if last_cell != Some(cell) {
                    spawn_cells.push(cell);
                    last_cell = Some(cell);
                }
                particle_world.wake_particles_in_radius(
                    cell_to_world_center(cell),
                    solver_params.wake_radius_m,
                );
            });
            interaction_state.last_water_spawn_cell = last_cell;
            if !spawn_cells.is_empty() {
                let _ = particle_world.spawn_material_particles_from_cells(
                    &spawn_cells,
                    ParticleMaterial::WaterLiquid,
                    Vec2::ZERO,
                );
            }
        }
        Some(tool) if tool.terrain_material().is_some() => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.last_granular_spawn_cell = None;
            let terrain_material = tool.terrain_material().unwrap_or(TerrainMaterial::Stone);
            interaction_state.stone_stroke.active = false;
            interaction_state.stone_stroke.generated_cells.clear();
            interaction_state.stone_stroke.candidate_cells.clear();
            stroke_cells(previous_world, cursor_world, TOOL_STROKE_STEP_M, |cell| {
                let chunk = IVec2::new(
                    cell.x.div_euclid(CHUNK_SIZE_I32),
                    cell.y.div_euclid(CHUNK_SIZE_I32),
                );
                if !ensure_chunk_ready_for_edit(
                    &mut terrain_world,
                    &mut generated_chunk_cache,
                    chunk,
                ) {
                    return;
                }
                terrain_changed |=
                    terrain_world.set_cell(cell, TerrainCell::solid(terrain_material));
                particle_world.wake_particles_in_radius(
                    cell_to_world_center(cell),
                    solver_params.wake_radius_m,
                );
            });
        }
        Some(tool) if tool.is_granular() => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.stone_stroke.active = false;
            interaction_state.stone_stroke.generated_cells.clear();
            interaction_state.stone_stroke.candidate_cells.clear();
            if let Some(material) = tool.material() {
                let mut spawn_cells = Vec::new();
                let mut last_cell = interaction_state.last_granular_spawn_cell;
                stroke_cells(previous_world, cursor_world, TOOL_STROKE_STEP_M, |cell| {
                    if last_cell != Some(cell) {
                        spawn_cells.push(cell);
                        last_cell = Some(cell);
                    }
                    particle_world.wake_particles_in_radius(
                        cell_to_world_center(cell),
                        solver_params.wake_radius_m,
                    );
                });
                interaction_state.last_granular_spawn_cell = last_cell;
                if !spawn_cells.is_empty() {
                    let _ = particle_world.spawn_material_particles_from_cells(
                        &spawn_cells,
                        material,
                        Vec2::ZERO,
                    );
                }
            }
        }
        Some(WorldTool::Delete) => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.last_granular_spawn_cell = None;
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                selected_tool,
                &mut particle_world,
                &mut object_world,
                &solver_params,
            );
            let mut had_particle_removal = false;
            let mut removed_terrain_cells = HashSet::new();
            stroke_cells(previous_world, cursor_world, TOOL_STROKE_STEP_M, |cell| {
                let point = cell_to_world_center(cell);
                let removal = particle_world
                    .remove_particles_in_radius_with_map(point, TOOL_DELETE_BRUSH_RADIUS_M);
                if removal.removed_count > 0 {
                    had_particle_removal = true;
                    object_world.apply_particle_remap(&removal.old_to_new, particle_world.masses());
                }
                let chunk = IVec2::new(
                    cell.x.div_euclid(CHUNK_SIZE_I32),
                    cell.y.div_euclid(CHUNK_SIZE_I32),
                );
                if !ensure_chunk_ready_for_edit(
                    &mut terrain_world,
                    &mut generated_chunk_cache,
                    chunk,
                ) {
                    return;
                }
                if terrain_world.set_cell(cell, TerrainCell::Empty) {
                    removed_terrain_cells.insert(cell);
                }
                particle_world.wake_particles_in_radius(point, solver_params.wake_radius_m);
            });
            terrain_changed |= !removed_terrain_cells.is_empty();
            if had_particle_removal {
                particle_world.postprocess_objects_after_topology_edit(&mut object_world);
            }
        }
        Some(WorldTool::Break) => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.last_granular_spawn_cell = None;
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                selected_tool,
                &mut particle_world,
                &mut object_world,
                &solver_params,
            );
            let mut detached_particles = HashSet::new();
            let mut removed_terrain_cells = HashSet::new();
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                let fractured = particle_world
                    .fracture_solid_particles_in_radius(point, TOOL_BREAK_BRUSH_RADIUS_M);
                detached_particles.extend(fractured);
                removed_terrain_cells.extend(break_terrain_solids_in_radius(
                    &mut terrain_world,
                    &mut generated_chunk_cache,
                    &mut particle_world,
                    point,
                    TOOL_BREAK_BRUSH_RADIUS_M,
                ));
                particle_world.wake_particles_in_radius(point, solver_params.wake_radius_m);
            });
            terrain_changed |= !removed_terrain_cells.is_empty();
            if !detached_particles.is_empty() {
                particle_world
                    .detach_and_postprocess_objects(&mut object_world, &detached_particles);
            }
            if !removed_terrain_cells.is_empty() {
                terrain_changed |= particle_world.detach_terrain_components_after_cell_removal(
                    &mut terrain_world,
                    &mut object_world,
                    &removed_terrain_cells,
                );
            }
        }
        Some(_) => {}
        None => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.last_granular_spawn_cell = None;
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                selected_tool,
                &mut particle_world,
                &mut object_world,
                &solver_params,
            );
            let velocity_delta = stroke_velocity * DRAG_VELOCITY_GAIN;
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                particle_world.add_velocity_in_radius(
                    point,
                    DRAG_VELOCITY_BRUSH_RADIUS_M,
                    velocity_delta,
                    material_params.particle_speed_limit_mps,
                );
            });
        }
    }

    if terrain_changed {
        terrain_world
            .rebuild_static_particles_if_dirty(terrain_boundary_radius_m(*material_params));
    }

    interaction_state.last_drag_world = Some(cursor_world);
}

pub(super) fn draw_world_tool_hover_highlight(
    mut gizmos: Gizmos,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    interaction_state: Res<WorldInteractionState>,
    save_load_ui_state: Res<SaveLoadUiState>,
) {
    if save_load_ui_state.mode.is_some() {
        return;
    }
    let Some(tool) = interaction_state.selected_tool else {
        return;
    };
    if !tool.uses_cell_hover_highlight() {
        return;
    }
    let Some(cursor_world) = cursor_world_position(&windows, &camera_query) else {
        return;
    };

    let cell = world_to_cell(cursor_world);
    let center = cell_to_world_center(cell);
    let half = CELL_SIZE_M * 0.5;
    let min = center - Vec2::splat(half);
    let max = center + Vec2::splat(half);
    let color = match tool {
        WorldTool::Delete => TOOL_HOVER_HIGHLIGHT_DELETE_COLOR,
        WorldTool::Break => TOOL_HOVER_HIGHLIGHT_BREAK_COLOR,
        _ => TOOL_HOVER_HIGHLIGHT_TERRAIN_COLOR,
    };
    gizmos.line_2d(Vec2::new(min.x, min.y), Vec2::new(max.x, min.y), color);
    gizmos.line_2d(Vec2::new(max.x, min.y), Vec2::new(max.x, max.y), color);
    gizmos.line_2d(Vec2::new(max.x, max.y), Vec2::new(min.x, max.y), color);
    gizmos.line_2d(Vec2::new(min.x, max.y), Vec2::new(min.x, min.y), color);
}

fn finalize_stone_stroke(
    stroke: &mut StoneStrokeState,
    selected_tool: Option<WorldTool>,
    particle_world: &mut ParticleWorld,
    object_world: &mut ObjectWorld,
    solver_params: &SolverParams,
) {
    if !stroke.active {
        stroke.generated_cells.clear();
        stroke.candidate_cells.clear();
        return;
    }

    if !stroke.candidate_cells.is_empty() {
        let mut cells: Vec<_> = stroke.candidate_cells.iter().copied().collect();
        cells.sort_by_key(|cell| (cell.y, cell.x));
        if let Some(tool) = selected_tool {
            let Some(material) = tool.material() else {
                stroke.active = false;
                stroke.generated_cells.clear();
                stroke.candidate_cells.clear();
                return;
            };
            let particle_indices =
                particle_world.spawn_material_particles_from_cells(&cells, material, Vec2::ZERO);

            if tool.terrain_material().is_some() {
                let _ = object_world.create_object(
                    particle_indices,
                    particle_world.positions(),
                    particle_world.masses(),
                    solver_params.object_shape_stiffness_alpha,
                    solver_params.object_shape_iters,
                );
                particle_world.postprocess_objects_after_topology_edit(object_world);
            }
        }
    }

    stroke.active = false;
    stroke.generated_cells.clear();
    stroke.candidate_cells.clear();
}
fn cursor_world_position(
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) -> Option<Vec2> {
    let window = windows.iter().next()?;
    let cursor = window.cursor_position()?;
    let (camera, camera_transform) = cameras.iter().next()?;
    camera.viewport_to_world_2d(camera_transform, cursor).ok()
}

fn stroke_points(from: Vec2, to: Vec2, spacing: f32, mut paint: impl FnMut(Vec2)) {
    if spacing <= 0.0 {
        paint(to);
        return;
    }

    let delta = to - from;
    let distance = delta.length();
    if distance <= spacing {
        paint(to);
        return;
    }

    let steps = (distance / spacing).ceil() as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        paint(from.lerp(to, t));
    }
}

fn stroke_cells(from: Vec2, to: Vec2, spacing: f32, mut paint_cell: impl FnMut(IVec2)) {
    let mut previous_cell = None;
    stroke_points(from, to, spacing, |point| {
        let current_cell = world_to_cell(point);
        if let Some(prev_cell) = previous_cell {
            append_4_connected_segment(prev_cell, current_cell, |cell| paint_cell(cell));
        } else {
            paint_cell(current_cell);
        }
        previous_cell = Some(current_cell);
    });
}

fn append_4_connected_segment(from: IVec2, to: IVec2, mut emit: impl FnMut(IVec2)) {
    let mut current = from;
    emit(current);
    while current != to {
        let dx = to.x - current.x;
        let dy = to.y - current.y;
        if dx.abs() >= dy.abs() {
            if dx != 0 {
                current.x += dx.signum();
                emit(current);
            }
            if dy != 0 {
                current.y += dy.signum();
                emit(current);
            }
        } else {
            if dy != 0 {
                current.y += dy.signum();
                emit(current);
            }
            if dx != 0 {
                current.x += dx.signum();
                emit(current);
            }
        }
    }
}

fn tool_uses_generated_chunk_data(tool: WorldTool) -> bool {
    tool.terrain_material().is_some() || matches!(tool, WorldTool::Delete | WorldTool::Break)
}

fn ensure_chunk_ready_for_edit(
    terrain_world: &mut TerrainWorld,
    generated_chunk_cache: &mut TerrainGeneratedChunkCache,
    chunk_coord: IVec2,
) -> bool {
    if !terrain_world.generation_enabled() {
        terrain_world.ensure_chunk_loaded(chunk_coord);
        return true;
    }
    if let Some(material_ids) = generated_chunk_cache.material_ids_for_chunk(chunk_coord) {
        terrain_world.load_generated_chunk_from_material_ids(chunk_coord, material_ids);
        return true;
    }
    generated_chunk_cache.enqueue_chunk_request(chunk_coord);
    false
}

fn break_terrain_solids_in_radius(
    terrain_world: &mut TerrainWorld,
    generated_chunk_cache: &mut TerrainGeneratedChunkCache,
    particle_world: &mut ParticleWorld,
    center: Vec2,
    radius: f32,
) -> HashSet<IVec2> {
    let min_cell = world_to_cell(center - Vec2::splat(radius));
    let max_cell = world_to_cell(center + Vec2::splat(radius));
    let radius2 = radius * radius;
    let mut removed = HashSet::new();
    let mut spawn_cells_by_material = HashMap::<ParticleMaterial, Vec<IVec2>>::new();
    let mut ready_chunks = HashSet::new();
    let mut missing_chunks = HashSet::new();

    for y in min_cell.y..=max_cell.y {
        for x in min_cell.x..=max_cell.x {
            let cell = IVec2::new(x, y);
            let chunk = IVec2::new(
                cell.x.div_euclid(CHUNK_SIZE_I32),
                cell.y.div_euclid(CHUNK_SIZE_I32),
            );
            if !ready_chunks.contains(&chunk) {
                if missing_chunks.contains(&chunk) {
                    continue;
                }
                if ensure_chunk_ready_for_edit(terrain_world, generated_chunk_cache, chunk) {
                    ready_chunks.insert(chunk);
                } else {
                    missing_chunks.insert(chunk);
                    continue;
                }
            }
            if cell_to_world_center(cell).distance_squared(center) > radius2 {
                continue;
            }
            let TerrainCell::Solid { material, .. } = terrain_world.get_loaded_cell_or_empty(cell)
            else {
                continue;
            };
            let Some(target_particle) = terrain_fracture_particle(material) else {
                continue;
            };
            if terrain_world.set_cell(cell, TerrainCell::Empty) {
                removed.insert(cell);
                spawn_cells_by_material
                    .entry(target_particle)
                    .or_default()
                    .push(cell);
            }
        }
    }

    for (material, cells) in spawn_cells_by_material {
        let _ = particle_world.spawn_material_particles_from_cells(&cells, material, Vec2::ZERO);
    }
    removed
}
