use std::collections::{HashSet, VecDeque};

use bevy::asset::RenderAssetUsages;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::PrimaryWindow;

use crate::physics::cell_to_world_center;
use crate::physics::object::{OBJECT_SHAPE_ITERS, OBJECT_SHAPE_STIFFNESS_ALPHA, ObjectWorld};
use crate::physics::particle::{
    PARTICLE_SPACING_M, PARTICLE_SPEED_LIMIT_MPS, ParticleMaterial, ParticleWorld,
    TERRAIN_BOUNDARY_RADIUS_M,
};
use crate::physics::state::{SimUpdateSet, SimulationPerfMetrics, SimulationState};
use crate::physics::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, TerrainCell, TerrainWorld, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y,
    WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y, world_to_cell,
};

const HUD_BG_COLOR: Color = Color::srgba(0.05, 0.06, 0.09, 0.82);
const BUTTON_BG_OFF: Color = Color::srgba(0.17, 0.18, 0.22, 0.95);
const BUTTON_BG_ON: Color = Color::srgba(0.16, 0.30, 0.46, 0.95);
const BUTTON_BG_HOVER: Color = Color::srgba(0.24, 0.25, 0.30, 0.98);
const BUTTON_BG_PRESS: Color = Color::srgba(0.38, 0.40, 0.48, 0.98);
const BUTTON_BORDER_OFF: Color = Color::srgba(0.08, 0.10, 0.14, 1.0);
const BUTTON_BORDER_ON: Color = Color::srgba(0.80, 0.92, 1.00, 1.0);
const TOOLBAR_BOTTOM_PX: f32 = 12.0;
const TOOLBAR_BG_COLOR: Color = Color::srgba(0.05, 0.06, 0.09, 0.88);
const TOOLBAR_ICON_SIZE_PX: u32 = 32;
const TOOLBAR_ICON_GRID_SIZE: usize = 8;
const TOOLBAR_ICON_DOT_PX: u32 = TOOLBAR_ICON_SIZE_PX / TOOLBAR_ICON_GRID_SIZE as u32;
const TOOLBAR_BUTTON_SIZE_PX: f32 = 44.0;
const TOOLTIP_BG_COLOR: Color = Color::srgba(0.04, 0.05, 0.08, 0.96);
const TOOLTIP_CURSOR_OFFSET_X: f32 = 14.0;
const TOOLTIP_CURSOR_OFFSET_Y: f32 = 20.0;
const TOOLTIP_GLOBAL_Z_INDEX: i32 = 10_000;
const DRAG_VELOCITY_BRUSH_RADIUS_M: f32 = 0.55;
const DRAG_VELOCITY_GAIN: f32 = 0.9;
const TOOL_STROKE_STEP_M: f32 = CELL_SIZE_M * 0.5;
const TOOL_WATER_SPAWN_SPACING_M: f32 = PARTICLE_SPACING_M;
const TOOL_DELETE_BRUSH_RADIUS_M: f32 = CELL_SIZE_M * 0.5;
const STONE_STROKE_NEIGHBOR_OFFSETS: [IVec2; 4] = [
    IVec2::new(1, 0),
    IVec2::new(-1, 0),
    IVec2::new(0, 1),
    IVec2::new(0, -1),
];
type MaterialPattern8 = [[bool; TOOLBAR_ICON_GRID_SIZE]; TOOLBAR_ICON_GRID_SIZE];

const MATERIAL_PATTERN_LIQUID: MaterialPattern8 = [
    [false, false, false, false, false, false, false, false],
    [false, true, true, false, false, false, false, false],
    [true, true, true, true, true, false, false, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
];

const MATERIAL_PATTERN_SOLID: MaterialPattern8 = [
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
];

#[allow(dead_code)]
const MATERIAL_PATTERN_POWDER: MaterialPattern8 = [
    [false, true, false, true, false, true, false, true],
    [true, false, true, false, true, false, true, false],
    [false, true, false, true, false, true, false, true],
    [true, false, true, false, true, false, true, false],
    [false, true, false, true, false, true, false, true],
    [true, false, true, false, true, false, true, false],
    [false, true, false, true, false, true, false, true],
    [true, false, true, false, true, false, true, false],
];

const MATERIAL_PALETTE_LIQUID: [[u8; 4]; 4] = [
    [42, 120, 202, 235],
    [52, 136, 218, 240],
    [65, 152, 228, 245],
    [78, 167, 238, 250],
];

const MATERIAL_PALETTE_SOLID: [[u8; 4]; 4] = [
    [70, 67, 63, 255],
    [83, 79, 74, 255],
    [95, 90, 84, 255],
    [108, 103, 96, 255],
];

#[allow(dead_code)]
const MATERIAL_PALETTE_POWDER: [[u8; 4]; 4] = [
    [172, 149, 111, 255],
    [185, 162, 124, 255],
    [198, 175, 136, 255],
    [210, 188, 148, 255],
];

pub struct InterfacePlugin;

impl Plugin for InterfacePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldInteractionState>()
            .add_systems(Startup, setup_simulation_ui)
            .add_systems(
                Update,
                (
                    handle_world_tool_button_interaction,
                    handle_world_interactions,
                )
                    .chain()
                    .in_set(SimUpdateSet::Interaction),
            )
            .add_systems(
                Update,
                (
                    update_world_tool_button_visuals,
                    update_world_tool_tooltip,
                    update_simulation_hud,
                )
                    .chain()
                    .in_set(SimUpdateSet::Ui),
            );
    }
}

#[derive(Component)]
struct SimulationHudText;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WorldTool {
    Water,
    Stone,
    Delete,
}

impl WorldTool {
    fn label(self) -> &'static str {
        match self {
            Self::Water => "Water",
            Self::Stone => "Stone",
            Self::Delete => "Delete",
        }
    }
}

#[derive(Resource, Debug, Default)]
struct WorldInteractionState {
    selected_tool: Option<WorldTool>,
    last_drag_world: Option<Vec2>,
    water_spawn_carry_m: f32,
    stone_stroke: StoneStrokeState,
}

#[derive(Debug, Default)]
struct StoneStrokeState {
    active: bool,
    generated_cells: HashSet<IVec2>,
    candidate_cells: HashSet<IVec2>,
}

#[derive(Component, Clone, Copy)]
struct WorldToolButton {
    tool: WorldTool,
}

#[derive(Component)]
struct WorldToolTooltip;

#[derive(Component)]
struct WorldToolTooltipText;

#[derive(Resource, Clone)]
struct WorldToolIconSet {
    water: Handle<Image>,
    stone: Handle<Image>,
    delete: Handle<Image>,
}

fn setup_simulation_ui(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let icon_set = create_world_tool_icon_set(&mut images);
    commands.insert_resource(icon_set.clone());

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(10.0),
                top: px(10.0),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(HUD_BG_COLOR),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("FPS: --\nPotential Max FPS: --"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                SimulationHudText,
            ));
        });

    commands
        .spawn((Node {
            position_type: PositionType::Absolute,
            left: px(0.0),
            right: px(0.0),
            bottom: px(TOOLBAR_BOTTOM_PX),
            justify_content: JustifyContent::Center,
            ..default()
        },))
        .with_children(|parent| {
            parent
                .spawn((
                    Node {
                        padding: UiRect::axes(px(10.0), px(8.0)),
                        column_gap: px(8.0),
                        ..default()
                    },
                    BackgroundColor(TOOLBAR_BG_COLOR),
                ))
                .with_children(|toolbar| {
                    for tool in [WorldTool::Water, WorldTool::Stone, WorldTool::Delete] {
                        toolbar
                            .spawn((
                                Button,
                                Node {
                                    width: px(TOOLBAR_BUTTON_SIZE_PX),
                                    height: px(TOOLBAR_BUTTON_SIZE_PX),
                                    align_items: AlignItems::Center,
                                    justify_content: JustifyContent::Center,
                                    border: UiRect::all(px(2.0)),
                                    ..default()
                                },
                                BackgroundColor(BUTTON_BG_OFF),
                                BorderColor::all(BUTTON_BORDER_OFF),
                                WorldToolButton { tool },
                            ))
                            .with_children(|button| {
                                button.spawn((
                                    ImageNode::new(icon_set.icon_for(tool)),
                                    Node {
                                        width: px(TOOLBAR_ICON_SIZE_PX as f32),
                                        height: px(TOOLBAR_ICON_SIZE_PX as f32),
                                        ..default()
                                    },
                                ));
                            });
                    }
                });
        });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                display: Display::None,
                left: px(0.0),
                top: px(0.0),
                padding: UiRect::axes(px(8.0), px(4.0)),
                ..default()
            },
            BackgroundColor(TOOLTIP_BG_COLOR),
            GlobalZIndex(TOOLTIP_GLOBAL_Z_INDEX),
            WorldToolTooltip,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(13.0),
                TextColor(Color::WHITE),
                WorldToolTooltipText,
            ));
        });
}

fn handle_world_tool_button_interaction(
    mut interactions: Query<
        (&Interaction, &WorldToolButton),
        (Changed<Interaction>, With<WorldToolButton>),
    >,
    mut interaction_state: ResMut<WorldInteractionState>,
) {
    for (interaction, button) in &mut interactions {
        if *interaction == Interaction::Pressed {
            select_world_tool(&mut interaction_state, Some(button.tool));
        }
    }
}

fn update_world_tool_button_visuals(
    interaction_state: Res<WorldInteractionState>,
    mut buttons: Query<(
        &Interaction,
        &WorldToolButton,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (interaction, button, mut bg, mut border_color) in &mut buttons {
        let selected = interaction_state.selected_tool == Some(button.tool);
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => toggle_button_bg(selected),
        };
        *border_color = if selected {
            BorderColor::all(BUTTON_BORDER_ON)
        } else {
            BorderColor::all(BUTTON_BORDER_OFF)
        };
    }
}

fn update_world_tool_tooltip(
    windows: Query<&Window, With<PrimaryWindow>>,
    buttons: Query<(&Interaction, &WorldToolButton), With<Button>>,
    mut tooltip: Single<&mut Node, With<WorldToolTooltip>>,
    mut tooltip_text: Single<&mut Text, With<WorldToolTooltipText>>,
) {
    let hovered_tool = buttons.iter().find_map(|(interaction, button)| {
        (*interaction == Interaction::Hovered).then_some(button.tool)
    });
    let Some(tool) = hovered_tool else {
        tooltip.display = Display::None;
        return;
    };

    let Some(window) = windows.iter().next() else {
        tooltip.display = Display::None;
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        tooltip.display = Display::None;
        return;
    };

    tooltip.display = Display::Flex;
    tooltip.left = px(cursor.x + TOOLTIP_CURSOR_OFFSET_X);
    tooltip.top = px(cursor.y + TOOLTIP_CURSOR_OFFSET_Y);
    tooltip_text.0 = tool.label().to_string();
}

fn handle_world_interactions(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    ui_buttons: Query<&Interaction, With<Button>>,
    mut interaction_state: ResMut<WorldInteractionState>,
    mut particle_world: ResMut<ParticleWorld>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut object_world: ResMut<ObjectWorld>,
) {
    if keyboard.just_pressed(KeyCode::Digit1) || keyboard.just_pressed(KeyCode::Numpad1) {
        select_world_tool(&mut interaction_state, Some(WorldTool::Water));
    } else if keyboard.just_pressed(KeyCode::Digit2) || keyboard.just_pressed(KeyCode::Numpad2) {
        select_world_tool(&mut interaction_state, Some(WorldTool::Stone));
    } else if keyboard.just_pressed(KeyCode::Digit3) || keyboard.just_pressed(KeyCode::Numpad3) {
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

    if !left_pressed || blocked_by_pan || blocked_by_ui || cursor_world.is_none() {
        finalize_stone_stroke(
            &mut interaction_state.stone_stroke,
            &mut particle_world,
            &mut object_world,
        );
        interaction_state.last_drag_world = None;
        interaction_state.water_spawn_carry_m = 0.0;
        return;
    }

    let cursor_world = cursor_world.unwrap_or(Vec2::ZERO);

    let previous_world = interaction_state.last_drag_world.unwrap_or(cursor_world);
    let dt = time.delta_secs().max(1e-4);
    let stroke_velocity = (cursor_world - previous_world) / dt;
    let mut terrain_changed = false;
    if interaction_state.selected_tool != Some(WorldTool::Water) {
        interaction_state.water_spawn_carry_m = 0.0;
    }

    match interaction_state.selected_tool {
        Some(WorldTool::Water) => {
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                &mut particle_world,
                &mut object_world,
            );
            particle_world.spawn_water_particles_along_segment(
                previous_world,
                cursor_world,
                TOOL_WATER_SPAWN_SPACING_M,
                Vec2::ZERO,
                &mut interaction_state.water_spawn_carry_m,
            );
        }
        Some(WorldTool::Stone) => {
            interaction_state.stone_stroke.active = true;
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                let cell = world_to_cell(point);
                if !cell_in_fixed_world(cell) {
                    return;
                }
                interaction_state.stone_stroke.generated_cells.insert(cell);
            });
            terrain_changed |= update_stone_stroke_partition(
                &mut interaction_state.stone_stroke,
                &mut terrain_world,
            );
        }
        Some(WorldTool::Delete) => {
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                &mut particle_world,
                &mut object_world,
            );
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                let removal = particle_world
                    .remove_particles_in_radius_with_map(point, TOOL_DELETE_BRUSH_RADIUS_M);
                if removal.removed_count > 0 {
                    object_world.apply_particle_remap(&removal.old_to_new, particle_world.masses());
                }
                terrain_changed |= paint_terrain_cells_in_radius(
                    &mut terrain_world,
                    point,
                    TOOL_DELETE_BRUSH_RADIUS_M,
                    TerrainCell::Empty,
                );
            });
        }
        None => {
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                &mut particle_world,
                &mut object_world,
            );
            interaction_state.water_spawn_carry_m = 0.0;
            let velocity_delta = stroke_velocity * DRAG_VELOCITY_GAIN;
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                particle_world.add_velocity_in_radius(
                    point,
                    DRAG_VELOCITY_BRUSH_RADIUS_M,
                    velocity_delta,
                    PARTICLE_SPEED_LIMIT_MPS,
                );
            });
        }
    }

    if terrain_changed {
        terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
    }

    interaction_state.last_drag_world = Some(cursor_world);
}

fn update_stone_stroke_partition(
    stroke: &mut StoneStrokeState,
    terrain_world: &mut TerrainWorld,
) -> bool {
    if stroke.generated_cells.is_empty() {
        stroke.candidate_cells.clear();
        return false;
    }

    let terrain_connected = collect_terrain_connected_stroke_cells(stroke, terrain_world);
    let mut terrain_changed = false;
    for &cell in &terrain_connected {
        terrain_changed |= terrain_world.set_cell(cell, TerrainCell::rock());
    }

    stroke.candidate_cells.clear();
    for &cell in &stroke.generated_cells {
        if !terrain_connected.contains(&cell) {
            stroke.candidate_cells.insert(cell);
        }
    }
    terrain_changed
}

fn collect_terrain_connected_stroke_cells(
    stroke: &StoneStrokeState,
    terrain_world: &TerrainWorld,
) -> HashSet<IVec2> {
    let mut connected = HashSet::new();
    let mut queue = VecDeque::new();

    for &cell in &stroke.generated_cells {
        if is_stroke_cell_connected_to_frozen_terrain(cell, stroke, terrain_world) {
            connected.insert(cell);
            queue.push_back(cell);
        }
    }

    while let Some(cell) = queue.pop_front() {
        for offset in STONE_STROKE_NEIGHBOR_OFFSETS {
            let next = cell + offset;
            if !stroke.generated_cells.contains(&next) || connected.contains(&next) {
                continue;
            }
            connected.insert(next);
            queue.push_back(next);
        }
    }

    connected
}

fn is_stroke_cell_connected_to_frozen_terrain(
    cell: IVec2,
    stroke: &StoneStrokeState,
    terrain_world: &TerrainWorld,
) -> bool {
    if matches!(
        terrain_world.get_loaded_cell_or_empty(cell),
        TerrainCell::Solid { .. }
    ) {
        return true;
    }

    for offset in STONE_STROKE_NEIGHBOR_OFFSETS {
        let neighbor = cell + offset;
        if stroke.generated_cells.contains(&neighbor) {
            continue;
        }
        if matches!(
            terrain_world.get_loaded_cell_or_empty(neighbor),
            TerrainCell::Solid { .. }
        ) {
            return true;
        }
    }
    false
}

fn finalize_stone_stroke(
    stroke: &mut StoneStrokeState,
    particle_world: &mut ParticleWorld,
    object_world: &mut ObjectWorld,
) {
    if !stroke.active {
        stroke.generated_cells.clear();
        stroke.candidate_cells.clear();
        return;
    }

    if !stroke.candidate_cells.is_empty() {
        let mut cells: Vec<_> = stroke.candidate_cells.iter().copied().collect();
        cells.sort_by_key(|cell| (cell.y, cell.x));
        let particle_indices = particle_world.spawn_stone_particles_from_cells(&cells, Vec2::ZERO);
        let _ = object_world.create_object(
            particle_indices,
            particle_world.positions(),
            particle_world.masses(),
            OBJECT_SHAPE_STIFFNESS_ALPHA,
            OBJECT_SHAPE_ITERS,
        );
    }

    stroke.active = false;
    stroke.generated_cells.clear();
    stroke.candidate_cells.clear();
}

fn update_simulation_hud(
    diagnostics: Res<DiagnosticsStore>,
    sim_state: Res<SimulationState>,
    particles: Res<ParticleWorld>,
    mut perf_metrics: ResMut<SimulationPerfMetrics>,
    mut texts: Query<&mut Text, With<SimulationHudText>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|diag| diag.smoothed())
        .unwrap_or(0.0);

    let potential = if perf_metrics.physics_time_this_frame_secs > 0.0 {
        1.0 / perf_metrics.physics_time_this_frame_secs
    } else {
        f64::INFINITY
    };
    perf_metrics.potential_max_fps = potential;

    for mut text in &mut texts {
        let sim_status = if sim_state.running {
            "Running"
        } else {
            "Paused"
        };
        let water_count = particles
            .materials()
            .iter()
            .filter(|&&m| matches!(m, ParticleMaterial::Water))
            .count();
        let stone_count = particles
            .materials()
            .iter()
            .filter(|&&m| matches!(m, ParticleMaterial::Stone))
            .count();
        let potential_str = if potential.is_finite() {
            format!("{potential:.1}")
        } else {
            "INF".to_string()
        };
        text.0 = format!(
            "FPS: {fps:.1}\nPotential Max FPS: {potential_str}\nSim: {sim_status}\nWater Particles: {water_count}\nStone Particles: {stone_count}"
        );
    }
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

fn paint_terrain_cells_in_radius(
    terrain_world: &mut TerrainWorld,
    center: Vec2,
    radius: f32,
    cell_value: TerrainCell,
) -> bool {
    let min_cell = world_to_cell(center - Vec2::splat(radius));
    let max_cell = world_to_cell(center + Vec2::splat(radius));
    let radius2 = radius * radius;
    let mut changed = false;

    for y in min_cell.y..=max_cell.y {
        for x in min_cell.x..=max_cell.x {
            let cell = IVec2::new(x, y);
            if !cell_in_fixed_world(cell) {
                continue;
            }
            if cell_to_world_center(cell).distance_squared(center) > radius2 {
                continue;
            }
            changed |= terrain_world.set_cell(cell, cell_value);
        }
    }

    changed
}

fn cell_in_fixed_world(cell: IVec2) -> bool {
    let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32;
    let max_cell_x = (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1;
    let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32;
    let max_cell_y = (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1;

    (min_cell_x..=max_cell_x).contains(&cell.x) && (min_cell_y..=max_cell_y).contains(&cell.y)
}

fn toggle_button_bg(enabled: bool) -> BackgroundColor {
    if enabled {
        BUTTON_BG_ON.into()
    } else {
        BUTTON_BG_OFF.into()
    }
}

fn select_world_tool(state: &mut WorldInteractionState, next_tool: Option<WorldTool>) {
    state.selected_tool = next_tool;
    state.last_drag_world = None;
    state.water_spawn_carry_m = 0.0;
    state.stone_stroke = StoneStrokeState::default();
}

impl WorldToolIconSet {
    fn icon_for(&self, tool: WorldTool) -> Handle<Image> {
        match tool {
            WorldTool::Water => self.water.clone(),
            WorldTool::Stone => self.stone.clone(),
            WorldTool::Delete => self.delete.clone(),
        }
    }
}

fn create_world_tool_icon_set(images: &mut Assets<Image>) -> WorldToolIconSet {
    let water = images.add(build_material_icon_image(
        MATERIAL_PALETTE_LIQUID,
        &MATERIAL_PATTERN_LIQUID,
        0x4e23_1f91,
    ));
    let stone = images.add(build_material_icon_image(
        MATERIAL_PALETTE_SOLID,
        &MATERIAL_PATTERN_SOLID,
        0x8a52_d9b7,
    ));
    let delete = images.add(build_delete_icon_image());
    WorldToolIconSet {
        water,
        stone,
        delete,
    }
}

fn build_material_icon_image(
    palette: [[u8; 4]; 4],
    pattern: &MaterialPattern8,
    seed: u32,
) -> Image {
    let width = TOOLBAR_ICON_SIZE_PX;
    let height = TOOLBAR_ICON_SIZE_PX;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for (gy, row) in pattern.iter().enumerate() {
        for (gx, enabled) in row.iter().copied().enumerate() {
            if !enabled {
                continue;
            }
            let palette_index = material_icon_palette_index(gx as u32, gy as u32, seed);
            let color = palette[palette_index];
            let start_x = gx as u32 * TOOLBAR_ICON_DOT_PX;
            let start_y = gy as u32 * TOOLBAR_ICON_DOT_PX;
            for py in start_y..(start_y + TOOLBAR_ICON_DOT_PX) {
                for px in start_x..(start_x + TOOLBAR_ICON_DOT_PX) {
                    if px >= width || py >= height {
                        continue;
                    }
                    let idx = ((py * width + px) * 4) as usize;
                    pixels[idx..idx + 4].copy_from_slice(&color);
                }
            }
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    image
}

fn material_icon_palette_index(x: u32, y: u32, seed: u32) -> usize {
    let mut state = x.wrapping_mul(0x45d9f3b);
    state ^= y.wrapping_mul(0x27d4eb2d);
    state ^= seed;
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    (state & 0b11) as usize
}

fn build_delete_icon_image() -> Image {
    let width = TOOLBAR_ICON_SIZE_PX;
    let height = TOOLBAR_ICON_SIZE_PX;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let red = [220u8, 52u8, 52u8, 255u8];
    let thickness = 3i32;
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let d1 = (x - y).abs();
            let d2 = ((width as i32 - 1 - x) - y).abs();
            if d1 <= thickness || d2 <= thickness {
                let idx = (((y as u32) * width + (x as u32)) * 4) as usize;
                pixels[idx..idx + 4].copy_from_slice(&red);
            }
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    image
}
