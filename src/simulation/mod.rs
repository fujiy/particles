mod particle;
mod render;
mod terrain;

use std::time::Instant;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use particle::{
    PARTICLE_RADIUS_M, PARTICLE_SPACING_M, PARTICLE_SPEED_LIMIT_MPS, ParticleWorld,
    TERRAIN_BOUNDARY_RADIUS_M, WATER_KERNEL_RADIUS_M, nominal_particle_draw_radius_m,
};
use render::{
    TerrainRenderState, WaterRenderState, bootstrap_terrain_chunks,
    sync_dirty_terrain_chunks_to_render, sync_water_dots_to_render,
};
pub use terrain::cell_to_world_center;
use terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, TerrainCell, TerrainWorld, WORLD_MAX_CHUNK_X,
    WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y, world_to_cell,
};

const HUD_BG_COLOR: Color = Color::srgba(0.05, 0.06, 0.09, 0.82);
const BUTTON_BG_OFF: Color = Color::srgba(0.17, 0.18, 0.22, 0.95);
const BUTTON_BG_ON: Color = Color::srgba(0.16, 0.30, 0.46, 0.95);
const BUTTON_BG_HOVER: Color = Color::srgba(0.24, 0.25, 0.30, 0.98);
const BUTTON_BG_PRESS: Color = Color::srgba(0.38, 0.40, 0.48, 0.98);
const GRID_NEIGHBOR_COLOR: Color = Color::srgba(0.27, 0.75, 0.98, 0.28);
const GRID_CHUNK_COLOR: Color = Color::srgba(0.98, 0.78, 0.25, 0.75);
const GRID_BUTTON_BOTTOM_PX: f32 = 50.0;
const DEBUG_BUTTON_BOTTOM_PX: f32 = 12.0;
const TOOLBAR_BOTTOM_PX: f32 = 12.0;
const TERRAIN_PARTICLE_RADIUS_M: f32 = PARTICLE_RADIUS_M * 0.55;
const DEBUG_OVERLAY_CIRCLE_RESOLUTION: u32 = 8;
const TOOLBAR_BG_COLOR: Color = Color::srgba(0.05, 0.06, 0.09, 0.88);
const DRAG_VELOCITY_BRUSH_RADIUS_M: f32 = 0.55;
const DRAG_VELOCITY_GAIN: f32 = 0.9;
const TOOL_STROKE_STEP_M: f32 = CELL_SIZE_M * 0.5;
const TOOL_WATER_SPAWN_SPACING_M: f32 = PARTICLE_SPACING_M;
const TOOL_DELETE_BRUSH_RADIUS_M: f32 = CELL_SIZE_M * 0.5;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainWorld>()
            .init_resource::<ParticleWorld>()
            .init_resource::<SimulationState>()
            .init_resource::<DebugOverlayState>()
            .init_resource::<GridOverlayState>()
            .init_resource::<WorldInteractionState>()
            .init_resource::<SimulationPerfMetrics>()
            .init_resource::<TerrainRenderState>()
            .init_resource::<WaterRenderState>()
            .add_message::<ResetSimulationRequest>()
            .add_systems(
                Startup,
                (bootstrap_terrain_chunks, setup_simulation_ui).chain(),
            )
            .add_systems(FixedUpdate, step_water_particles)
            .add_systems(
                Update,
                (
                    handle_sim_controls,
                    apply_sim_reset,
                    handle_grid_overlay_button,
                    handle_overlay_button,
                    handle_world_tool_button_interaction,
                    handle_world_interactions,
                    sync_dirty_terrain_chunks_to_render,
                    sync_water_dots_to_render,
                    update_simulation_hud,
                    update_grid_overlay_button_label,
                    update_overlay_button_label,
                    update_world_tool_button_visuals,
                    draw_grid_overlay,
                    draw_particle_debug_overlay,
                    finalize_frame_metrics,
                )
                    .chain(),
            );
    }
}

#[derive(Resource, Debug, Default)]
pub struct SimulationState {
    running: bool,
}

#[derive(Resource, Debug)]
struct DebugOverlayState {
    enabled: bool,
}

impl Default for DebugOverlayState {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Resource, Debug)]
struct GridOverlayState {
    enabled: bool,
}

impl Default for GridOverlayState {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

#[derive(Resource, Debug, Default)]
struct SimulationPerfMetrics {
    physics_time_this_frame_secs: f64,
    potential_max_fps: f64,
}

#[derive(Message)]
struct ResetSimulationRequest;

#[derive(Component)]
struct SimulationHudText;

#[derive(Component)]
struct OverlayToggleButton;

#[derive(Component)]
struct OverlayToggleButtonLabel;

#[derive(Component)]
struct GridOverlayToggleButton;

#[derive(Component)]
struct GridOverlayToggleButtonLabel;

#[derive(Component, Clone, Copy)]
struct WorldToolButton {
    tool: WorldTool,
}

#[derive(Component, Clone, Copy)]
struct WorldToolButtonLabel {
    tool: WorldTool,
}

fn step_water_particles(
    sim_state: Res<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut perf_metrics: ResMut<SimulationPerfMetrics>,
) {
    let running = sim_state.running;
    if running {
        terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let start = Instant::now();
        particle_world.step_if_running(&terrain_world, running);
        perf_metrics.physics_time_this_frame_secs += start.elapsed().as_secs_f64();
    } else {
        particle_world.step_if_running(&terrain_world, running);
    }
}

fn handle_sim_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut sim_state: ResMut<SimulationState>,
    mut reset_writer: MessageWriter<ResetSimulationRequest>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.running = !sim_state.running;
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        reset_writer.write(ResetSimulationRequest);
    }
}

fn apply_sim_reset(
    mut reset_reader: MessageReader<ResetSimulationRequest>,
    mut sim_state: ResMut<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
) {
    if reset_reader.read().next().is_none() {
        return;
    }

    terrain_world.reset_fixed_world();
    terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
    particle_world.reset_to_initial();
    sim_state.running = false;
}

fn setup_simulation_ui(mut commands: Commands) {
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
                                    padding: UiRect::axes(px(10.0), px(6.0)),
                                    ..default()
                                },
                                BackgroundColor(BUTTON_BG_OFF),
                                WorldToolButton { tool },
                            ))
                            .with_children(|button| {
                                button.spawn((
                                    Text::new(tool.label()),
                                    TextFont::from_font_size(14.0),
                                    TextColor(Color::WHITE),
                                    WorldToolButtonLabel { tool },
                                ));
                            });
                    }
                });
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(12.0),
                bottom: px(GRID_BUTTON_BOTTOM_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(BUTTON_BG_OFF),
            GridOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Grid Overlay: OFF"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                GridOverlayToggleButtonLabel,
            ));
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(12.0),
                bottom: px(DEBUG_BUTTON_BOTTOM_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(BUTTON_BG_OFF),
            OverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Debug Overlay: OFF"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                OverlayToggleButtonLabel,
            ));
        });
}

fn handle_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<OverlayToggleButton>),
    >,
    mut overlay_state: ResMut<DebugOverlayState>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                overlay_state.enabled = !overlay_state.enabled;
                *bg = BUTTON_BG_PRESS.into();
            }
            Interaction::Hovered => {
                *bg = BUTTON_BG_HOVER.into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(overlay_state.enabled);
            }
        }
    }
}

fn handle_grid_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<GridOverlayToggleButton>),
    >,
    mut overlay_state: ResMut<GridOverlayState>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                overlay_state.enabled = !overlay_state.enabled;
                *bg = BUTTON_BG_PRESS.into();
            }
            Interaction::Hovered => {
                *bg = BUTTON_BG_HOVER.into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(overlay_state.enabled);
            }
        }
    }
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
            interaction_state.selected_tool = Some(button.tool);
            interaction_state.last_drag_world = None;
            interaction_state.water_spawn_carry_m = 0.0;
        }
    }
}

fn update_world_tool_button_visuals(
    interaction_state: Res<WorldInteractionState>,
    mut buttons: Query<(&Interaction, &WorldToolButton, &mut BackgroundColor)>,
    mut labels: Query<(&WorldToolButtonLabel, &mut Text)>,
) {
    for (interaction, button, mut bg) in &mut buttons {
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => {
                toggle_button_bg(interaction_state.selected_tool == Some(button.tool))
            }
        };
    }

    for (label, mut text) in &mut labels {
        if interaction_state.selected_tool == Some(label.tool) {
            text.0 = format!("{} [Selected]", label.tool.label());
        } else {
            text.0 = label.tool.label().to_string();
        }
    }
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
) {
    if keyboard.just_pressed(KeyCode::Escape) {
        interaction_state.selected_tool = None;
        interaction_state.last_drag_world = None;
        interaction_state.water_spawn_carry_m = 0.0;
    }

    if !mouse_buttons.pressed(MouseButton::Left) {
        interaction_state.last_drag_world = None;
        interaction_state.water_spawn_carry_m = 0.0;
        return;
    }

    let alt_pressed = keyboard.pressed(KeyCode::AltLeft) || keyboard.pressed(KeyCode::AltRight);
    if alt_pressed || mouse_buttons.pressed(MouseButton::Middle) {
        interaction_state.last_drag_world = None;
        interaction_state.water_spawn_carry_m = 0.0;
        return;
    }

    if ui_buttons
        .iter()
        .any(|interaction| *interaction != Interaction::None)
    {
        interaction_state.last_drag_world = None;
        interaction_state.water_spawn_carry_m = 0.0;
        return;
    }

    let Some(cursor_world) = cursor_world_position(&windows, &camera_query) else {
        interaction_state.last_drag_world = None;
        interaction_state.water_spawn_carry_m = 0.0;
        return;
    };

    let previous_world = interaction_state.last_drag_world.unwrap_or(cursor_world);
    let dt = time.delta_secs().max(1e-4);
    let stroke_velocity = (cursor_world - previous_world) / dt;
    let mut terrain_changed = false;
    if interaction_state.selected_tool != Some(WorldTool::Water) {
        interaction_state.water_spawn_carry_m = 0.0;
    }

    match interaction_state.selected_tool {
        Some(WorldTool::Water) => {
            particle_world.spawn_water_particles_along_segment(
                previous_world,
                cursor_world,
                TOOL_WATER_SPAWN_SPACING_M,
                Vec2::ZERO,
                &mut interaction_state.water_spawn_carry_m,
            );
        }
        Some(WorldTool::Stone) => {
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                terrain_changed |= paint_single_terrain_cell(
                    &mut terrain_world,
                    world_to_cell(point),
                    TerrainCell::rock(),
                );
            });
        }
        Some(WorldTool::Delete) => {
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                particle_world.remove_particles_in_radius(point, TOOL_DELETE_BRUSH_RADIUS_M);
                terrain_changed |= paint_terrain_cells_in_radius(
                    &mut terrain_world,
                    point,
                    TOOL_DELETE_BRUSH_RADIUS_M,
                    TerrainCell::Empty,
                );
            });
        }
        None => {
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

fn update_grid_overlay_button_label(
    overlay_state: Res<GridOverlayState>,
    mut labels: Query<&mut Text, With<GridOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() {
        return;
    }

    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            "Grid Overlay: ON".to_string()
        } else {
            "Grid Overlay: OFF".to_string()
        };
    }
}

fn update_overlay_button_label(
    overlay_state: Res<DebugOverlayState>,
    mut labels: Query<&mut Text, With<OverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() {
        return;
    }

    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            "Debug Overlay: ON".to_string()
        } else {
            "Debug Overlay: OFF".to_string()
        };
    }
}

fn draw_grid_overlay(mut gizmos: Gizmos, overlay_state: Res<GridOverlayState>) {
    if !overlay_state.enabled {
        return;
    }

    let min_x = WORLD_MIN_CHUNK_X as f32 * CHUNK_WORLD_SIZE_M;
    let max_x = (WORLD_MAX_CHUNK_X + 1) as f32 * CHUNK_WORLD_SIZE_M;
    let min_y = WORLD_MIN_CHUNK_Y as f32 * CHUNK_WORLD_SIZE_M;
    let max_y = (WORLD_MAX_CHUNK_Y + 1) as f32 * CHUNK_WORLD_SIZE_M;

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

    for chunk_x in WORLD_MIN_CHUNK_X..=(WORLD_MAX_CHUNK_X + 1) {
        let x = chunk_x as f32 * CHUNK_WORLD_SIZE_M;
        gizmos.line_2d(Vec2::new(x, min_y), Vec2::new(x, max_y), GRID_CHUNK_COLOR);
    }
    for chunk_y in WORLD_MIN_CHUNK_Y..=(WORLD_MAX_CHUNK_Y + 1) {
        let y = chunk_y as f32 * CHUNK_WORLD_SIZE_M;
        gizmos.line_2d(Vec2::new(min_x, y), Vec2::new(max_x, y), GRID_CHUNK_COLOR);
    }
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
        let potential_str = if potential.is_finite() {
            format!("{potential:.1}")
        } else {
            "INF".to_string()
        };
        text.0 = format!(
            "FPS: {fps:.1}\nPotential Max FPS: {potential_str}\nSim: {sim_status}\nWater Particles: {}",
            particles.particle_count()
        );
    }
}

fn draw_particle_debug_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<DebugOverlayState>,
    terrain_world: Res<TerrainWorld>,
    particle_world: Res<ParticleWorld>,
) {
    if !overlay_state.enabled {
        return;
    }

    let water_overlay_radius = nominal_particle_draw_radius_m();

    for pos in terrain_world.static_particle_positions() {
        gizmos
            .circle_2d(
                *pos,
                TERRAIN_PARTICLE_RADIUS_M,
                Color::srgba(0.84, 0.54, 0.28, 0.65),
            )
            .resolution(DEBUG_OVERLAY_CIRCLE_RESOLUTION);
    }
    for pos in particle_world.positions() {
        gizmos
            .circle_2d(
                *pos,
                water_overlay_radius,
                Color::srgba(0.10, 0.80, 0.95, 0.85),
            )
            .resolution(DEBUG_OVERLAY_CIRCLE_RESOLUTION);
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

fn paint_single_terrain_cell(
    terrain_world: &mut TerrainWorld,
    cell: IVec2,
    cell_value: TerrainCell,
) -> bool {
    if !cell_in_fixed_world(cell) {
        return false;
    }
    terrain_world.set_cell(cell, cell_value)
}

fn cell_in_fixed_world(cell: IVec2) -> bool {
    let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32;
    let max_cell_x = (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1;
    let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32;
    let max_cell_y = (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1;

    (min_cell_x..=max_cell_x).contains(&cell.x) && (min_cell_y..=max_cell_y).contains(&cell.y)
}

fn finalize_frame_metrics(mut perf_metrics: ResMut<SimulationPerfMetrics>) {
    perf_metrics.physics_time_this_frame_secs = 0.0;
}

fn toggle_button_bg(enabled: bool) -> BackgroundColor {
    if enabled {
        BUTTON_BG_ON.into()
    } else {
        BUTTON_BG_OFF.into()
    }
}
