mod particle;
mod render;
mod terrain;

use std::time::Instant;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use particle::{
    PARTICLE_RADIUS_M, ParticleWorld, SMOOTHING_RADIUS_M, nominal_particle_draw_radius_m,
};
use render::{
    TerrainRenderState, WaterRenderState, bootstrap_terrain_chunks,
    sync_dirty_terrain_chunks_to_render, sync_water_dots_to_render,
};
pub use terrain::cell_to_world_center;
use terrain::{
    CHUNK_WORLD_SIZE_M, TerrainWorld, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X,
    WORLD_MIN_CHUNK_Y,
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
const TERRAIN_PARTICLE_RADIUS_M: f32 = PARTICLE_RADIUS_M * 0.55;
const DEBUG_OVERLAY_CIRCLE_RESOLUTION: u32 = 8;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainWorld>()
            .init_resource::<ParticleWorld>()
            .init_resource::<SimulationState>()
            .init_resource::<DebugOverlayState>()
            .init_resource::<GridOverlayState>()
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
                    sync_dirty_terrain_chunks_to_render,
                    sync_water_dots_to_render,
                    handle_grid_overlay_button,
                    handle_overlay_button,
                    update_simulation_hud,
                    update_grid_overlay_button_label,
                    update_overlay_button_label,
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

fn step_water_particles(
    sim_state: Res<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut perf_metrics: ResMut<SimulationPerfMetrics>,
) {
    let running = sim_state.running;
    if running {
        terrain_world.rebuild_static_particles_if_dirty(SMOOTHING_RADIUS_M);
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
    terrain_world.rebuild_static_particles_if_dirty(SMOOTHING_RADIUS_M);
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

    let neighbor_step = SMOOTHING_RADIUS_M;
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
