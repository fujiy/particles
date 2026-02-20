use bevy::prelude::*;

use crate::physics::object::{ObjectData, ObjectWorld};
use crate::physics::particle::{
    PARTICLE_RADIUS_M, ParticleMaterial, ParticleWorld, WATER_KERNEL_RADIUS_M,
    nominal_particle_draw_radius_m,
};
use crate::physics::state::SimUpdateSet;
use crate::physics::terrain::{
    CELL_SIZE_M, CHUNK_WORLD_SIZE_M, TerrainWorld, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y,
    WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y,
};

const BUTTON_BG_OFF: Color = Color::srgba(0.17, 0.18, 0.22, 0.95);
const BUTTON_BG_ON: Color = Color::srgba(0.16, 0.30, 0.46, 0.95);
const BUTTON_BG_HOVER: Color = Color::srgba(0.24, 0.25, 0.30, 0.98);
const BUTTON_BG_PRESS: Color = Color::srgba(0.38, 0.40, 0.48, 0.98);
const GRID_NEIGHBOR_COLOR: Color = Color::srgba(0.27, 0.75, 0.98, 0.28);
const GRID_CHUNK_COLOR: Color = Color::srgba(0.98, 0.78, 0.25, 0.75);
const GRID_OBJECT_COLOR: Color = Color::srgba(0.92, 0.36, 0.12, 0.70);
const GRID_BUTTON_BOTTOM_PX: f32 = 50.0;
const DEBUG_BUTTON_BOTTOM_PX: f32 = 12.0;
const TERRAIN_PARTICLE_RADIUS_M: f32 = PARTICLE_RADIUS_M * 0.55;
const DEBUG_OVERLAY_CIRCLE_RESOLUTION: u32 = 8;

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugOverlayState>()
            .init_resource::<GridOverlayState>()
            .add_systems(Startup, setup_overlay_ui)
            .add_systems(
                Update,
                (handle_grid_overlay_button, handle_overlay_button)
                    .chain()
                    .in_set(SimUpdateSet::Interaction),
            )
            .add_systems(
                Update,
                (
                    update_grid_overlay_button_label,
                    update_overlay_button_label,
                )
                    .chain()
                    .in_set(SimUpdateSet::Ui),
            )
            .add_systems(
                Update,
                (draw_grid_overlay, draw_particle_debug_overlay).in_set(SimUpdateSet::Overlay),
            );
    }
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

#[derive(Component)]
struct OverlayToggleButton;

#[derive(Component)]
struct OverlayToggleButtonLabel;

#[derive(Component)]
struct GridOverlayToggleButton;

#[derive(Component)]
struct GridOverlayToggleButtonLabel;

fn setup_overlay_ui(mut commands: Commands) {
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

fn draw_grid_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<GridOverlayState>,
    object_world: Res<ObjectWorld>,
    particle_world: Res<ParticleWorld>,
) {
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

    let particle_positions = particle_world.positions();
    let particle_masses = particle_world.masses();
    for object in object_world.objects() {
        if let Some((center, theta)) =
            object_pose_for_overlay(object, particle_positions, particle_masses)
        {
            draw_object_grid_cells(&mut gizmos, object, center, theta);
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
    for (index, pos) in particle_world.positions().iter().enumerate() {
        let color = match particle_world.materials()[index] {
            ParticleMaterial::WaterLiquid => Color::srgba(0.10, 0.80, 0.95, 0.85),
            ParticleMaterial::StoneSolid
            | ParticleMaterial::StoneGranular
            | ParticleMaterial::SoilSolid
            | ParticleMaterial::SoilGranular
            | ParticleMaterial::SandSolid
            | ParticleMaterial::SandGranular => Color::srgba(0.63, 0.50, 0.34, 0.85),
        };
        gizmos
            .circle_2d(*pos, water_overlay_radius, color)
            .resolution(DEBUG_OVERLAY_CIRCLE_RESOLUTION);
    }
}

fn toggle_button_bg(enabled: bool) -> BackgroundColor {
    if enabled {
        BUTTON_BG_ON.into()
    } else {
        BUTTON_BG_OFF.into()
    }
}
