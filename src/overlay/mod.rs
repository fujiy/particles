use bevy::prelude::*;

use crate::physics::object::{ObjectData, ObjectWorld};
use crate::physics::particle::{
    PARTICLE_RADIUS_M, ParticleMaterial, ParticleWorld, WATER_KERNEL_RADIUS_M,
    nominal_particle_draw_radius_m,
};
use crate::physics::state::{PhysicsActiveRegion, PhysicsRegionSettings, SimUpdateSet};
use crate::physics::terrain::{
    CELL_SIZE_M, CHUNK_WORLD_SIZE_M, TerrainWorld,
};
use crate::render::TerrainRenderDiagnostics;

const BUTTON_BG_OFF: Color = Color::srgba(0.17, 0.18, 0.22, 0.95);
const BUTTON_BG_ON: Color = Color::srgba(0.16, 0.30, 0.46, 0.95);
const BUTTON_BG_HOVER: Color = Color::srgba(0.24, 0.25, 0.30, 0.98);
const BUTTON_BG_PRESS: Color = Color::srgba(0.38, 0.40, 0.48, 0.98);
const GRID_NEIGHBOR_COLOR: Color = Color::srgba(0.27, 0.75, 0.98, 0.28);
const GRID_CHUNK_BOUNDARY_COLOR: Color = Color::srgba(0.90, 0.90, 0.94, 0.32);
const GRID_ACTIVE_CHUNK_COLOR: Color = Color::srgba(1.00, 0.08, 0.78, 0.95);
const GRID_HALO_CHUNK_COLOR: Color = Color::srgba(0.16, 0.88, 0.60, 0.72);
const GRID_PHYSICS_REGION_COLOR: Color = Color::srgba(0.96, 0.72, 0.12, 0.98);
const GRID_TERRAIN_UPDATED_COLOR: Color = Color::srgba(0.13, 0.85, 0.92, 1.00);
const GRID_PARTICLE_UPDATED_COLOR: Color = Color::srgba(0.76, 0.56, 0.98, 1.00);
const GRID_OBJECT_COLOR: Color = Color::srgba(0.92, 0.36, 0.12, 0.70);
const GRID_OBJECT_CENTER_COLOR: Color = Color::srgba(0.98, 0.98, 0.98, 0.90);
const GRID_OBJECT_LOCAL_X_COLOR: Color = Color::srgba(0.95, 0.26, 0.21, 0.95);
const GRID_OBJECT_LOCAL_Y_COLOR: Color = Color::srgba(0.18, 0.80, 0.44, 0.95);
const GRID_BUTTON_BOTTOM_PX: f32 = 50.0;
const PARTICLE_BUTTON_BOTTOM_PX: f32 = 12.0;
const TERRAIN_PARTICLE_RADIUS_M: f32 = PARTICLE_RADIUS_M * 0.55;
const PARTICLE_OVERLAY_CIRCLE_RESOLUTION: u32 = 8;
const GRID_OBJECT_AXIS_LENGTH_M: f32 = CELL_SIZE_M * 1.6;
const GRID_OBJECT_CENTER_RADIUS_M: f32 = CELL_SIZE_M * 0.12;

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleOverlayState>()
            .init_resource::<GridOverlayState>()
            .add_systems(Startup, setup_overlay_ui)
            .add_systems(
                Update,
                (handle_grid_overlay_button, handle_particle_overlay_button)
                    .chain()
                    .in_set(SimUpdateSet::Interaction),
            )
            .add_systems(
                Update,
                (
                    update_grid_overlay_button_label,
                    update_particle_overlay_button_label,
                    update_grid_overlay_info_text,
                )
                    .chain()
                    .in_set(SimUpdateSet::Ui),
            )
            .add_systems(
                Update,
                (draw_grid_overlay, draw_particle_overlay).in_set(SimUpdateSet::Overlay),
            );
    }
}

#[derive(Resource, Debug)]
struct ParticleOverlayState {
    enabled: bool,
}

impl Default for ParticleOverlayState {
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
struct ParticleOverlayToggleButton;

#[derive(Component)]
struct ParticleOverlayToggleButtonLabel;

#[derive(Component)]
struct GridOverlayToggleButton;

#[derive(Component)]
struct GridOverlayToggleButtonLabel;

#[derive(Component)]
struct GridOverlayInfoText;

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
                bottom: px(PARTICLE_BUTTON_BOTTOM_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(BUTTON_BG_OFF),
            ParticleOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Particle Overlay: OFF"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                ParticleOverlayToggleButtonLabel,
            ));
        });

    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(14.0),
        TextColor(Color::srgba(0.96, 0.96, 0.98, 0.95)),
        Node {
            position_type: PositionType::Absolute,
            left: px(12.0),
            top: px(12.0),
            ..default()
        },
        GridOverlayInfoText,
    ));
}

fn handle_particle_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<ParticleOverlayToggleButton>),
    >,
    mut overlay_state: ResMut<ParticleOverlayState>,
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
    render_diagnostics: Res<TerrainRenderDiagnostics>,
    mut labels: Query<&mut Text, With<GridOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() && !render_diagnostics.is_changed() {
        return;
    }

    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            format!(
                "Grid Overlay: ON (T:{} P:{})",
                render_diagnostics.terrain_updated_chunk_highlight_frames.len(),
                render_diagnostics.particle_updated_chunk_highlight_frames.len(),
            )
        } else {
            "Grid Overlay: OFF".to_string()
        };
    }
}

fn update_particle_overlay_button_label(
    overlay_state: Res<ParticleOverlayState>,
    mut labels: Query<&mut Text, With<ParticleOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() {
        return;
    }

    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            "Particle Overlay: ON".to_string()
        } else {
            "Particle Overlay: OFF".to_string()
        };
    }
}

fn update_grid_overlay_info_text(
    overlay_state: Res<GridOverlayState>,
    active_region: Res<PhysicsActiveRegion>,
    mut labels: Query<&mut Text, With<GridOverlayInfoText>>,
) {
    if !overlay_state.is_changed() && !active_region.is_changed() {
        return;
    }
    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            format!("Physics Chunks: {}", active_region.active_chunks.len())
        } else {
            String::new()
        };
    }
}

fn draw_grid_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<GridOverlayState>,
    active_region: Res<PhysicsActiveRegion>,
    region_settings: Res<PhysicsRegionSettings>,
    terrain_world: Res<TerrainWorld>,
    render_diagnostics: Res<TerrainRenderDiagnostics>,
    object_world: Res<ObjectWorld>,
    particle_world: Res<ParticleWorld>,
) {
    if !overlay_state.enabled {
        return;
    }

    let loaded_chunks = terrain_world.loaded_chunk_coords();
    for &chunk in &loaded_chunks {
        draw_chunk_outline(&mut gizmos, chunk, GRID_CHUNK_BOUNDARY_COLOR);
    }

    let (Some(min_chunk), Some(max_chunk)) = (active_region.chunk_min, active_region.chunk_max) else {
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
        if chunk.x < min_chunk.x || chunk.x > max_chunk.x || chunk.y < min_chunk.y || chunk.y > max_chunk.y {
            continue;
        }
        draw_chunk_outline(&mut gizmos, chunk, GRID_TERRAIN_UPDATED_COLOR);
    }
    for &chunk in render_diagnostics
        .particle_updated_chunk_highlight_frames
        .keys()
    {
        if chunk.x < min_chunk.x || chunk.x > max_chunk.x || chunk.y < min_chunk.y || chunk.y > max_chunk.y {
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

fn draw_particle_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<ParticleOverlayState>,
    terrain_world: Res<TerrainWorld>,
    particle_world: Res<ParticleWorld>,
    object_world: Res<ObjectWorld>,
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
            .resolution(PARTICLE_OVERLAY_CIRCLE_RESOLUTION);
    }
    for (index, pos) in particle_world.positions().iter().enumerate() {
        let is_object_particle = object_world.object_of_particle(index).is_some();
        let is_active = particle_world.is_particle_active_in_region(index);
        let is_halo = particle_world.is_particle_in_halo_region(index);
        if !is_object_particle && !is_active && !is_halo {
            continue;
        }
        let color = if is_object_particle {
            let alpha = if is_active { 0.95 } else { 0.55 };
            Color::srgba(0.98, 0.90, 0.38, alpha)
        } else if is_halo {
            Color::srgba(0.22, 0.92, 0.62, 0.82)
        } else {
            match particle_world.materials()[index] {
                ParticleMaterial::WaterLiquid => Color::srgba(0.10, 0.80, 0.95, 0.85),
                ParticleMaterial::StoneSolid
                | ParticleMaterial::StoneGranular
                | ParticleMaterial::SoilSolid
                | ParticleMaterial::SoilGranular
                | ParticleMaterial::SandSolid
                | ParticleMaterial::SandGranular => Color::srgba(0.63, 0.50, 0.34, 0.85),
            }
        };
        gizmos
            .circle_2d(*pos, water_overlay_radius, color)
            .resolution(PARTICLE_OVERLAY_CIRCLE_RESOLUTION);
    }
}

fn toggle_button_bg(enabled: bool) -> BackgroundColor {
    if enabled {
        BUTTON_BG_ON.into()
    } else {
        BUTTON_BG_OFF.into()
    }
}

fn draw_chunk_outline(gizmos: &mut Gizmos, chunk: IVec2, color: Color) {
    let x0 = chunk.x as f32 * CHUNK_WORLD_SIZE_M;
    let x1 = (chunk.x as f32 + 1.0) * CHUNK_WORLD_SIZE_M;
    let y0 = chunk.y as f32 * CHUNK_WORLD_SIZE_M;
    let y1 = (chunk.y as f32 + 1.0) * CHUNK_WORLD_SIZE_M;
    gizmos.line_2d(Vec2::new(x0, y0), Vec2::new(x1, y0), color);
    gizmos.line_2d(Vec2::new(x1, y0), Vec2::new(x1, y1), color);
    gizmos.line_2d(Vec2::new(x1, y1), Vec2::new(x0, y1), color);
    gizmos.line_2d(Vec2::new(x0, y1), Vec2::new(x0, y0), color);
}
