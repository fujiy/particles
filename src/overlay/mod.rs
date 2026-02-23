use bevy::prelude::*;

use crate::physics::material::{DEFAULT_MATERIAL_PARAMS, particle_radius_m, water_kernel_radius_m};
use crate::physics::state::{PhysicsActiveRegion, PhysicsRegionSettings, SimUpdateSet};
use crate::physics::world::object::{ObjectData, ObjectWorld};
use crate::physics::world::particle::{
    ParticleMaterial, ParticleWorld, nominal_particle_draw_radius_m,
};
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, TerrainWorld, world_to_cell,
};
use crate::render::TerrainRenderDiagnostics;

const BUTTON_BG_OFF: Color = Color::srgba(0.17, 0.18, 0.22, 0.95);
const BUTTON_BG_ON: Color = Color::srgba(0.16, 0.30, 0.46, 0.95);
const BUTTON_BG_HOVER: Color = Color::srgba(0.24, 0.25, 0.30, 0.98);
const BUTTON_BG_PRESS: Color = Color::srgba(0.38, 0.40, 0.48, 0.98);
const GRID_NEIGHBOR_COLOR: Color = Color::srgba(0.27, 0.75, 0.98, 0.28);
const GRID_CHUNK_BOUNDARY_COLOR: Color = Color::srgba(0.90, 0.90, 0.94, 0.32);
const GRID_LOD_CHUNK_COLOR: Color = Color::srgba(0.84, 0.84, 0.86, 0.36);
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
const WATER_KERNEL_RADIUS_M: f32 = water_kernel_radius_m(DEFAULT_MATERIAL_PARAMS);
const TERRAIN_PARTICLE_RADIUS_M: f32 = particle_radius_m(DEFAULT_MATERIAL_PARAMS) * 0.55;
const PARTICLE_OVERLAY_CIRCLE_RESOLUTION: u32 = 8;
const GRID_OBJECT_AXIS_LENGTH_M: f32 = CELL_SIZE_M * 1.6;
const GRID_OBJECT_CENTER_RADIUS_M: f32 = CELL_SIZE_M * 0.12;

mod grid;
mod particle;
mod ui;

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

use ui::{
    handle_grid_overlay_button, handle_particle_overlay_button, setup_overlay_ui,
    update_grid_overlay_button_label, update_grid_overlay_info_text,
    update_particle_overlay_button_label,
};

use grid::draw_grid_overlay;

use particle::draw_particle_overlay;
