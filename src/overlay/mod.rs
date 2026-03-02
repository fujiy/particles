use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::ViewNodeRunner;
use bevy::render::render_graph::RenderGraphExt;
use bevy::render::render_resource::SpecializedRenderPipelines;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::physics::material::{DEFAULT_MATERIAL_PARAMS, water_kernel_radius_m};
use crate::physics::state::{PhysicsActiveRegion, PhysicsRegionSettings, SimUpdateSet};
use crate::physics::world::object::{ObjectData, ObjectWorld};
use crate::physics::world::particle::ParticleWorld;
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, TerrainWorld,
};
use crate::render::{TerrainRenderDiagnostics, WaterDotGpuLabel};

const BUTTON_BG_OFF: Color = Color::srgba(0.17, 0.18, 0.22, 0.95);
const BUTTON_BG_ON: Color = Color::srgba(0.16, 0.30, 0.46, 0.95);
const BUTTON_BG_HOVER: Color = Color::srgba(0.24, 0.25, 0.30, 0.98);
const BUTTON_BG_PRESS: Color = Color::srgba(0.38, 0.40, 0.48, 0.98);
const GRID_NEIGHBOR_COLOR: Color = Color::srgba(0.27, 0.75, 0.98, 0.28);
const GRID_FULL_TILE_COLOR: Color = Color::srgba(0.78, 0.84, 0.92, 0.34);
const GRID_LOD_CHUNK_COLOR: Color = Color::srgba(0.84, 0.84, 0.86, 0.36);
const GRID_ACTIVE_CHUNK_COLOR: Color = Color::srgba(1.00, 0.08, 0.78, 0.95);
const GRID_HALO_CHUNK_COLOR: Color = Color::srgba(0.16, 0.88, 0.60, 0.72);
const GRID_PHYSICS_REGION_COLOR: Color = Color::srgba(0.96, 0.72, 0.12, 0.98);
const GRID_TERRAIN_UPDATED_COLOR: Color = Color::srgba(0.13, 0.85, 0.92, 1.00);
const GRID_PARTICLE_UPDATED_COLOR: Color = Color::srgba(0.76, 0.56, 0.98, 1.00);
const GRID_SUB_BLOCK_DEBT_HOT_COLOR: Color = Color::srgba(0.96, 0.12, 0.12, 0.96);
const GRID_SUB_BLOCK_LABEL_BASE_COLOR: Color = Color::srgba(1.00, 1.00, 1.00, 0.98);
const GRID_OBJECT_COLOR: Color = Color::srgba(0.92, 0.36, 0.12, 0.70);
const GRID_OBJECT_CENTER_COLOR: Color = Color::srgba(0.98, 0.98, 0.98, 0.90);
const GRID_OBJECT_LOCAL_X_COLOR: Color = Color::srgba(0.95, 0.26, 0.21, 0.95);
const GRID_OBJECT_LOCAL_Y_COLOR: Color = Color::srgba(0.18, 0.80, 0.44, 0.95);
const TILE_BUTTON_BOTTOM_PX: f32 = 88.0;
const SDF_BUTTON_BOTTOM_PX: f32 = 126.0;
const PHYSICS_AREA_BUTTON_BOTTOM_PX: f32 = 50.0;
const PARTICLE_BUTTON_BOTTOM_PX: f32 = 12.0;
const WATER_KERNEL_RADIUS_M: f32 = water_kernel_radius_m(DEFAULT_MATERIAL_PARAMS);
const GRID_OBJECT_AXIS_LENGTH_M: f32 = CELL_SIZE_M * 1.6;
const GRID_OBJECT_CENTER_RADIUS_M: f32 = CELL_SIZE_M * 0.12;
const TERRAIN_SDF_OVERLAY_STEP_M: f32 = CELL_SIZE_M;
const TERRAIN_SDF_OVERLAY_CELL_RADIUS_M: f32 = CELL_SIZE_M * 0.45;
const TERRAIN_SDF_OVERLAY_RANGE_M: f32 = CELL_SIZE_M * 6.0;
const TERRAIN_SDF_OVERLAY_NEGATIVE_FILL_Z: f32 = 0.2;

mod grid;
mod particle;
mod ui;

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleOverlayState>()
            .init_resource::<TileOverlayState>()
            .init_resource::<SdfOverlayState>()
            .init_resource::<PhysicsAreaOverlayState>()
            .add_plugins(ExtractResourcePlugin::<ParticleOverlayState>::default())
            .add_systems(Startup, setup_overlay_ui)
            .add_systems(
                Update,
                (
                    handle_tile_overlay_button,
                    handle_sdf_overlay_button,
                    handle_physics_area_overlay_button,
                    handle_particle_overlay_button,
                )
                    .chain()
                    .in_set(SimUpdateSet::Interaction),
            )
            .add_systems(
                Update,
                (
                    update_tile_overlay_button_label,
                    update_sdf_overlay_button_label,
                    update_physics_area_overlay_button_label,
                    update_particle_overlay_button_label,
                    update_overlay_info_text,
                )
                    .chain()
                    .in_set(SimUpdateSet::Ui),
            )
            .add_systems(
                Update,
                (
                    draw_tile_overlay,
                    draw_sdf_overlay,
                    draw_physics_area_overlay,
                )
                    .in_set(SimUpdateSet::Overlay),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SpecializedRenderPipelines<ParticleOverlayGpuPipeline>>()
            .add_systems(RenderStartup, init_particle_overlay_gpu_pipeline)
            .add_systems(
                Render,
                prepare_particle_overlay_gpu_pipeline.in_set(RenderSystems::Prepare),
            )
            .add_render_graph_node::<ViewNodeRunner<ParticleOverlayGpuNode>>(
                Core2d,
                ParticleOverlayGpuLabel,
            )
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::MainTransparentPass,
                    WaterDotGpuLabel,
                    ParticleOverlayGpuLabel,
                    Node2d::EndMainPass,
                ),
            );
    }
}

#[derive(Resource, Debug, Clone, Copy)]
struct ParticleOverlayState {
    enabled: bool,
}

impl Default for ParticleOverlayState {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Resource, Debug)]
struct TileOverlayState {
    enabled: bool,
}

impl Default for TileOverlayState {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Component)]
struct ParticleOverlayToggleButton;

#[derive(Component)]
struct ParticleOverlayToggleButtonLabel;

#[derive(Component)]
struct TileOverlayToggleButton;

#[derive(Component)]
struct TileOverlayToggleButtonLabel;

#[derive(Component)]
struct SdfOverlayToggleButton;

#[derive(Component)]
struct SdfOverlayToggleButtonLabel;

#[derive(Component)]
struct PhysicsAreaOverlayToggleButton;

#[derive(Component)]
struct PhysicsAreaOverlayToggleButtonLabel;

#[derive(Resource, Debug)]
struct PhysicsAreaOverlayState {
    enabled: bool,
}

impl Default for PhysicsAreaOverlayState {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Resource, Debug)]
struct SdfOverlayState {
    enabled: bool,
}

impl Default for SdfOverlayState {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Component)]
struct OverlayInfoText;

use ui::{
    handle_particle_overlay_button, handle_physics_area_overlay_button, handle_sdf_overlay_button,
    handle_tile_overlay_button, setup_overlay_ui, update_overlay_info_text,
    update_particle_overlay_button_label, update_physics_area_overlay_button_label,
    update_sdf_overlay_button_label, update_tile_overlay_button_label,
};

use grid::{draw_physics_area_overlay, draw_sdf_overlay, draw_tile_overlay};

impl ExtractResource for ParticleOverlayState {
    type Source = ParticleOverlayState;
    fn extract_resource(source: &Self::Source) -> Self {
        *source
    }
}

use particle::{
    ParticleOverlayGpuLabel, ParticleOverlayGpuNode, ParticleOverlayGpuPipeline,
    init_particle_overlay_gpu_pipeline, prepare_particle_overlay_gpu_pipeline,
};
