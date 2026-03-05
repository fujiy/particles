use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::RenderGraphExt;
use bevy::render::render_graph::ViewNodeRunner;
use bevy::render::render_resource::SpecializedRenderPipelines;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::physics::state::SimUpdateSet;
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, TerrainWorld,
};
use crate::render::{TerrainGeneratedChunkCache, TerrainRenderDiagnostics};

const BUTTON_BG_OFF: Color = Color::srgba(0.17, 0.18, 0.22, 0.95);
const BUTTON_BG_ON: Color = Color::srgba(0.16, 0.30, 0.46, 0.95);
const BUTTON_BG_HOVER: Color = Color::srgba(0.24, 0.25, 0.30, 0.98);
const BUTTON_BG_PRESS: Color = Color::srgba(0.38, 0.40, 0.48, 0.98);
const GRID_CHUNK_BOUNDARY_COLOR: Color = Color::srgba(0.80, 0.86, 0.93, 0.22);
const GRID_CACHED_CHUNK_COLOR: Color = Color::srgba(0.32, 0.86, 0.98, 0.62);
const GRID_MODIFIED_CHUNK_COLOR: Color = Color::srgba(0.99, 0.43, 0.22, 0.95);
const GRID_TERRAIN_UPDATED_COLOR: Color = Color::srgba(0.13, 0.85, 0.92, 1.00);
const GRID_PARTICLE_UPDATED_COLOR: Color = Color::srgba(0.76, 0.56, 0.98, 1.00);
const TILE_BUTTON_BOTTOM_PX: f32 = 88.0;
const SDF_BUTTON_BOTTOM_PX: f32 = 126.0;
const PHYSICS_AREA_BUTTON_BOTTOM_PX: f32 = 50.0;
const PARTICLE_BUTTON_BOTTOM_PX: f32 = 12.0;
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
            .init_resource::<OverlayVisibilityOverrides>()
            .add_plugins(ExtractResourcePlugin::<ParticleOverlayState>::default())
            .add_systems(Startup, setup_overlay_ui)
            .add_systems(
                Update,
                (
                    apply_overlay_visibility_overrides,
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

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct OverlayVisibilityOverrides {
    pub chunk: Option<bool>,
    pub sdf: Option<bool>,
    pub physics_area: Option<bool>,
    pub particle: Option<bool>,
}

fn apply_overlay_visibility_overrides(
    mut overrides: ResMut<OverlayVisibilityOverrides>,
    mut tile_overlay_state: ResMut<TileOverlayState>,
    mut sdf_overlay_state: ResMut<SdfOverlayState>,
    mut physics_overlay_state: ResMut<PhysicsAreaOverlayState>,
    mut particle_overlay_state: ResMut<ParticleOverlayState>,
) {
    if let Some(v) = overrides.chunk.take() {
        tile_overlay_state.enabled = v;
    }
    if let Some(v) = overrides.sdf.take() {
        sdf_overlay_state.enabled = v;
    }
    if let Some(v) = overrides.physics_area.take() {
        physics_overlay_state.enabled = v;
    }
    if let Some(v) = overrides.particle.take() {
        particle_overlay_state.enabled = v;
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
