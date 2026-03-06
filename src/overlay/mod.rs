use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::RenderGraphExt;
use bevy::render::render_graph::ViewNodeRunner;
use bevy::render::render_resource::SpecializedRenderPipelines;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::physics::state::SimUpdateSet;
use crate::physics::world::terrain::{CHUNK_WORLD_SIZE_M, TerrainWorld};

mod chunk_physics;
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
            .add_plugins(ExtractResourcePlugin::<TileOverlayState>::default())
            .add_plugins(ExtractResourcePlugin::<PhysicsAreaOverlayState>::default())
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
                    apply_overlay_ui_params,
                    sync_overlay_button_backgrounds,
                    update_tile_overlay_button_label,
                    update_sdf_overlay_button_label,
                    update_physics_area_overlay_button_label,
                    update_particle_overlay_button_label,
                    update_overlay_info_text,
                )
                    .chain()
                    .in_set(SimUpdateSet::Ui),
            )
            .add_systems(Update, (draw_sdf_overlay,).in_set(SimUpdateSet::Overlay));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SpecializedRenderPipelines<ChunkPhysicsOverlayGpuPipeline>>()
            .add_systems(RenderStartup, init_chunk_physics_overlay_gpu_pipeline)
            .add_systems(
                Render,
                prepare_chunk_physics_overlay_gpu_pipeline.in_set(RenderSystems::Prepare),
            )
            .init_resource::<SpecializedRenderPipelines<ParticleOverlayGpuPipeline>>()
            .add_systems(RenderStartup, init_particle_overlay_gpu_pipeline)
            .add_systems(
                Render,
                prepare_particle_overlay_gpu_pipeline.in_set(RenderSystems::Prepare),
            )
            .add_render_graph_node::<ViewNodeRunner<ChunkPhysicsOverlayGpuNode>>(
                Core2d,
                ChunkPhysicsOverlayGpuLabel,
            )
            .add_render_graph_node::<ViewNodeRunner<ParticleOverlayGpuNode>>(
                Core2d,
                ParticleOverlayGpuLabel,
            )
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::MainTransparentPass,
                    ChunkPhysicsOverlayGpuLabel,
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

#[derive(Resource, Debug, Clone, Copy)]
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

#[derive(Resource, Debug, Clone, Copy)]
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
    apply_overlay_ui_params, handle_particle_overlay_button, handle_physics_area_overlay_button,
    handle_sdf_overlay_button, handle_tile_overlay_button, setup_overlay_ui,
    sync_overlay_button_backgrounds, update_overlay_info_text, update_particle_overlay_button_label,
    update_physics_area_overlay_button_label, update_sdf_overlay_button_label,
    update_tile_overlay_button_label,
};

use grid::draw_sdf_overlay;

impl ExtractResource for ParticleOverlayState {
    type Source = ParticleOverlayState;
    fn extract_resource(source: &Self::Source) -> Self {
        *source
    }
}

impl ExtractResource for TileOverlayState {
    type Source = TileOverlayState;
    fn extract_resource(source: &Self::Source) -> Self {
        *source
    }
}

impl ExtractResource for PhysicsAreaOverlayState {
    type Source = PhysicsAreaOverlayState;
    fn extract_resource(source: &Self::Source) -> Self {
        *source
    }
}

use chunk_physics::{
    ChunkPhysicsOverlayGpuLabel, ChunkPhysicsOverlayGpuNode, ChunkPhysicsOverlayGpuPipeline,
    init_chunk_physics_overlay_gpu_pipeline, prepare_chunk_physics_overlay_gpu_pipeline,
};

use particle::{
    ParticleOverlayGpuLabel, ParticleOverlayGpuNode, ParticleOverlayGpuPipeline,
    init_particle_overlay_gpu_pipeline, prepare_particle_overlay_gpu_pipeline,
};
