use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::RenderGraphExt;
use bevy::render::render_graph::ViewNodeRunner;
use bevy::render::render_resource::SpecializedRenderPipelines;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};

use crate::physics::state::SimUpdateSet;

mod chunk_physics;
mod particle;
mod sdf;
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
            .add_plugins(ExtractResourcePlugin::<SdfOverlayState>::default())
            .add_plugins(ExtractResourcePlugin::<PhysicsAreaOverlayState>::default())
            .add_systems(Startup, setup_overlay_ui)
            .add_systems(
                Update,
                (
                    apply_overlay_visibility_overrides,
                    handle_tile_overlay_button,
                    handle_sdf_overlay_button,
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
                    update_particle_overlay_button_label,
                )
                    .chain()
                    .in_set(SimUpdateSet::Ui),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SpecializedRenderPipelines<SdfOverlayGpuPipeline>>()
            .add_systems(RenderStartup, init_sdf_overlay_gpu_pipeline)
            .add_systems(
                Render,
                prepare_sdf_overlay_gpu_pipeline.in_set(RenderSystems::Prepare),
            )
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
            .add_render_graph_node::<ViewNodeRunner<SdfOverlayGpuNode>>(Core2d, SdfOverlayGpuLabel)
            .add_render_graph_node::<ViewNodeRunner<ParticleOverlayGpuNode>>(
                Core2d,
                ParticleOverlayGpuLabel,
            )
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::MainTransparentPass,
                    SdfOverlayGpuLabel,
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
pub(crate) struct TileOverlayState {
    pub enabled: bool,
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
    let chunk_override = overrides.chunk.take();
    let physics_override = overrides.physics_area.take();
    if let Some(v) = chunk_override {
        tile_overlay_state.enabled = v;
        physics_overlay_state.enabled = v;
    } else if let Some(v) = physics_override {
        tile_overlay_state.enabled = v;
        physics_overlay_state.enabled = v;
    }
    if let Some(v) = overrides.sdf.take() {
        sdf_overlay_state.enabled = v;
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

#[derive(Resource, Debug, Clone, Copy)]
pub(crate) struct PhysicsAreaOverlayState {
    pub enabled: bool,
}

impl Default for PhysicsAreaOverlayState {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Resource, Debug)]
pub(crate) struct SdfOverlayState {
    pub enabled: bool,
}

impl Default for SdfOverlayState {
    fn default() -> Self {
        Self { enabled: false }
    }
}

use ui::{
    apply_overlay_ui_params, handle_particle_overlay_button, handle_sdf_overlay_button,
    handle_tile_overlay_button, setup_overlay_ui, sync_overlay_button_backgrounds,
    update_particle_overlay_button_label, update_sdf_overlay_button_label,
    update_tile_overlay_button_label,
};

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

impl ExtractResource for SdfOverlayState {
    type Source = SdfOverlayState;
    fn extract_resource(source: &Self::Source) -> Self {
        Self {
            enabled: source.enabled,
        }
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

use sdf::{
    SdfOverlayGpuLabel, SdfOverlayGpuNode, SdfOverlayGpuPipeline, init_sdf_overlay_gpu_pipeline,
    prepare_sdf_overlay_gpu_pipeline,
};
