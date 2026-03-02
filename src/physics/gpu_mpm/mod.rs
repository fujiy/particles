// GPU-first MLS-MPM compute module.
//
// Implements ADR-0002: single-resolution grid, GPU-resident data, active tile
// (placeholder in v1).
//
// Pipeline: clear → p2g → grid_update → g2p
// All compute runs in the Bevy render world. CPU holds only metadata.

pub mod buffers;
pub mod gpu_resources;
pub mod node;
pub mod pipeline;
pub mod readback;
pub mod shaders;
pub mod sync;

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{RenderGraph, RenderLabel};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Render, RenderApp, RenderSystems};
use std::mem::size_of;

use self::gpu_resources::{
    MpmGpuBuffers, MpmGpuControl, MpmGpuParamsRequest, MpmGpuRunRequest, MpmGpuStepClock,
    MpmGpuUploadRequest,
};
use self::node::MpmComputeNode;
use self::pipeline::MpmComputePipelines;
use self::readback::{GpuReadbackResult, GpuReadbackState};
use self::shaders::MpmShaders;

// ---------------------------------------------------------------------------
// Render graph label
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct MpmComputeLabel;

// ---------------------------------------------------------------------------
// ExtractResource: transfer upload requests to render world
// ---------------------------------------------------------------------------

impl ExtractResource for MpmGpuUploadRequest {
    type Source = MpmGpuUploadRequest;
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

impl Clone for MpmGpuUploadRequest {
    fn clone(&self) -> Self {
        Self {
            upload_particles: self.upload_particles,
            upload_terrain: self.upload_terrain,
            particles: self.particles.clone(),
            terrain_sdf: self.terrain_sdf.clone(),
            terrain_normal: self.terrain_normal.clone(),
        }
    }
}

impl ExtractResource for MpmGpuParamsRequest {
    type Source = MpmGpuParamsRequest;
    fn extract_resource(source: &Self::Source) -> Self {
        Self {
            params: source.params,
        }
    }
}

impl ExtractResource for MpmGpuRunRequest {
    type Source = MpmGpuRunRequest;
    fn extract_resource(source: &Self::Source) -> Self {
        Self {
            enabled: source.enabled,
            substeps: source.substeps,
        }
    }
}

impl ExtractResource for MpmGpuControl {
    type Source = MpmGpuControl;
    fn extract_resource(source: &Self::Source) -> Self {
        *source
    }
}

impl Clone for MpmGpuParamsRequest {
    fn clone(&self) -> Self {
        Self {
            params: self.params,
        }
    }
}

impl Clone for MpmGpuRunRequest {
    fn clone(&self) -> Self {
        Self {
            enabled: self.enabled,
            substeps: self.substeps,
        }
    }
}

// ---------------------------------------------------------------------------
// Prepare system: perform CPU→GPU uploads in render world
// ---------------------------------------------------------------------------

fn prepare_gpu_uploads(
    control: Res<MpmGpuControl>,
    upload: Res<MpmGpuUploadRequest>,
    params_req: Res<MpmGpuParamsRequest>,
    mut buffers: ResMut<MpmGpuBuffers>,
    queue: Res<RenderQueue>,
) {
    if control.init_only {
        return;
    }
    buffers.upload_params(&queue, &params_req.params);

    if upload.upload_particles && !upload.particles.is_empty() {
        buffers.upload_particles(&queue, &upload.particles);
        buffers.ready = true;
    } else if buffers.particle_count > 0 {
        buffers.particle_count = params_req.params.particle_count;
        buffers.ready = buffers.particle_count > 0;
    }

    if upload.upload_terrain && !upload.terrain_sdf.is_empty() && !upload.terrain_normal.is_empty()
    {
        buffers.upload_terrain(&queue, &upload.terrain_sdf, &upload.terrain_normal);
    }
}

// ---------------------------------------------------------------------------
// Cleanup system: async readback — no poll(Wait), no frame stall.
//
// State machine (GpuReadbackState):
//   mapped=false → issue map_async; set mapped=true, record pending_count.
//   mapped=true  → check if callback fired (mapped_ready flag).
//                  If ready: get_mapped_range → parse → store → unmap → mapped=false.
//                  If not:   skip this frame (callback pending).
// ---------------------------------------------------------------------------

fn readback_particles(
    control: Res<MpmGpuControl>,
    buffers: Res<MpmGpuBuffers>,
    mut state: ResMut<GpuReadbackState>,
    readback_result: Res<GpuReadbackResult>,
) {
    if control.init_only || !control.readback_enabled {
        return;
    }
    if !buffers.ready || buffers.particle_count == 0 {
        return;
    }

    // Step 1: if a map_async from a previous frame completed, harvest the data.
    if state.mapped {
        let ready = state
            .mapped_ready
            .load(std::sync::atomic::Ordering::Acquire);
        if ready {
            let particle_count = state.pending_count as usize;
            let byte_size = particle_count as u64 * size_of::<buffers::GpuParticle>() as u64;
            let slice = buffers.readback_buf.slice(..byte_size);
            let data = slice.get_mapped_range();
            let particles: Vec<buffers::GpuParticle> = bytemuck::cast_slice(&data[..])
                .iter()
                .take(particle_count)
                .copied()
                .collect();
            drop(data);
            buffers.readback_buf.unmap();
            state
                .mapped_ready
                .store(false, std::sync::atomic::Ordering::Release);
            state.mapped = false;
            readback_result.store(particles);
        }
        // If not ready yet: callback still pending — don't issue another map_async.
        return;
    }

    // Step 2: issue map_async according to configured cadence.
    state.frame_counter = state.frame_counter.wrapping_add(1);
    let interval = control.readback_interval_frames.max(1) as u64;
    if state.frame_counter % interval != 0 {
        return;
    }
    let particle_count = buffers.particle_count;
    let byte_size = particle_count as u64 * size_of::<buffers::GpuParticle>() as u64;
    let slice = buffers.readback_buf.slice(..byte_size);

    let flag = state.mapped_ready.clone();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        if result.is_ok() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
    });

    state.pending_count = particle_count;
    state.mapped = true;
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct GpuMpmPlugin;

impl Plugin for GpuMpmPlugin {
    fn build(&self, app: &mut App) {
        let readback = GpuReadbackResult::default();
        let readback_for_render = readback.clone();

        app.init_resource::<MpmGpuUploadRequest>()
            .init_resource::<MpmGpuParamsRequest>()
            .init_resource::<MpmGpuRunRequest>()
            .init_resource::<MpmGpuStepClock>()
            .init_resource::<MpmGpuControl>()
            .insert_resource(readback)
            .add_systems(
                Update,
                (
                    sync::ensure_continuum_seed_from_particle_world,
                    sync::prepare_particle_upload,
                    sync::prepare_terrain_upload,
                    sync::prepare_gpu_params,
                    sync::prepare_gpu_run_state,
                )
                    .chain(),
            )
            .add_systems(Update, sync::apply_gpu_readback)
            .add_plugins(ExtractResourcePlugin::<MpmGpuUploadRequest>::default())
            .add_plugins(ExtractResourcePlugin::<MpmGpuParamsRequest>::default())
            .add_plugins(ExtractResourcePlugin::<MpmGpuRunRequest>::default())
            .add_plugins(ExtractResourcePlugin::<MpmGpuControl>::default());

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .insert_resource(readback_for_render)
            .init_resource::<GpuReadbackState>()
            .add_systems(Render, prepare_gpu_uploads.in_set(RenderSystems::Prepare))
            .add_systems(Render, readback_particles.in_set(RenderSystems::Cleanup));

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(MpmComputeLabel, MpmComputeNode);
        // Run compute before camera sub-graphs so updated particle data is visible this frame.
        render_graph.add_node_edge(MpmComputeLabel, CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        let device = render_app.world().resource::<RenderDevice>().clone();
        let buffers = MpmGpuBuffers::new(&device);
        render_app.world_mut().insert_resource(buffers);
        render_app.world_mut().init_resource::<MpmShaders>();
        render_app
            .world_mut()
            .init_resource::<MpmComputePipelines>();
    }
}
