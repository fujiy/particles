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
pub mod phase;
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
use self::readback::{
    GpuReadbackResult, GpuReadbackState, GpuStatisticsReadbackResult, GpuStatisticsReadbackState,
};
use self::shaders::MpmShaders;
use crate::physics::state::SimUpdateSet;

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
        Self {
            upload_particles: source.upload_particles_frame,
            upload_particles_frame: false,
            upload_chunks: source.upload_chunks,
            upload_terrain: source.upload_terrain,
            chunk_meta: source.chunk_meta.clone(),
            particles: source.particles.clone(),
            terrain_sdf: source.terrain_sdf.clone(),
            terrain_normal: source.terrain_normal.clone(),
            last_uploaded_terrain_version: source.last_uploaded_terrain_version,
        }
    }
}

impl Clone for MpmGpuUploadRequest {
    fn clone(&self) -> Self {
        Self {
            upload_particles: self.upload_particles,
            upload_particles_frame: self.upload_particles_frame,
            upload_chunks: self.upload_chunks,
            upload_terrain: self.upload_terrain,
            chunk_meta: self.chunk_meta.clone(),
            particles: self.particles.clone(),
            terrain_sdf: self.terrain_sdf.clone(),
            terrain_normal: self.terrain_normal.clone(),
            last_uploaded_terrain_version: self.last_uploaded_terrain_version,
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

impl ExtractResource for sync::MpmStatisticsStatus {
    type Source = sync::MpmStatisticsStatus;
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

    if upload.upload_particles {
        buffers.upload_particles(&queue, &upload.particles);
        buffers.ready = true;
    } else if buffers.particle_count > 0 {
        buffers.particle_count = params_req.params.particle_count;
        buffers.ready = buffers.particle_count > 0;
    }

    if upload.upload_chunks {
        buffers.upload_chunks(&queue, &upload.chunk_meta);
    } else {
        buffers.active_chunk_count = params_req.params.resident_chunk_count;
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

fn readback_statistics(
    control: Res<MpmGpuControl>,
    stats_status: Res<sync::MpmStatisticsStatus>,
    buffers: Res<MpmGpuBuffers>,
    mut state: ResMut<GpuStatisticsReadbackState>,
    readback_result: Res<GpuStatisticsReadbackResult>,
) {
    if control.init_only || !control.readback_enabled {
        return;
    }
    if !stats_status.any_enabled() {
        return;
    }

    if state.mapped {
        let ready = state
            .mapped_ready
            .load(std::sync::atomic::Ordering::Acquire);
        if ready {
            let slice = buffers
                .stats_scalar_readback_buf
                .slice(..size_of::<buffers::GpuStatisticsScalars>() as u64);
            let data = slice.get_mapped_range();
            if data.len() >= size_of::<buffers::GpuStatisticsScalars>() {
                let counts = bytemuck::from_bytes::<buffers::GpuStatisticsScalars>(
                    &data[..size_of::<buffers::GpuStatisticsScalars>()],
                );
                readback_result.store(*counts);
            }
            drop(data);
            buffers.stats_scalar_readback_buf.unmap();
            state
                .mapped_ready
                .store(false, std::sync::atomic::Ordering::Release);
            state.mapped = false;
        }
        return;
    }

    state.frame_counter = state.frame_counter.wrapping_add(1);
    let interval = control.readback_interval_frames.max(1) as u64;
    if state.frame_counter % interval != 0 {
        return;
    }

    let slice = buffers
        .stats_scalar_readback_buf
        .slice(..size_of::<buffers::GpuStatisticsScalars>() as u64);
    let flag = state.mapped_ready.clone();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        if result.is_ok() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
    });
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
        let statistics_readback = GpuStatisticsReadbackResult::default();
        let statistics_readback_for_render = statistics_readback.clone();

        app.init_resource::<MpmGpuUploadRequest>()
            .init_resource::<MpmGpuParamsRequest>()
            .init_resource::<MpmGpuRunRequest>()
            .init_resource::<MpmGpuStepClock>()
            .init_resource::<MpmGpuControl>()
            .init_resource::<sync::MpmStatisticsStatus>()
            .add_message::<sync::GpuWorldEditRequest>()
            .init_resource::<sync::MpmReadbackSnapshot>()
            .init_resource::<sync::MpmChunkResidencyState>()
            .init_resource::<sync::MpmStatisticsSnapshot>()
            .insert_resource(readback)
            .insert_resource(statistics_readback)
            .add_systems(
                Update,
                (
                    sync::apply_world_edit_requests,
                    sync::prepare_particle_upload,
                    sync::prepare_terrain_upload,
                    sync::prepare_gpu_params,
                    sync::prepare_gpu_run_state,
                )
                    .chain()
                    .in_set(SimUpdateSet::Rendering),
            )
            .add_systems(
                Update,
                (sync::apply_gpu_readback, sync::apply_statistics_readback)
                    .chain()
                    .in_set(SimUpdateSet::Rendering),
            )
            .add_plugins(ExtractResourcePlugin::<MpmGpuUploadRequest>::default())
            .add_plugins(ExtractResourcePlugin::<MpmGpuParamsRequest>::default())
            .add_plugins(ExtractResourcePlugin::<MpmGpuRunRequest>::default())
            .add_plugins(ExtractResourcePlugin::<MpmGpuControl>::default())
            .add_plugins(ExtractResourcePlugin::<sync::MpmStatisticsStatus>::default());

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .insert_resource(readback_for_render)
            .insert_resource(statistics_readback_for_render)
            .init_resource::<GpuReadbackState>()
            .init_resource::<GpuStatisticsReadbackState>()
            .add_systems(Render, prepare_gpu_uploads.in_set(RenderSystems::Prepare))
            .add_systems(
                Render,
                (readback_particles, readback_statistics).in_set(RenderSystems::Cleanup),
            );

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
