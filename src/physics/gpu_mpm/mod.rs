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
    MAX_CHUNK_EVENT_RECORDS, MAX_MOVER_RECORDS, MpmGpuBuffers, MpmGpuControl, MpmGpuParamsRequest,
    MpmGpuRunRequest, MpmGpuStepClock, MpmGpuUploadRequest,
};
use self::node::MpmComputeNode;
use self::pipeline::MpmComputePipelines;
use self::readback::{
    GpuChunkEventReadbackResult, GpuChunkEventReadbackState, GpuMoverApplyAck,
    GpuMoverReadbackResult, GpuMoverReadbackState, GpuReadbackResult, GpuReadbackState,
    GpuStatisticsReadbackResult, GpuStatisticsReadbackState,
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
            upload_mover_results: source.upload_mover_results_frame,
            upload_mover_results_frame: false,
            upload_chunks: source.upload_chunks,
            upload_chunk_diffs: source.upload_chunk_diffs,
            upload_terrain: source.upload_terrain,
            upload_terrain_cell_slot_diffs: source.upload_terrain_cell_slot_diffs,
            chunk_meta: source.chunk_meta.clone(),
            particles: source.particles.clone(),
            mover_results: source.mover_results.clone(),
            chunk_meta_diffs: source.chunk_meta_diffs.clone(),
            terrain_sdf: source.terrain_sdf.clone(),
            terrain_normal: source.terrain_normal.clone(),
            terrain_slot_ids: source.terrain_slot_ids.clone(),
            terrain_cell_solid_slot_diffs: source.terrain_cell_solid_slot_diffs.clone(),
            last_uploaded_terrain_version: source.last_uploaded_terrain_version,
        }
    }
}

impl Clone for MpmGpuUploadRequest {
    fn clone(&self) -> Self {
        Self {
            upload_particles: self.upload_particles,
            upload_particles_frame: self.upload_particles_frame,
            upload_mover_results: self.upload_mover_results,
            upload_mover_results_frame: self.upload_mover_results_frame,
            upload_chunks: self.upload_chunks,
            upload_chunk_diffs: self.upload_chunk_diffs,
            upload_terrain: self.upload_terrain,
            upload_terrain_cell_slot_diffs: self.upload_terrain_cell_slot_diffs,
            chunk_meta: self.chunk_meta.clone(),
            particles: self.particles.clone(),
            mover_results: self.mover_results.clone(),
            chunk_meta_diffs: self.chunk_meta_diffs.clone(),
            terrain_sdf: self.terrain_sdf.clone(),
            terrain_normal: self.terrain_normal.clone(),
            terrain_slot_ids: self.terrain_slot_ids.clone(),
            terrain_cell_solid_slot_diffs: self.terrain_cell_solid_slot_diffs.clone(),
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
    } else if upload.upload_chunk_diffs {
        buffers.upload_chunk_diffs(&queue, &upload.chunk_meta_diffs);
    } else {
        buffers.active_chunk_count = params_req.params.resident_chunk_count;
    }

    if upload.upload_mover_results {
        buffers.upload_mover_results(&queue, &upload.mover_results);
    }

    if upload.upload_terrain && !upload.terrain_sdf.is_empty() && !upload.terrain_normal.is_empty()
    {
        buffers.upload_terrain(&queue, &upload.terrain_sdf, &upload.terrain_normal);
    }
    if upload.upload_terrain_cell_slot_diffs
        && !upload.terrain_slot_ids.is_empty()
        && !upload.terrain_cell_solid_slot_diffs.is_empty()
    {
        buffers.upload_terrain_cell_slot_diffs(
            &queue,
            &upload.terrain_slot_ids,
            &upload.terrain_cell_solid_slot_diffs,
        );
    }
    if upload.upload_terrain_cell_slot_diffs {
        buffers.upload_terrain_update_slots(&queue, &upload.terrain_slot_ids);
    } else {
        buffers.upload_terrain_update_slots(&queue, &[]);
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

fn readback_movers(
    control: Res<MpmGpuControl>,
    buffers: Res<MpmGpuBuffers>,
    mut state: ResMut<GpuMoverReadbackState>,
    readback_result: Res<GpuMoverReadbackResult>,
) {
    if control.init_only || !control.readback_enabled {
        return;
    }
    if !buffers.ready || buffers.particle_count == 0 {
        return;
    }

    if state.mapped {
        let ready = state
            .mapped_ready
            .load(std::sync::atomic::Ordering::Acquire);
        if ready {
            let byte_size = size_of::<u32>() as u64
                + MAX_MOVER_RECORDS as u64 * size_of::<buffers::GpuMoverRecord>() as u64;
            let slice = buffers.mover_readback_buf.slice(..byte_size);
            let data = slice.get_mapped_range();
            if data.len() >= size_of::<u32>() {
                let count = *bytemuck::from_bytes::<u32>(&data[..size_of::<u32>()]);
                let max_count = MAX_MOVER_RECORDS.min(buffers.particle_count);
                let mover_count = count.min(max_count);
                let records_offset = size_of::<u32>();
                let records_bytes = mover_count as usize * size_of::<buffers::GpuMoverRecord>();
                if data.len() >= records_offset + records_bytes {
                    let mut records = Vec::with_capacity(mover_count as usize);
                    for chunk in data[records_offset..records_offset + records_bytes]
                        .chunks_exact(size_of::<buffers::GpuMoverRecord>())
                    {
                        records.push(bytemuck::pod_read_unaligned::<buffers::GpuMoverRecord>(
                            chunk,
                        ));
                    }
                    readback_result.store(records);
                }
            }
            drop(data);
            buffers.mover_readback_buf.unmap();
            state
                .mapped_ready
                .store(false, std::sync::atomic::Ordering::Release);
            state.mapped = false;
            state
                .copy_pending
                .store(false, std::sync::atomic::Ordering::Release);
        }
        return;
    }

    if !state
        .copy_pending
        .load(std::sync::atomic::Ordering::Acquire)
    {
        return;
    }

    let byte_size = size_of::<u32>() as u64
        + MAX_MOVER_RECORDS as u64 * size_of::<buffers::GpuMoverRecord>() as u64;
    let slice = buffers.mover_readback_buf.slice(..byte_size);
    let flag = state.mapped_ready.clone();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        if result.is_ok() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
    });
    state.mapped = true;
    state
        .copy_pending
        .store(false, std::sync::atomic::Ordering::Release);
}

fn readback_chunk_events(
    control: Res<MpmGpuControl>,
    buffers: Res<MpmGpuBuffers>,
    mut state: ResMut<GpuChunkEventReadbackState>,
    readback_result: Res<GpuChunkEventReadbackResult>,
) {
    if control.init_only || !control.readback_enabled {
        return;
    }

    if state.mapped {
        let ready = state
            .mapped_ready
            .load(std::sync::atomic::Ordering::Acquire);
        if ready {
            let byte_size = size_of::<u32>() as u64
                + MAX_CHUNK_EVENT_RECORDS as u64 * size_of::<buffers::GpuChunkEventRecord>() as u64;
            let slice = buffers.chunk_event_readback_buf.slice(..byte_size);
            let data = slice.get_mapped_range();
            if data.len() >= size_of::<u32>() {
                let count = *bytemuck::from_bytes::<u32>(&data[..size_of::<u32>()]);
                let event_count = count.min(MAX_CHUNK_EVENT_RECORDS);
                let records_offset = size_of::<u32>();
                let records_bytes =
                    event_count as usize * size_of::<buffers::GpuChunkEventRecord>();
                if data.len() >= records_offset + records_bytes {
                    let mut events = Vec::with_capacity(event_count as usize);
                    for chunk in data[records_offset..records_offset + records_bytes]
                        .chunks_exact(size_of::<buffers::GpuChunkEventRecord>())
                    {
                        events.push(
                            bytemuck::pod_read_unaligned::<buffers::GpuChunkEventRecord>(chunk),
                        );
                    }
                    readback_result.store(events);
                }
            }
            drop(data);
            buffers.chunk_event_readback_buf.unmap();
            state
                .mapped_ready
                .store(false, std::sync::atomic::Ordering::Release);
            state.mapped = false;
            state
                .copy_pending
                .store(false, std::sync::atomic::Ordering::Release);
        }
        return;
    }

    if !state
        .copy_pending
        .load(std::sync::atomic::Ordering::Acquire)
    {
        return;
    }

    let byte_size = size_of::<u32>() as u64
        + MAX_CHUNK_EVENT_RECORDS as u64 * size_of::<buffers::GpuChunkEventRecord>() as u64;
    let slice = buffers.chunk_event_readback_buf.slice(..byte_size);
    let flag = state.mapped_ready.clone();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        if result.is_ok() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
    });
    state.mapped = true;
    state
        .copy_pending
        .store(false, std::sync::atomic::Ordering::Release);
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
        let mover_readback = GpuMoverReadbackResult::default();
        let mover_readback_for_render = mover_readback.clone();
        let chunk_event_readback = GpuChunkEventReadbackResult::default();
        let chunk_event_readback_for_render = chunk_event_readback.clone();
        let mover_apply_ack = GpuMoverApplyAck::default();
        let mover_apply_ack_for_render = mover_apply_ack.clone();

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
            .insert_resource(mover_readback)
            .insert_resource(chunk_event_readback)
            .insert_resource(mover_apply_ack)
            .configure_sets(
                Update,
                (
                    sync::MpmSyncSet::ApplyReadback,
                    sync::MpmSyncSet::PrepareUpload,
                )
                    .chain()
                    .in_set(SimUpdateSet::Rendering),
            )
            .add_systems(
                Update,
                (
                    sync::apply_world_edit_requests,
                    sync::prepare_particle_upload,
                    sync::prepare_terrain_upload,
                    sync::apply_chunk_event_readback,
                    sync::prepare_gpu_params,
                    sync::prepare_gpu_run_state,
                )
                    .chain()
                    .in_set(sync::MpmSyncSet::PrepareUpload),
            )
            .add_systems(
                Update,
                (
                    sync::apply_gpu_readback,
                    sync::apply_statistics_readback,
                    sync::consume_mover_apply_ack,
                    sync::apply_mover_readback,
                )
                    .chain()
                    .in_set(sync::MpmSyncSet::ApplyReadback),
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
            .insert_resource(mover_readback_for_render)
            .insert_resource(chunk_event_readback_for_render)
            .insert_resource(mover_apply_ack_for_render)
            .init_resource::<GpuReadbackState>()
            .init_resource::<GpuStatisticsReadbackState>()
            .init_resource::<GpuMoverReadbackState>()
            .init_resource::<GpuChunkEventReadbackState>()
            .add_systems(Render, prepare_gpu_uploads.in_set(RenderSystems::Prepare))
            .add_systems(
                Render,
                (
                    readback_particles,
                    readback_statistics,
                    readback_movers,
                    readback_chunk_events,
                )
                    .in_set(RenderSystems::Cleanup),
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
