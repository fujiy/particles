// Render graph node for MLS-MPM GPU compute passes.
//
// Executes: clear -> p2g -> grid_update -> g2p each frame (when simulation is running).

use bevy::log::warn;
use bevy::prelude::*;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext};
use bevy::render::render_resource::{
    BindGroupEntries, CachedPipelineState, ComputePassDescriptor, PipelineCache,
};
use bevy::render::renderer::RenderContext;
use std::mem::size_of;

use super::gpu_resources::MpmGpuBuffers;
use super::gpu_resources::MpmGpuControl;
use super::gpu_resources::MpmGpuRunRequest;
use super::pipeline::MpmComputePipelines;
use super::readback::GpuReadbackState;

const WORKGROUP_SIZE: u32 = 64;
static GPU_READBACK_FRAME_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static CLEAR_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static P2G_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static GRID_UPDATE_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static G2P_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn warn_missing_pipeline_once(
    warned: &std::sync::atomic::AtomicBool,
    name: &str,
    id: bevy::render::render_resource::CachedComputePipelineId,
    cache: &PipelineCache,
) {
    if warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let state = cache.get_compute_pipeline_state(id);
    match state {
        CachedPipelineState::Queued => warn!("gpu_mpm: {name} pipeline is queued"),
        CachedPipelineState::Creating(_) => warn!("gpu_mpm: {name} pipeline is creating"),
        CachedPipelineState::Err(err) => warn!("gpu_mpm: {name} pipeline failed: {err}"),
        CachedPipelineState::Ok(_) => warn!("gpu_mpm: {name} pipeline unexpectedly missing"),
    }
}

pub struct MpmComputeNode;

impl Node for MpmComputeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(buffers) = world.get_resource::<MpmGpuBuffers>() else {
            return Ok(());
        };
        let Some(run_req) = world.get_resource::<MpmGpuRunRequest>() else {
            return Ok(());
        };
        let Some(control) = world.get_resource::<MpmGpuControl>() else {
            return Ok(());
        };
        if !buffers.ready || buffers.particle_count == 0 {
            return Ok(());
        }

        let particle_count = buffers.particle_count;
        let particles_wgs = (particle_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let node_count = buffers.layout.node_count() as u32;
        let nodes_wgs = (node_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let pipelines = world.resource::<MpmComputePipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device();

        // Ensure all pipelines are compiled before trying to dispatch.
        let Some(clear_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.clear_pipeline)
        else {
            warn_missing_pipeline_once(
                &CLEAR_PIPELINE_WARNED,
                "clear",
                pipelines.clear_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(p2g_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.p2g_pipeline) else {
            warn_missing_pipeline_once(
                &P2G_PIPELINE_WARNED,
                "p2g",
                pipelines.p2g_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(grid_update_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.grid_update_pipeline)
        else {
            warn_missing_pipeline_once(
                &GRID_UPDATE_PIPELINE_WARNED,
                "grid_update",
                pipelines.grid_update_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(g2p_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.g2p_pipeline) else {
            warn_missing_pipeline_once(
                &G2P_PIPELINE_WARNED,
                "g2p",
                pipelines.g2p_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };

        if run_req.enabled && run_req.substeps > 0 {
            // ---- Clear ----
            let clear_bg = device.create_bind_group(
                "mpm_clear_bg",
                &pipelines.clear_layout,
                &BindGroupEntries::sequential((
                    buffers.params_buf.as_entire_binding(),
                    buffers.grid_buf.as_entire_binding(),
                )),
            );

            // ---- P2G ----
            // grid_atomic view == grid_buf: atomic u32 CAS writes, no struct read needed.
            let p2g_bg = device.create_bind_group(
                "mpm_p2g_bg",
                &pipelines.p2g_layout,
                &BindGroupEntries::sequential((
                    buffers.params_buf.as_entire_binding(),
                    buffers.particle_buf.as_entire_binding(),
                    buffers.grid_buf.as_entire_binding(),
                )),
            );

            // ---- Grid update ----
            let grid_update_bg = device.create_bind_group(
                "mpm_grid_update_bg",
                &pipelines.grid_update_layout,
                &BindGroupEntries::sequential((
                    buffers.params_buf.as_entire_binding(),
                    buffers.grid_buf.as_entire_binding(),
                    buffers.terrain_sdf_buf.as_entire_binding(),
                    buffers.terrain_normal_buf.as_entire_binding(),
                )),
            );

            // ---- G2P ----
            let g2p_bg = device.create_bind_group(
                "mpm_g2p_bg",
                &pipelines.g2p_layout,
                &BindGroupEntries::sequential((
                    buffers.params_buf.as_entire_binding(),
                    buffers.particle_buf.as_entire_binding(),
                    buffers.grid_buf.as_entire_binding(),
                )),
            );

            let encoder = render_context.command_encoder();
            for _ in 0..run_req.substeps {
                // Pass 1: Clear grid
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("mpm_clear"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(clear_pipeline);
                    pass.set_bind_group(0, &clear_bg, &[]);
                    pass.dispatch_workgroups(nodes_wgs, 1, 1);
                }

                // Pass 2: P2G
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("mpm_p2g"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(p2g_pipeline);
                    pass.set_bind_group(0, &p2g_bg, &[]);
                    pass.dispatch_workgroups(particles_wgs, 1, 1);
                }

                // Pass 3: Grid update
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("mpm_grid_update"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(grid_update_pipeline);
                    pass.set_bind_group(0, &grid_update_bg, &[]);
                    pass.dispatch_workgroups(nodes_wgs, 1, 1);
                }

                // Pass 4: G2P
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("mpm_g2p"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(g2p_pipeline);
                    pass.set_bind_group(0, &g2p_bg, &[]);
                    pass.dispatch_workgroups(particles_wgs, 1, 1);
                }
            }
        }

        // Copy particle buffer to readback buffer each frame.
        // The copy size covers only the active particles.
        if control.readback_enabled && run_req.enabled && run_req.substeps > 0 {
            let can_copy = world
                .get_resource::<GpuReadbackState>()
                .map(|state| !state.mapped)
                .unwrap_or(true);
            if !can_copy {
                return Ok(());
            }
            let interval = control.readback_interval_frames.max(1) as u64;
            let frame =
                GPU_READBACK_FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if frame % interval != 0 {
                return Ok(());
            }
            let particle_byte_size =
                particle_count as u64 * size_of::<super::buffers::GpuParticle>() as u64;
            render_context.command_encoder().copy_buffer_to_buffer(
                &buffers.particle_buf,
                0,
                &buffers.readback_buf,
                0,
                particle_byte_size,
            );
        }

        Ok(())
    }
}
