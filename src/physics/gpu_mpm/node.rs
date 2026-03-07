// Render graph node for MLS-MPM GPU compute passes.
//
// Executes: clear -> p2g -> grid_update -> g2p each frame (when simulation is running),
// plus optional statistics compute passes.

use bevy::log::warn;
use bevy::prelude::*;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext};
use bevy::render::render_resource::{
    BindGroupEntries, CachedPipelineState, ComputePassDescriptor, PipelineCache,
};
use bevy::render::renderer::RenderContext;
use std::mem::size_of;

use super::buffers::{GPU_STATS_SCALAR_LANES, GpuStatisticsScalars};
use super::gpu_resources::{MpmGpuBuffers, MpmGpuControl, MpmGpuParamsRequest, MpmGpuRunRequest};
use super::pipeline::MpmComputePipelines;
use super::readback::{GpuMoverReadbackState, GpuReadbackState, GpuStatisticsReadbackState};
use super::sync::MpmStatisticsStatus;

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
static EXTRACT_MOVERS_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static APPLY_MOVER_RESULTS_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_CLEAR_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_TOTAL_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_PHASE_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_MAX_SPEED_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_PEN_TRACK_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_WATER_HIST_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_WATER_FINAL_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_REPOSE_BOUNDS_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_REPOSE_FINAL_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_INTERACTION_CLEAR_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_INTERACTION_MARK_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_INTERACTION_PRIMARY_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_INTERACTION_FINAL_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_GRID_DENSITY_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static STATS_GRID_DENSITY_FINAL_PIPELINE_WARNED: std::sync::atomic::AtomicBool =
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
        let Some(params_req) = world.get_resource::<MpmGpuParamsRequest>() else {
            return Ok(());
        };
        let Some(control) = world.get_resource::<MpmGpuControl>() else {
            return Ok(());
        };

        let particle_count = buffers.particle_count;
        let particles_wgs = (particle_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let node_count = params_req
            .params
            .grid_width
            .saturating_mul(params_req.params.grid_height);
        if node_count == 0 {
            return Ok(());
        }
        let nodes_wgs = (node_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let has_particles = buffers.ready && particle_count > 0;
        let should_step = has_particles && run_req.enabled && run_req.substeps > 0;

        let pipelines = world.resource::<MpmComputePipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = render_context.render_device().clone();

        if should_step {
            let Some(clear_pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.clear_pipeline)
            else {
                warn_missing_pipeline_once(
                    &CLEAR_PIPELINE_WARNED,
                    "clear",
                    pipelines.clear_pipeline,
                    pipeline_cache,
                );
                return Ok(());
            };
            let Some(p2g_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.p2g_pipeline)
            else {
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
            let Some(g2p_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.g2p_pipeline)
            else {
                warn_missing_pipeline_once(
                    &G2P_PIPELINE_WARNED,
                    "g2p",
                    pipelines.g2p_pipeline,
                    pipeline_cache,
                );
                return Ok(());
            };
            let Some(extract_movers_pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.extract_movers_pipeline)
            else {
                warn_missing_pipeline_once(
                    &EXTRACT_MOVERS_PIPELINE_WARNED,
                    "extract_movers",
                    pipelines.extract_movers_pipeline,
                    pipeline_cache,
                );
                return Ok(());
            };

            let clear_bg = device.create_bind_group(
                "mpm_clear_bg",
                &pipelines.clear_layout,
                &BindGroupEntries::sequential((
                    buffers.params_buf.as_entire_binding(),
                    buffers.grid_buf.as_entire_binding(),
                )),
            );
            let p2g_bg = device.create_bind_group(
                "mpm_p2g_bg",
                &pipelines.p2g_layout,
                &BindGroupEntries::sequential((
                    buffers.params_buf.as_entire_binding(),
                    buffers.particle_buf.as_entire_binding(),
                    buffers.grid_buf.as_entire_binding(),
                )),
            );
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
            let g2p_bg = device.create_bind_group(
                "mpm_g2p_bg",
                &pipelines.g2p_layout,
                &BindGroupEntries::sequential((
                    buffers.params_buf.as_entire_binding(),
                    buffers.particle_buf.as_entire_binding(),
                    buffers.grid_buf.as_entire_binding(),
                )),
            );
            let extract_movers_bg = device.create_bind_group(
                "mpm_extract_movers_bg",
                &pipelines.extract_movers_layout,
                &BindGroupEntries::sequential((
                    buffers.params_buf.as_entire_binding(),
                    buffers.particle_buf.as_entire_binding(),
                    buffers.chunk_meta_buf.as_entire_binding(),
                    buffers.mover_count_buf.as_entire_binding(),
                    buffers.mover_buf.as_entire_binding(),
                )),
            );
            let mover_result_wgs =
                (buffers.mover_result_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let apply_mover_results_bg = if buffers.mover_result_count > 0 {
                let Some(pipeline) =
                    pipeline_cache.get_compute_pipeline(pipelines.apply_mover_results_pipeline)
                else {
                    warn_missing_pipeline_once(
                        &APPLY_MOVER_RESULTS_PIPELINE_WARNED,
                        "apply_mover_results",
                        pipelines.apply_mover_results_pipeline,
                        pipeline_cache,
                    );
                    return Ok(());
                };
                let bg = device.create_bind_group(
                    "mpm_apply_mover_results_bg",
                    &pipelines.apply_mover_results_layout,
                    &BindGroupEntries::sequential((
                        buffers.params_buf.as_entire_binding(),
                        buffers.particle_buf.as_entire_binding(),
                        buffers.mover_result_count_buf.as_entire_binding(),
                        buffers.mover_result_buf.as_entire_binding(),
                    )),
                );
                Some((pipeline, bg))
            } else {
                None
            };

            let encoder = render_context.command_encoder();
            if let Some((pipeline, bg)) = apply_mover_results_bg.as_ref() {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("mpm_apply_mover_results"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(mover_result_wgs.max(1), 1, 1);
            }
            for _ in 0..run_req.substeps {
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("mpm_clear"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(clear_pipeline);
                    pass.set_bind_group(0, &clear_bg, &[]);
                    pass.dispatch_workgroups(nodes_wgs, 1, 1);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("mpm_p2g"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(p2g_pipeline);
                    pass.set_bind_group(0, &p2g_bg, &[]);
                    pass.dispatch_workgroups(particles_wgs, 1, 1);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("mpm_grid_update"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(grid_update_pipeline);
                    pass.set_bind_group(0, &grid_update_bg, &[]);
                    pass.dispatch_workgroups(nodes_wgs, 1, 1);
                }
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
            encoder.clear_buffer(&buffers.mover_count_buf, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("mpm_extract_movers"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(extract_movers_pipeline);
                pass.set_bind_group(0, &extract_movers_bg, &[]);
                pass.dispatch_workgroups(particles_wgs.max(1), 1, 1);
            }
        }

        let stats_status = world
            .get_resource::<MpmStatisticsStatus>()
            .copied()
            .unwrap_or_default();
        let stats_enabled = stats_status.any_enabled();

        if stats_enabled {
            let Some(stats_clear_pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.stats_clear_pipeline)
            else {
                warn_missing_pipeline_once(
                    &STATS_CLEAR_PIPELINE_WARNED,
                    "stats_clear",
                    pipelines.stats_clear_pipeline,
                    pipeline_cache,
                );
                return Ok(());
            };

            let stats_clear_bg = device.create_bind_group(
                "mpm_stats_clear_bg",
                &pipelines.stats_clear_layout,
                &BindGroupEntries::sequential((buffers.stats_scalar_buf.as_entire_binding(),)),
            );
            {
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("mpm_stats_clear"),
                            timestamp_writes: None,
                        });
                pass.set_pipeline(stats_clear_pipeline);
                pass.set_bind_group(0, &stats_clear_bg, &[]);
                let clear_wgs =
                    ((GPU_STATS_SCALAR_LANES as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(clear_wgs.max(1), 1, 1);
            }

            if has_particles {
                if stats_status.total_particles {
                    let Some(stats_total_pipeline) =
                        pipeline_cache.get_compute_pipeline(pipelines.stats_total_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_TOTAL_PIPELINE_WARNED,
                            "stats_total_particles",
                            pipelines.stats_total_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let stats_total_bg = device.create_bind_group(
                        "mpm_stats_total_bg",
                        &pipelines.stats_total_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.particle_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("mpm_stats_total_particles"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_pipeline(stats_total_pipeline);
                    pass.set_bind_group(0, &stats_total_bg, &[]);
                    pass.dispatch_workgroups(particles_wgs, 1, 1);
                }

                if stats_status.phase_counts {
                    let Some(stats_phase_pipeline) =
                        pipeline_cache.get_compute_pipeline(pipelines.stats_phase_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_PHASE_PIPELINE_WARNED,
                            "stats_phase_counts",
                            pipelines.stats_phase_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let stats_phase_bg = device.create_bind_group(
                        "mpm_stats_phase_bg",
                        &pipelines.stats_phase_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.particle_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("mpm_stats_phase_counts"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_pipeline(stats_phase_pipeline);
                    pass.set_bind_group(0, &stats_phase_bg, &[]);
                    pass.dispatch_workgroups(particles_wgs, 1, 1);
                }

                if stats_status.max_speed {
                    let Some(stats_max_speed_pipeline) =
                        pipeline_cache.get_compute_pipeline(pipelines.stats_max_speed_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_MAX_SPEED_PIPELINE_WARNED,
                            "stats_max_speed",
                            pipelines.stats_max_speed_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let stats_max_speed_bg = device.create_bind_group(
                        "mpm_stats_max_speed_bg",
                        &pipelines.stats_max_speed_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.particle_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("mpm_stats_max_speed"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_pipeline(stats_max_speed_pipeline);
                    pass.set_bind_group(0, &stats_max_speed_bg, &[]);
                    pass.dispatch_workgroups(particles_wgs, 1, 1);
                }

                if stats_status.penetration || stats_status.tracked_summary {
                    let Some(stats_pen_track_pipeline) = pipeline_cache
                        .get_compute_pipeline(pipelines.stats_penetration_tracking_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_PEN_TRACK_PIPELINE_WARNED,
                            "stats_penetration_tracking",
                            pipelines.stats_penetration_tracking_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let bg = device.create_bind_group(
                        "mpm_stats_penetration_tracking_bg",
                        &pipelines.stats_penetration_tracking_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.particle_buf.as_entire_binding(),
                            buffers.terrain_sdf_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("mpm_stats_penetration_tracking"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_pipeline(stats_pen_track_pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(particles_wgs, 1, 1);
                }

                if stats_status.water_surface_p95 {
                    let Some(hist_pipeline) = pipeline_cache
                        .get_compute_pipeline(pipelines.stats_water_surface_hist_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_WATER_HIST_PIPELINE_WARNED,
                            "stats_water_surface_hist",
                            pipelines.stats_water_surface_hist_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let hist_bg = device.create_bind_group(
                        "mpm_stats_water_hist_bg",
                        &pipelines.stats_water_surface_hist_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.particle_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    {
                        let mut pass = render_context.command_encoder().begin_compute_pass(
                            &ComputePassDescriptor {
                                label: Some("mpm_stats_water_surface_hist"),
                                timestamp_writes: None,
                            },
                        );
                        pass.set_pipeline(hist_pipeline);
                        pass.set_bind_group(0, &hist_bg, &[]);
                        pass.dispatch_workgroups(particles_wgs, 1, 1);
                    }

                    let Some(final_pipeline) = pipeline_cache
                        .get_compute_pipeline(pipelines.stats_water_surface_finalize_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_WATER_FINAL_PIPELINE_WARNED,
                            "stats_water_surface_finalize",
                            pipelines.stats_water_surface_finalize_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let final_bg = device.create_bind_group(
                        "mpm_stats_water_final_bg",
                        &pipelines.stats_water_surface_finalize_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("mpm_stats_water_surface_finalize"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_pipeline(final_pipeline);
                    pass.set_bind_group(0, &final_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }

                if stats_status.granular_repose {
                    let Some(bounds_pipeline) =
                        pipeline_cache.get_compute_pipeline(pipelines.stats_repose_bounds_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_REPOSE_BOUNDS_PIPELINE_WARNED,
                            "stats_repose_bounds",
                            pipelines.stats_repose_bounds_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let bounds_bg = device.create_bind_group(
                        "mpm_stats_repose_bounds_bg",
                        &pipelines.stats_repose_bounds_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.particle_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    {
                        let mut pass = render_context.command_encoder().begin_compute_pass(
                            &ComputePassDescriptor {
                                label: Some("mpm_stats_repose_bounds"),
                                timestamp_writes: None,
                            },
                        );
                        pass.set_pipeline(bounds_pipeline);
                        pass.set_bind_group(0, &bounds_bg, &[]);
                        pass.dispatch_workgroups(particles_wgs, 1, 1);
                    }

                    let Some(final_pipeline) = pipeline_cache
                        .get_compute_pipeline(pipelines.stats_repose_finalize_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_REPOSE_FINAL_PIPELINE_WARNED,
                            "stats_repose_finalize",
                            pipelines.stats_repose_finalize_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let final_bg = device.create_bind_group(
                        "mpm_stats_repose_final_bg",
                        &pipelines.stats_repose_finalize_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("mpm_stats_repose_finalize"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_pipeline(final_pipeline);
                    pass.set_bind_group(0, &final_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }

                if stats_status.material_interaction {
                    let Some(clear_pipeline) = pipeline_cache
                        .get_compute_pipeline(pipelines.stats_interaction_clear_cells_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_INTERACTION_CLEAR_PIPELINE_WARNED,
                            "stats_interaction_clear_cells",
                            pipelines.stats_interaction_clear_cells_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let clear_bg = device.create_bind_group(
                        "mpm_stats_interaction_clear_bg",
                        &pipelines.stats_interaction_clear_cells_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.stats_cell_flags_buf.as_entire_binding(),
                        )),
                    );
                    {
                        let mut pass = render_context.command_encoder().begin_compute_pass(
                            &ComputePassDescriptor {
                                label: Some("mpm_stats_interaction_clear_cells"),
                                timestamp_writes: None,
                            },
                        );
                        pass.set_pipeline(clear_pipeline);
                        pass.set_bind_group(0, &clear_bg, &[]);
                        pass.dispatch_workgroups(nodes_wgs, 1, 1);
                    }

                    let Some(mark_pipeline) = pipeline_cache
                        .get_compute_pipeline(pipelines.stats_interaction_mark_secondary_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_INTERACTION_MARK_PIPELINE_WARNED,
                            "stats_interaction_mark_secondary",
                            pipelines.stats_interaction_mark_secondary_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let mark_bg = device.create_bind_group(
                        "mpm_stats_interaction_mark_bg",
                        &pipelines.stats_interaction_mark_secondary_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.particle_buf.as_entire_binding(),
                            buffers.stats_cell_flags_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    {
                        let mut pass = render_context.command_encoder().begin_compute_pass(
                            &ComputePassDescriptor {
                                label: Some("mpm_stats_interaction_mark_secondary"),
                                timestamp_writes: None,
                            },
                        );
                        pass.set_pipeline(mark_pipeline);
                        pass.set_bind_group(0, &mark_bg, &[]);
                        pass.dispatch_workgroups(particles_wgs, 1, 1);
                    }

                    let Some(primary_pipeline) = pipeline_cache
                        .get_compute_pipeline(pipelines.stats_interaction_primary_contact_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_INTERACTION_PRIMARY_PIPELINE_WARNED,
                            "stats_interaction_primary_contact",
                            pipelines.stats_interaction_primary_contact_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let primary_bg = device.create_bind_group(
                        "mpm_stats_interaction_primary_bg",
                        &pipelines.stats_interaction_primary_contact_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.particle_buf.as_entire_binding(),
                            buffers.stats_cell_flags_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    {
                        let mut pass = render_context.command_encoder().begin_compute_pass(
                            &ComputePassDescriptor {
                                label: Some("mpm_stats_interaction_primary_contact"),
                                timestamp_writes: None,
                            },
                        );
                        pass.set_pipeline(primary_pipeline);
                        pass.set_bind_group(0, &primary_bg, &[]);
                        pass.dispatch_workgroups(particles_wgs, 1, 1);
                    }

                    let Some(final_pipeline) = pipeline_cache
                        .get_compute_pipeline(pipelines.stats_interaction_finalize_pipeline)
                    else {
                        warn_missing_pipeline_once(
                            &STATS_INTERACTION_FINAL_PIPELINE_WARNED,
                            "stats_interaction_finalize",
                            pipelines.stats_interaction_finalize_pipeline,
                            pipeline_cache,
                        );
                        return Ok(());
                    };
                    let final_bg = device.create_bind_group(
                        "mpm_stats_interaction_finalize_bg",
                        &pipelines.stats_interaction_finalize_layout,
                        &BindGroupEntries::sequential((
                            buffers.params_buf.as_entire_binding(),
                            buffers.stats_scalar_buf.as_entire_binding(),
                        )),
                    );
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("mpm_stats_interaction_finalize"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_pipeline(final_pipeline);
                    pass.set_bind_group(0, &final_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }

            if stats_status.grid_density {
                let Some(grid_pipeline) =
                    pipeline_cache.get_compute_pipeline(pipelines.stats_grid_density_pipeline)
                else {
                    warn_missing_pipeline_once(
                        &STATS_GRID_DENSITY_PIPELINE_WARNED,
                        "stats_grid_density",
                        pipelines.stats_grid_density_pipeline,
                        pipeline_cache,
                    );
                    return Ok(());
                };
                let grid_bg = device.create_bind_group(
                    "mpm_stats_grid_density_bg",
                    &pipelines.stats_grid_density_layout,
                    &BindGroupEntries::sequential((
                        buffers.params_buf.as_entire_binding(),
                        buffers.grid_buf.as_entire_binding(),
                        buffers.stats_scalar_buf.as_entire_binding(),
                    )),
                );
                {
                    let mut pass = render_context.command_encoder().begin_compute_pass(
                        &ComputePassDescriptor {
                            label: Some("mpm_stats_grid_density"),
                            timestamp_writes: None,
                        },
                    );
                    pass.set_pipeline(grid_pipeline);
                    pass.set_bind_group(0, &grid_bg, &[]);
                    pass.dispatch_workgroups(nodes_wgs, 1, 1);
                }

                let Some(final_pipeline) = pipeline_cache
                    .get_compute_pipeline(pipelines.stats_grid_density_finalize_pipeline)
                else {
                    warn_missing_pipeline_once(
                        &STATS_GRID_DENSITY_FINAL_PIPELINE_WARNED,
                        "stats_grid_density_finalize",
                        pipelines.stats_grid_density_finalize_pipeline,
                        pipeline_cache,
                    );
                    return Ok(());
                };
                let final_bg = device.create_bind_group(
                    "mpm_stats_grid_density_finalize_bg",
                    &pipelines.stats_grid_density_finalize_layout,
                    &BindGroupEntries::sequential((buffers.stats_scalar_buf.as_entire_binding(),)),
                );
                let mut pass =
                    render_context
                        .command_encoder()
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("mpm_stats_grid_density_finalize"),
                            timestamp_writes: None,
                        });
                pass.set_pipeline(final_pipeline);
                pass.set_bind_group(0, &final_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        if control.readback_enabled {
            let interval = control.readback_interval_frames.max(1) as u64;
            let frame =
                GPU_READBACK_FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if frame % interval == 0 {
                let can_copy_particles = world
                    .get_resource::<GpuReadbackState>()
                    .map(|state| !state.mapped)
                    .unwrap_or(true);
                let can_copy_stats = world
                    .get_resource::<GpuStatisticsReadbackState>()
                    .map(|state| !state.mapped)
                    .unwrap_or(true);
                let can_copy_movers = world
                    .get_resource::<GpuMoverReadbackState>()
                    .map(|state| !state.mapped)
                    .unwrap_or(true);

                if has_particles && can_copy_particles {
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
                if stats_enabled && can_copy_stats {
                    render_context.command_encoder().copy_buffer_to_buffer(
                        &buffers.stats_scalar_buf,
                        0,
                        &buffers.stats_scalar_readback_buf,
                        0,
                        size_of::<GpuStatisticsScalars>() as u64,
                    );
                }
                if has_particles && can_copy_movers {
                    let mover_record_bytes =
                        super::gpu_resources::MAX_MOVER_RECORDS as u64
                            * size_of::<super::buffers::GpuMoverRecord>() as u64;
                    render_context.command_encoder().copy_buffer_to_buffer(
                        &buffers.mover_count_buf,
                        0,
                        &buffers.mover_readback_buf,
                        0,
                        size_of::<u32>() as u64,
                    );
                    render_context.command_encoder().copy_buffer_to_buffer(
                        &buffers.mover_buf,
                        0,
                        &buffers.mover_readback_buf,
                        size_of::<u32>() as u64,
                        mover_record_bytes,
                    );
                }
            }
        }

        Ok(())
    }
}
