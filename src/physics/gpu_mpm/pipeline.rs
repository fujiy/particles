// GPU compute pipeline definitions for MLS-MPM.
//
// Each pass (clear, p2g, grid_update, g2p) has its own pipeline.

use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedComputePipelineId,
    ComputePipelineDescriptor, PipelineCache, ShaderStages, binding_types,
};
use bevy::render::renderer::RenderDevice;

use super::shaders::MpmShaders;

// ---------------------------------------------------------------------------
// Pipeline resource
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct MpmComputePipelines {
    pub active_tile_clear_layout: BindGroupLayout,
    pub active_tile_clear_pipeline: CachedComputePipelineId,
    pub active_tile_mark_layout: BindGroupLayout,
    pub active_tile_mark_pipeline: CachedComputePipelineId,
    pub active_tile_compact_layout: BindGroupLayout,
    pub active_tile_compact_pipeline: CachedComputePipelineId,
    pub active_tile_dispatch_layout: BindGroupLayout,
    pub active_tile_dispatch_pipeline: CachedComputePipelineId,

    pub clear_layout: BindGroupLayout,
    pub clear_pipeline: CachedComputePipelineId,

    pub p2g_layout: BindGroupLayout,
    pub p2g_pipeline: CachedComputePipelineId,

    pub grid_update_layout: BindGroupLayout,
    pub grid_update_pipeline: CachedComputePipelineId,

    pub g2p_layout: BindGroupLayout,
    pub g2p_pipeline: CachedComputePipelineId,
    pub extract_movers_layout: BindGroupLayout,
    pub extract_movers_pipeline: CachedComputePipelineId,
    pub extract_chunk_events_layout: BindGroupLayout,
    pub extract_chunk_events_pipeline: CachedComputePipelineId,
    pub apply_mover_results_layout: BindGroupLayout,
    pub apply_mover_results_pipeline: CachedComputePipelineId,
    pub chunk_meta_clear_layout: BindGroupLayout,
    pub chunk_meta_clear_pipeline: CachedComputePipelineId,
    pub chunk_meta_accumulate_layout: BindGroupLayout,
    pub chunk_meta_accumulate_pipeline: CachedComputePipelineId,
    pub chunk_meta_finalize_layout: BindGroupLayout,
    pub chunk_meta_finalize_pipeline: CachedComputePipelineId,
    pub terrain_sdf_update_layout: BindGroupLayout,
    pub terrain_sdf_update_pipeline: CachedComputePipelineId,

    pub stats_clear_layout: BindGroupLayout,
    pub stats_clear_pipeline: CachedComputePipelineId,

    pub stats_total_layout: BindGroupLayout,
    pub stats_total_pipeline: CachedComputePipelineId,

    pub stats_phase_layout: BindGroupLayout,
    pub stats_phase_pipeline: CachedComputePipelineId,

    pub stats_max_speed_layout: BindGroupLayout,
    pub stats_max_speed_pipeline: CachedComputePipelineId,

    pub stats_penetration_tracking_layout: BindGroupLayout,
    pub stats_penetration_tracking_pipeline: CachedComputePipelineId,

    pub stats_water_surface_hist_layout: BindGroupLayout,
    pub stats_water_surface_hist_pipeline: CachedComputePipelineId,

    pub stats_water_surface_finalize_layout: BindGroupLayout,
    pub stats_water_surface_finalize_pipeline: CachedComputePipelineId,

    pub stats_repose_bounds_layout: BindGroupLayout,
    pub stats_repose_bounds_pipeline: CachedComputePipelineId,

    pub stats_repose_finalize_layout: BindGroupLayout,
    pub stats_repose_finalize_pipeline: CachedComputePipelineId,

    pub stats_interaction_clear_cells_layout: BindGroupLayout,
    pub stats_interaction_clear_cells_pipeline: CachedComputePipelineId,

    pub stats_interaction_mark_secondary_layout: BindGroupLayout,
    pub stats_interaction_mark_secondary_pipeline: CachedComputePipelineId,

    pub stats_interaction_primary_contact_layout: BindGroupLayout,
    pub stats_interaction_primary_contact_pipeline: CachedComputePipelineId,

    pub stats_interaction_finalize_layout: BindGroupLayout,
    pub stats_interaction_finalize_pipeline: CachedComputePipelineId,

    pub stats_grid_density_layout: BindGroupLayout,
    pub stats_grid_density_pipeline: CachedComputePipelineId,

    pub stats_grid_density_finalize_layout: BindGroupLayout,
    pub stats_grid_density_finalize_pipeline: CachedComputePipelineId,
}

impl FromWorld for MpmComputePipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let shaders = world.resource::<MpmShaders>();

        // active_tile_clear: 0=params(uniform), 1=chunk_meta(storage rw)
        let active_tile_clear_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let active_tile_clear_layout = render_device
            .create_bind_group_layout("mpm_active_tile_clear_layout", &*active_tile_clear_entries);
        let active_tile_clear_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_active_tile_clear".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_active_tile_clear_layout",
                    &*active_tile_clear_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.active_tiles.clone(),
                shader_defs: vec![],
                entry_point: Some("clear_active_tiles".into()),
                zero_initialize_workgroup_memory: false,
            });

        // active_tile_mark: 0=params, 1=particles(read), 2=chunk_meta(rw)
        let active_tile_mark_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let active_tile_mark_layout = render_device
            .create_bind_group_layout("mpm_active_tile_mark_layout", &*active_tile_mark_entries);
        let active_tile_mark_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_active_tile_mark".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_active_tile_mark_layout",
                    &*active_tile_mark_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.active_tiles.clone(),
                shader_defs: vec![],
                entry_point: Some("mark_active_tiles".into()),
                zero_initialize_workgroup_memory: false,
            });

        // active_tile_compact: 0=params, 1=chunk_meta(read), 2=tile_count(rw), 3=tile_list(rw)
        let active_tile_compact_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let active_tile_compact_layout = render_device.create_bind_group_layout(
            "mpm_active_tile_compact_layout",
            &*active_tile_compact_entries,
        );
        let active_tile_compact_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_active_tile_compact".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_active_tile_compact_layout",
                    &*active_tile_compact_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.active_tiles.clone(),
                shader_defs: vec![],
                entry_point: Some("compact_active_tiles".into()),
                zero_initialize_workgroup_memory: false,
            });

        // active_tile_dispatch: 0=tile_count(read), 1=dispatch_args(rw)
        let active_tile_dispatch_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let active_tile_dispatch_layout = render_device.create_bind_group_layout(
            "mpm_active_tile_dispatch_layout",
            &*active_tile_dispatch_entries,
        );
        let active_tile_dispatch_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_active_tile_dispatch".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_active_tile_dispatch_layout",
                    &*active_tile_dispatch_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.active_tiles.clone(),
                shader_defs: vec![],
                entry_point: Some("finalize_active_tile_dispatch".into()),
                zero_initialize_workgroup_memory: false,
            });

        // clear: 0=params(uniform), 1=grid(storage rw), 2=active_tile_list(read)
        let clear_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
            ),
        );
        let clear_layout =
            render_device.create_bind_group_layout("mpm_clear_layout", &*clear_entries);
        let clear_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("mpm_clear".into()),
            layout: vec![BindGroupLayoutDescriptor::new(
                "mpm_clear_layout",
                &*clear_entries,
            )],
            push_constant_ranges: vec![],
            shader: shaders.clear.clone(),
            shader_defs: vec![],
            entry_point: Some("clear_grid".into()),
            zero_initialize_workgroup_memory: false,
        });

        // p2g: 0=params, 1=particles(read), 2=grid_atomic(rw), 3=chunk_meta(read)
        let p2g_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
            ),
        );
        let p2g_layout = render_device.create_bind_group_layout("mpm_p2g_layout", &*p2g_entries);
        let p2g_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("mpm_p2g".into()),
            layout: vec![BindGroupLayoutDescriptor::new(
                "mpm_p2g_layout",
                &*p2g_entries,
            )],
            push_constant_ranges: vec![],
            shader: shaders.p2g.clone(),
            shader_defs: vec![],
            entry_point: Some("p2g".into()),
            zero_initialize_workgroup_memory: false,
        });

        // grid_update:
        // 0=params, 1=grid(rw), 2=terrain_sdf(read), 3=terrain_normal(read),
        // 4=chunk_meta(read), 5=active_tile_list(read)
        let gu_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
            ),
        );
        let grid_update_layout =
            render_device.create_bind_group_layout("mpm_grid_update_layout", &*gu_entries);
        let grid_update_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_grid_update".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_grid_update_layout",
                    &*gu_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.grid_update.clone(),
                shader_defs: vec![],
                entry_point: Some("grid_update".into()),
                zero_initialize_workgroup_memory: false,
            });

        // g2p: 0=params, 1=particles(rw), 2=grid(read), 3=chunk_meta(read)
        let g2p_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
            ),
        );
        let g2p_layout = render_device.create_bind_group_layout("mpm_g2p_layout", &*g2p_entries);
        let g2p_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("mpm_g2p".into()),
            layout: vec![BindGroupLayoutDescriptor::new(
                "mpm_g2p_layout",
                &*g2p_entries,
            )],
            push_constant_ranges: vec![],
            shader: shaders.g2p.clone(),
            shader_defs: vec![],
            entry_point: Some("g2p".into()),
            zero_initialize_workgroup_memory: false,
        });

        // extract_movers: 0=params, 1=particles(rw), 2=chunk_meta(read), 3=mover_count(rw), 4=movers(rw)
        let extract_movers_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let extract_movers_layout = render_device
            .create_bind_group_layout("mpm_extract_movers_layout", &*extract_movers_entries);
        let extract_movers_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_extract_movers".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_extract_movers_layout",
                    &*extract_movers_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.extract_movers.clone(),
                shader_defs: vec![],
                entry_point: Some("extract_movers".into()),
                zero_initialize_workgroup_memory: false,
            });

        // extract_chunk_events: 0=params, 1=chunk_meta(read), 2=event_count(rw), 3=events(rw)
        let extract_chunk_events_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let extract_chunk_events_layout = render_device.create_bind_group_layout(
            "mpm_extract_chunk_events_layout",
            &*extract_chunk_events_entries,
        );
        let extract_chunk_events_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_extract_chunk_events".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_extract_chunk_events_layout",
                    &*extract_chunk_events_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.extract_chunk_events.clone(),
                shader_defs: vec![],
                entry_point: Some("extract_chunk_events".into()),
                zero_initialize_workgroup_memory: false,
            });

        // apply_mover_results: 0=params, 1=particles(rw), 2=result_count(read), 3=results(read)
        let apply_mover_results_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
            ),
        );
        let apply_mover_results_layout = render_device.create_bind_group_layout(
            "mpm_apply_mover_results_layout",
            &*apply_mover_results_entries,
        );
        let apply_mover_results_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_apply_mover_results".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_apply_mover_results_layout",
                    &*apply_mover_results_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.apply_mover_results.clone(),
                shader_defs: vec![],
                entry_point: Some("apply_mover_results".into()),
                zero_initialize_workgroup_memory: false,
            });

        // chunk_meta_{clear,accumulate,finalize}: 0=params, 1=particles(read), 2=chunk_meta(rw)
        let chunk_meta_clear_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let chunk_meta_clear_layout = render_device
            .create_bind_group_layout("mpm_chunk_meta_clear_layout", &*chunk_meta_clear_entries);
        let chunk_meta_clear_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_chunk_meta_clear".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_chunk_meta_clear_layout",
                    &*chunk_meta_clear_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.chunk_meta_update.clone(),
                shader_defs: vec![],
                entry_point: Some("clear_chunk_counts".into()),
                zero_initialize_workgroup_memory: false,
            });

        // chunk_meta_accumulate: 0=params, 1=particles(read), 2=chunk_meta(rw)
        let chunk_meta_accumulate_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let chunk_meta_accumulate_layout = render_device.create_bind_group_layout(
            "mpm_chunk_meta_accumulate_layout",
            &*chunk_meta_accumulate_entries,
        );
        let chunk_meta_accumulate_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_chunk_meta_accumulate".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_chunk_meta_accumulate_layout",
                    &*chunk_meta_accumulate_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.chunk_meta_update.clone(),
                shader_defs: vec![],
                entry_point: Some("accumulate_chunk_counts".into()),
                zero_initialize_workgroup_memory: false,
            });

        // chunk_meta_finalize: 0=params, 1=particles(read), 2=chunk_meta(rw)
        let chunk_meta_finalize_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let chunk_meta_finalize_layout = render_device.create_bind_group_layout(
            "mpm_chunk_meta_finalize_layout",
            &*chunk_meta_finalize_entries,
        );
        let chunk_meta_finalize_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_chunk_meta_finalize".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_chunk_meta_finalize_layout",
                    &*chunk_meta_finalize_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.chunk_meta_update.clone(),
                shader_defs: vec![],
                entry_point: Some("finalize_chunk_flags".into()),
                zero_initialize_workgroup_memory: false,
            });

        // terrain_sdf_update:
        // 0=params, 1=chunk_meta(read), 2=terrain_cell_solid(read),
        // 3=terrain_sdf(rw), 4=terrain_normal(rw), 5=update_count(read), 6=update_slots(read)
        let terrain_sdf_update_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
            ),
        );
        let terrain_sdf_update_layout = render_device.create_bind_group_layout(
            "mpm_terrain_sdf_update_layout",
            &*terrain_sdf_update_entries,
        );
        let terrain_sdf_update_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_terrain_sdf_update".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_terrain_sdf_update_layout",
                    &*terrain_sdf_update_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.terrain_sdf_update.clone(),
                shader_defs: vec![],
                entry_point: Some("update_terrain_sdf_slots".into()),
                zero_initialize_workgroup_memory: false,
            });

        // stats_clear: 0=stats_scalars(rw)
        let stats_clear_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (binding_types::storage_buffer_sized(false, None),),
        );
        let stats_clear_layout =
            render_device.create_bind_group_layout("mpm_stats_clear_layout", &*stats_clear_entries);
        let stats_clear_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_clear".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_clear_layout",
                    &*stats_clear_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_clear.clone(),
                shader_defs: vec![],
                entry_point: Some("clear_stats_scalars".into()),
                zero_initialize_workgroup_memory: false,
            });

        // Shared layout for particle-stat passes: 0=params, 1=particles(read), 2=stats_scalars(rw)
        let stats_particle_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let stats_total_layout = render_device
            .create_bind_group_layout("mpm_stats_total_layout", &*stats_particle_entries);
        let stats_total_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_total_particles".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_total_layout",
                    &*stats_particle_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_total_particles.clone(),
                shader_defs: vec![],
                entry_point: Some("count_total_particles".into()),
                zero_initialize_workgroup_memory: false,
            });

        let stats_phase_layout = render_device
            .create_bind_group_layout("mpm_stats_phase_layout", &*stats_particle_entries);
        let stats_phase_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_phase_counts".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_phase_layout",
                    &*stats_particle_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_phase_counts.clone(),
                shader_defs: vec![],
                entry_point: Some("count_phase_particles".into()),
                zero_initialize_workgroup_memory: false,
            });

        let stats_max_speed_layout = render_device
            .create_bind_group_layout("mpm_stats_max_speed_layout", &*stats_particle_entries);
        let stats_max_speed_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_max_speed".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_max_speed_layout",
                    &*stats_particle_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_max_speed.clone(),
                shader_defs: vec![],
                entry_point: Some("compute_max_speed".into()),
                zero_initialize_workgroup_memory: false,
            });

        // 0=params, 1=particles(read), 2=terrain_sdf(read), 3=stats_scalars(rw)
        let stats_penetration_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let stats_penetration_tracking_layout = render_device.create_bind_group_layout(
            "mpm_stats_penetration_tracking_layout",
            &*stats_penetration_entries,
        );
        let stats_penetration_tracking_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_penetration_tracking".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_penetration_tracking_layout",
                    &*stats_penetration_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_penetration_tracking.clone(),
                shader_defs: vec![],
                entry_point: Some("compute_penetration_tracking".into()),
                zero_initialize_workgroup_memory: false,
            });

        let stats_water_surface_hist_layout = render_device.create_bind_group_layout(
            "mpm_stats_water_surface_hist_layout",
            &*stats_particle_entries,
        );
        let stats_water_surface_hist_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_water_surface_hist".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_water_surface_hist_layout",
                    &*stats_particle_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_water_surface_histogram.clone(),
                shader_defs: vec![],
                entry_point: Some("build_water_surface_histogram".into()),
                zero_initialize_workgroup_memory: false,
            });

        // 0=params, 1=stats_scalars(rw)
        let stats_finalize_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let stats_water_surface_finalize_layout = render_device.create_bind_group_layout(
            "mpm_stats_water_surface_finalize_layout",
            &*stats_finalize_entries,
        );
        let stats_water_surface_finalize_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_water_surface_finalize".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_water_surface_finalize_layout",
                    &*stats_finalize_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_water_surface_finalize.clone(),
                shader_defs: vec![],
                entry_point: Some("finalize_water_surface_p95".into()),
                zero_initialize_workgroup_memory: false,
            });

        let stats_repose_bounds_layout = render_device
            .create_bind_group_layout("mpm_stats_repose_bounds_layout", &*stats_particle_entries);
        let stats_repose_bounds_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_repose_bounds".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_repose_bounds_layout",
                    &*stats_particle_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_granular_repose_bounds.clone(),
                shader_defs: vec![],
                entry_point: Some("collect_granular_repose_bounds".into()),
                zero_initialize_workgroup_memory: false,
            });

        let stats_repose_finalize_layout = render_device
            .create_bind_group_layout("mpm_stats_repose_finalize_layout", &*stats_finalize_entries);
        let stats_repose_finalize_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_repose_finalize".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_repose_finalize_layout",
                    &*stats_finalize_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_granular_repose_finalize.clone(),
                shader_defs: vec![],
                entry_point: Some("finalize_granular_repose".into()),
                zero_initialize_workgroup_memory: false,
            });

        // 0=params, 1=cell_flags(rw)
        let interaction_clear_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let stats_interaction_clear_cells_layout = render_device.create_bind_group_layout(
            "mpm_stats_interaction_clear_cells_layout",
            &*interaction_clear_entries,
        );
        let stats_interaction_clear_cells_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_interaction_clear_cells".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_interaction_clear_cells_layout",
                    &*interaction_clear_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_interaction_clear_cells.clone(),
                shader_defs: vec![],
                entry_point: Some("clear_interaction_cells".into()),
                zero_initialize_workgroup_memory: false,
            });

        // 0=params, 1=particles(read), 2=cell_flags, 3=stats_scalars
        let interaction_particle_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let stats_interaction_mark_secondary_layout = render_device.create_bind_group_layout(
            "mpm_stats_interaction_mark_secondary_layout",
            &*interaction_particle_entries,
        );
        let stats_interaction_mark_secondary_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_interaction_mark_secondary".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_interaction_mark_secondary_layout",
                    &*interaction_particle_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_interaction_mark_secondary.clone(),
                shader_defs: vec![],
                entry_point: Some("mark_interaction_secondary".into()),
                zero_initialize_workgroup_memory: false,
            });

        let stats_interaction_primary_contact_layout = render_device.create_bind_group_layout(
            "mpm_stats_interaction_primary_contact_layout",
            &*interaction_particle_entries,
        );
        let stats_interaction_primary_contact_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_interaction_primary_contact".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_interaction_primary_contact_layout",
                    &*interaction_particle_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_interaction_primary_contact.clone(),
                shader_defs: vec![],
                entry_point: Some("compute_interaction_primary_contact".into()),
                zero_initialize_workgroup_memory: false,
            });

        let stats_interaction_finalize_layout = render_device.create_bind_group_layout(
            "mpm_stats_interaction_finalize_layout",
            &*stats_finalize_entries,
        );
        let stats_interaction_finalize_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_interaction_finalize".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_interaction_finalize_layout",
                    &*stats_finalize_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_interaction_finalize.clone(),
                shader_defs: vec![],
                entry_point: Some("finalize_interaction".into()),
                zero_initialize_workgroup_memory: false,
            });

        // 0=params, 1=grid(read), 2=stats_scalars(rw)
        let stats_grid_density_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_read_only_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
            ),
        );
        let stats_grid_density_layout = render_device.create_bind_group_layout(
            "mpm_stats_grid_density_layout",
            &*stats_grid_density_entries,
        );
        let stats_grid_density_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_grid_density".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_grid_density_layout",
                    &*stats_grid_density_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_grid_density.clone(),
                shader_defs: vec![],
                entry_point: Some("collect_grid_density".into()),
                zero_initialize_workgroup_memory: false,
            });

        // 0=stats_scalars(rw)
        let stats_scalar_only_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (binding_types::storage_buffer_sized(false, None),),
        );
        let stats_grid_density_finalize_layout = render_device.create_bind_group_layout(
            "mpm_stats_grid_density_finalize_layout",
            &*stats_scalar_only_entries,
        );
        let stats_grid_density_finalize_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("mpm_stats_grid_density_finalize".into()),
                layout: vec![BindGroupLayoutDescriptor::new(
                    "mpm_stats_grid_density_finalize_layout",
                    &*stats_scalar_only_entries,
                )],
                push_constant_ranges: vec![],
                shader: shaders.stats_grid_density_finalize.clone(),
                shader_defs: vec![],
                entry_point: Some("finalize_grid_density".into()),
                zero_initialize_workgroup_memory: false,
            });

        Self {
            active_tile_clear_layout,
            active_tile_clear_pipeline,
            active_tile_mark_layout,
            active_tile_mark_pipeline,
            active_tile_compact_layout,
            active_tile_compact_pipeline,
            active_tile_dispatch_layout,
            active_tile_dispatch_pipeline,
            clear_layout,
            clear_pipeline,
            p2g_layout,
            p2g_pipeline,
            grid_update_layout,
            grid_update_pipeline,
            g2p_layout,
            g2p_pipeline,
            extract_movers_layout,
            extract_movers_pipeline,
            extract_chunk_events_layout,
            extract_chunk_events_pipeline,
            apply_mover_results_layout,
            apply_mover_results_pipeline,
            chunk_meta_clear_layout,
            chunk_meta_clear_pipeline,
            chunk_meta_accumulate_layout,
            chunk_meta_accumulate_pipeline,
            chunk_meta_finalize_layout,
            chunk_meta_finalize_pipeline,
            terrain_sdf_update_layout,
            terrain_sdf_update_pipeline,
            stats_clear_layout,
            stats_clear_pipeline,
            stats_total_layout,
            stats_total_pipeline,
            stats_phase_layout,
            stats_phase_pipeline,
            stats_max_speed_layout,
            stats_max_speed_pipeline,
            stats_penetration_tracking_layout,
            stats_penetration_tracking_pipeline,
            stats_water_surface_hist_layout,
            stats_water_surface_hist_pipeline,
            stats_water_surface_finalize_layout,
            stats_water_surface_finalize_pipeline,
            stats_repose_bounds_layout,
            stats_repose_bounds_pipeline,
            stats_repose_finalize_layout,
            stats_repose_finalize_pipeline,
            stats_interaction_clear_cells_layout,
            stats_interaction_clear_cells_pipeline,
            stats_interaction_mark_secondary_layout,
            stats_interaction_mark_secondary_pipeline,
            stats_interaction_primary_contact_layout,
            stats_interaction_primary_contact_pipeline,
            stats_interaction_finalize_layout,
            stats_interaction_finalize_pipeline,
            stats_grid_density_layout,
            stats_grid_density_pipeline,
            stats_grid_density_finalize_layout,
            stats_grid_density_finalize_pipeline,
        }
    }
}
