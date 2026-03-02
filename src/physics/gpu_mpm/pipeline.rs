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
    pub clear_layout: BindGroupLayout,
    pub clear_pipeline: CachedComputePipelineId,

    pub p2g_layout: BindGroupLayout,
    pub p2g_pipeline: CachedComputePipelineId,

    pub grid_update_layout: BindGroupLayout,
    pub grid_update_pipeline: CachedComputePipelineId,

    pub g2p_layout: BindGroupLayout,
    pub g2p_pipeline: CachedComputePipelineId,
}

impl FromWorld for MpmComputePipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let shaders = world.resource::<MpmShaders>();

        // clear: 0=params(uniform), 1=grid(storage rw)
        let clear_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
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

        // p2g: 0=params, 1=particles(read), 2=grid_atomic(rw)
        // grid is written only via atomic u32 view; no separate GpuGridNode binding needed.
        let p2g_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
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

        // grid_update: 0=params, 1=grid(rw), 2=terrain_sdf(read), 3=terrain_normal(read)
        let gu_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
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

        // g2p: 0=params, 1=particles(rw), 2=grid(read)
        let g2p_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                binding_types::uniform_buffer_sized(false, None),
                binding_types::storage_buffer_sized(false, None),
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

        Self {
            clear_layout,
            clear_pipeline,
            p2g_layout,
            p2g_pipeline,
            grid_update_layout,
            grid_update_pipeline,
            g2p_layout,
            g2p_pipeline,
        }
    }
}
