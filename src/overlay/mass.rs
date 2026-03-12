use bevy::asset::AssetServer;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendComponent,
    BlendFactor, BlendOperation, BlendState, Buffer, BufferDescriptor, BufferUsages,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, MultisampleState,
    PipelineCache, PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat, VertexState,
    binding_types::{storage_buffer_read_only_sized, uniform_buffer, uniform_buffer_sized},
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;

use super::MassOverlayState;
use crate::params::ActiveOverlayParams;
use crate::physics::gpu_mpm::buffers::{GpuChunkMeta, GpuGridNode, GpuMpmParams};
use crate::physics::gpu_mpm::gpu_resources::MpmGpuBuffers;
use crate::physics::profiler::{
    begin_gpu_pass_query, cpu_profile_span, end_gpu_pass_query, resolve_gpu_profiler_queries,
};

const MASS_OVERLAY_GPU_SHADER_PATH: &str = "shaders/overlay/mass_overlay_gpu.wgsl";

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuMassOverlayStyle {
    low: [f32; 4],
    mid: [f32; 4],
    high: [f32; 4],
    opacity: f32,
    max_ref_cell_mass_scale: f32,
    _pad: [f32; 2],
}

impl Default for GpuMassOverlayStyle {
    fn default() -> Self {
        Self {
            low: [0.15, 0.44, 0.90, 0.18],
            mid: [0.99, 0.78, 0.18, 0.42],
            high: [0.88, 0.21, 0.10, 0.78],
            opacity: 0.72,
            max_ref_cell_mass_scale: 1.0,
            _pad: [0.0; 2],
        }
    }
}

impl GpuMassOverlayStyle {
    fn from_active_overlay(params: &ActiveOverlayParams) -> Self {
        let colors = &params.0.colors;
        Self {
            low: [
                colors.mass_overlay_low.r,
                colors.mass_overlay_low.g,
                colors.mass_overlay_low.b,
                colors.mass_overlay_low.a,
            ],
            mid: [
                colors.mass_overlay_mid.r,
                colors.mass_overlay_mid.g,
                colors.mass_overlay_mid.b,
                colors.mass_overlay_mid.a,
            ],
            high: [
                colors.mass_overlay_high.r,
                colors.mass_overlay_high.g,
                colors.mass_overlay_high.b,
                colors.mass_overlay_high.a,
            ],
            opacity: params.0.mass.opacity,
            max_ref_cell_mass_scale: params.0.mass.max_ref_cell_mass_scale,
            _pad: [0.0; 2],
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct MassOverlayGpuLabel;

#[derive(Default)]
pub(super) struct MassOverlayGpuNode;

#[derive(Resource)]
pub(super) struct MassOverlayGpuPipeline {
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
    pub fallback_params_buf: Buffer,
    pub fallback_grid_buf: Buffer,
    pub fallback_chunk_meta_buf: Buffer,
    pub fallback_style_buf: Buffer,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct MassOverlayPipelineKey {
    target_format: TextureFormat,
    sample_count: u32,
}

impl SpecializedRenderPipeline for MassOverlayGpuPipeline {
    type Key = MassOverlayPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("mass_overlay_gpu_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some("vs_main".into()),
                shader_defs: vec![],
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some("fs_main".into()),
                shader_defs: vec![],
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.sample_count,
                ..Default::default()
            },
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        }
    }
}

#[derive(Component)]
pub(super) struct ViewMassOverlayGpuPipeline {
    pub id: CachedRenderPipelineId,
}

pub(super) fn init_mass_overlay_gpu_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "mass_overlay_gpu_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer_sized(
                    false,
                    core::num::NonZeroU64::new(size_of::<GpuMpmParams>() as u64),
                ),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                uniform_buffer_sized(
                    false,
                    core::num::NonZeroU64::new(size_of::<GpuMassOverlayStyle>() as u64),
                ),
            ),
        ),
    );

    let fallback_params_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("mass_overlay_fallback_params"),
        size: size_of::<GpuMpmParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fallback_grid_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("mass_overlay_fallback_grid"),
        size: size_of::<GpuGridNode>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fallback_chunk_meta_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("mass_overlay_fallback_chunk_meta"),
        size: size_of::<GpuChunkMeta>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fallback_style_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("mass_overlay_fallback_style"),
        size: size_of::<GpuMassOverlayStyle>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(MassOverlayGpuPipeline {
        bind_group_layout,
        shader: asset_server.load(MASS_OVERLAY_GPU_SHADER_PATH),
        fallback_params_buf,
        fallback_grid_buf,
        fallback_chunk_meta_buf,
        fallback_style_buf,
    });
}

pub(super) fn prepare_mass_overlay_gpu_pipeline(
    mut commands: Commands,
    pipeline: Res<MassOverlayGpuPipeline>,
    queue: Res<RenderQueue>,
    overlay_params: Option<Res<ActiveOverlayParams>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<MassOverlayGpuPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(Entity, &ViewTarget, Option<&Msaa>)>,
) {
    let style = overlay_params
        .as_deref()
        .map(GpuMassOverlayStyle::from_active_overlay)
        .unwrap_or_default();
    queue.write_buffer(&pipeline.fallback_style_buf, 0, bytemuck::bytes_of(&style));

    for (entity, view_target, msaa) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            MassOverlayPipelineKey {
                target_format: view_target.main_texture_format(),
                sample_count: msaa.map_or(1, Msaa::samples),
            },
        );
        commands.entity(entity).insert(ViewMassOverlayGpuPipeline { id });
    }
}

impl ViewNode for MassOverlayGpuNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static ViewMassOverlayGpuPipeline,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, view_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let _profile_span = cpu_profile_span("overlay", "mass_node").entered();
        let Some(overlay_state) = world.get_resource::<MassOverlayState>() else {
            return Ok(());
        };
        if !overlay_state.enabled {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(view_pipeline.id) else {
            return Ok(());
        };
        let overlay_pipeline = world.resource::<MassOverlayGpuPipeline>();

        let (params_binding, grid_binding, chunk_meta_binding, chunk_count) =
            if let Some(buffers) = world.get_resource::<MpmGpuBuffers>() {
                (
                    buffers.params_buf.as_entire_binding(),
                    buffers.grid_buf.as_entire_binding(),
                    buffers.chunk_meta_buf.as_entire_binding(),
                    buffers.active_chunk_count,
                )
            } else {
                (
                    overlay_pipeline.fallback_params_buf.as_entire_binding(),
                    overlay_pipeline.fallback_grid_buf.as_entire_binding(),
                    overlay_pipeline.fallback_chunk_meta_buf.as_entire_binding(),
                    0,
                )
            };
        if chunk_count == 0 {
            return Ok(());
        }

        let view_uniforms_resource = world.resource::<ViewUniforms>();
        let Some(view_binding) = view_uniforms_resource.uniforms.binding() else {
            return Ok(());
        };
        let bind_group = render_context.render_device().create_bind_group(
            "mass_overlay_gpu_bind_group",
            &pipeline_cache.get_bind_group_layout(&overlay_pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                view_binding,
                params_binding,
                grid_binding,
                chunk_meta_binding,
                overlay_pipeline.fallback_style_buf.as_entire_binding(),
            )),
        );

        let profile_query = begin_gpu_pass_query(
            world,
            "overlay",
            "mass",
            render_context.command_encoder(),
        );
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("mass_overlay_gpu_pass"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: profile_query
                .as_ref()
                .and_then(|query| query.render_pass_timestamp_writes()),
            occlusion_query_set: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..6, 0..1);
        drop(render_pass);
        end_gpu_pass_query(world, render_context.command_encoder(), profile_query);
        resolve_gpu_profiler_queries(world, render_context.command_encoder());
        Ok(())
    }
}
