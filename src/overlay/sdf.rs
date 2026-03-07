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
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use std::mem::size_of;

use super::SdfOverlayState;
use crate::physics::gpu_mpm::buffers::GpuMpmParams;
use crate::physics::gpu_mpm::gpu_resources::MpmGpuBuffers;

const SDF_OVERLAY_GPU_SHADER_PATH: &str = "shaders/overlay/sdf_overlay_gpu.wgsl";

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct SdfOverlayGpuLabel;

#[derive(Default)]
pub(super) struct SdfOverlayGpuNode;

#[derive(Resource)]
pub(super) struct SdfOverlayGpuPipeline {
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
    pub fallback_params_buf: Buffer,
    pub fallback_sdf_buf: Buffer,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct SdfOverlayPipelineKey {
    target_format: TextureFormat,
    sample_count: u32,
}

impl SpecializedRenderPipeline for SdfOverlayGpuPipeline {
    type Key = SdfOverlayPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("sdf_overlay_gpu_pipeline".into()),
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
pub(super) struct ViewSdfOverlayGpuPipeline {
    pub id: CachedRenderPipelineId,
}

pub(super) fn init_sdf_overlay_gpu_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "sdf_overlay_gpu_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer_sized(
                    false,
                    core::num::NonZeroU64::new(size_of::<GpuMpmParams>() as u64),
                ),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );

    let fallback_params_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("sdf_overlay_fallback_params"),
        size: size_of::<GpuMpmParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fallback_sdf_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("sdf_overlay_fallback_sdf"),
        size: size_of::<f32>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(SdfOverlayGpuPipeline {
        bind_group_layout,
        shader: asset_server.load(SDF_OVERLAY_GPU_SHADER_PATH),
        fallback_params_buf,
        fallback_sdf_buf,
    });
}

pub(super) fn prepare_sdf_overlay_gpu_pipeline(
    mut commands: Commands,
    pipeline: Res<SdfOverlayGpuPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SdfOverlayGpuPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(Entity, &ViewTarget, Option<&Msaa>)>,
) {
    for (entity, view_target, msaa) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SdfOverlayPipelineKey {
                target_format: view_target.main_texture_format(),
                sample_count: msaa.map_or(1, Msaa::samples),
            },
        );
        commands
            .entity(entity)
            .insert(ViewSdfOverlayGpuPipeline { id });
    }
}

impl ViewNode for SdfOverlayGpuNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static ViewSdfOverlayGpuPipeline,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, view_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(overlay_state) = world.get_resource::<SdfOverlayState>() else {
            return Ok(());
        };
        if !overlay_state.enabled {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(view_pipeline.id) else {
            return Ok(());
        };
        let overlay_pipeline = world.resource::<SdfOverlayGpuPipeline>();

        let (params_binding, sdf_binding, chunk_count) =
            if let Some(buffers) = world.get_resource::<MpmGpuBuffers>() {
                (
                    buffers.params_buf.as_entire_binding(),
                    buffers.terrain_sdf_buf.as_entire_binding(),
                    buffers.active_chunk_count,
                )
            } else {
                (
                    overlay_pipeline.fallback_params_buf.as_entire_binding(),
                    overlay_pipeline.fallback_sdf_buf.as_entire_binding(),
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
            "sdf_overlay_gpu_bind_group",
            &pipeline_cache.get_bind_group_layout(&overlay_pipeline.bind_group_layout),
            &BindGroupEntries::sequential((view_binding, params_binding, sdf_binding)),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("sdf_overlay_gpu_pass"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..6, 0..1);
        Ok(())
    }
}
