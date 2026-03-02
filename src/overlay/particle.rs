use bevy::asset::AssetServer;
use bevy::ecs::query::QueryItem;
use bevy::log::warn;
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

use super::ParticleOverlayState;
use crate::physics::gpu_mpm::buffers::{GpuMpmParams, GpuParticle};
use crate::physics::gpu_mpm::gpu_resources::MpmGpuBuffers;

const PARTICLE_OVERLAY_SHADER_PATH: &str = "shaders/overlay/particle_overlay_gpu.wgsl";

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct ParticleOverlayGpuLabel;

#[derive(Default)]
pub(super) struct ParticleOverlayGpuNode;

#[derive(Resource)]
pub(super) struct ParticleOverlayGpuPipeline {
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
    pub fallback_params_buf: Buffer,
    pub fallback_particle_buf: Buffer,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct ParticleOverlayPipelineKey {
    target_format: TextureFormat,
    sample_count: u32,
}

impl SpecializedRenderPipeline for ParticleOverlayGpuPipeline {
    type Key = ParticleOverlayPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("particle_overlay_gpu_pipeline".into()),
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
pub(super) struct ViewParticleOverlayGpuPipeline {
    pub id: CachedRenderPipelineId,
}

pub(super) fn init_particle_overlay_gpu_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "particle_overlay_gpu_bind_group_layout",
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

    // Keep fallback buffers to avoid bind-group creation failures when MPM is not ready.
    let fallback_params_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("particle_overlay_fallback_params"),
        size: size_of::<GpuMpmParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fallback_particle_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("particle_overlay_fallback_particles"),
        size: size_of::<GpuParticle>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(ParticleOverlayGpuPipeline {
        bind_group_layout,
        shader: asset_server.load(PARTICLE_OVERLAY_SHADER_PATH),
        fallback_params_buf,
        fallback_particle_buf,
    });
}

pub(super) fn prepare_particle_overlay_gpu_pipeline(
    mut commands: Commands,
    pipeline: Res<ParticleOverlayGpuPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ParticleOverlayGpuPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(Entity, &ViewTarget, Option<&Msaa>)>,
) {
    if views.is_empty() {
        static NO_VIEW_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !NO_VIEW_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            warn!("particle_overlay: no render views found during prepare");
        }
        return;
    }
    for (entity, view_target, msaa) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            ParticleOverlayPipelineKey {
                target_format: view_target.main_texture_format(),
                sample_count: msaa.map_or(1, Msaa::samples),
            },
        );
        commands
            .entity(entity)
            .insert(ViewParticleOverlayGpuPipeline { id });
    }
}

impl ViewNode for ParticleOverlayGpuNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static ViewParticleOverlayGpuPipeline,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, view_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let overlay_state = world.resource::<ParticleOverlayState>();
        if !overlay_state.enabled {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(view_pipeline.id) else {
            static PIPELINE_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !PIPELINE_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                warn!("particle_overlay: render pipeline is not ready");
            }
            return Ok(());
        };
        let overlay_pipeline = world.resource::<ParticleOverlayGpuPipeline>();

        let (params_binding, particles_binding, particle_count) =
            if let Some(buffers) = world.get_resource::<MpmGpuBuffers>() {
                (
                    buffers.params_buf.as_entire_binding(),
                    buffers.particle_buf.as_entire_binding(),
                    buffers.particle_count,
                )
            } else {
                (
                    overlay_pipeline.fallback_params_buf.as_entire_binding(),
                    overlay_pipeline.fallback_particle_buf.as_entire_binding(),
                    0,
                )
            };

        let view_uniforms_resource = world.resource::<ViewUniforms>();
        let Some(view_binding) = view_uniforms_resource.uniforms.binding() else {
            static VIEW_UNIFORM_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !VIEW_UNIFORM_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                warn!("particle_overlay: view uniform binding unavailable");
            }
            return Ok(());
        };
        let bind_group = render_context.render_device().create_bind_group(
            "particle_overlay_gpu_bind_group",
            &pipeline_cache.get_bind_group_layout(&overlay_pipeline.bind_group_layout),
            &BindGroupEntries::sequential((view_binding, params_binding, particles_binding)),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("particle_overlay_gpu_pass"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..6, 0..particle_count);

        Ok(())
    }
}
