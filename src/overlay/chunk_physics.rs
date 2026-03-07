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

use super::{PhysicsAreaOverlayState, TileOverlayState};
use crate::params::ActiveOverlayParams;
use crate::physics::gpu_mpm::buffers::{GpuChunkMeta, GpuMpmParams};
use crate::physics::gpu_mpm::gpu_resources::MpmGpuBuffers;

const CHUNK_PHYSICS_OVERLAY_SHADER_PATH: &str = "shaders/overlay/chunk_physics_overlay_gpu.wgsl";

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuChunkOverlayColors {
    occupied_edge: [f32; 4],
    occupied_grid: [f32; 4],
    halo_edge: [f32; 4],
    halo_grid: [f32; 4],
    free_edge: [f32; 4],
    free_grid: [f32; 4],
}

impl Default for GpuChunkOverlayColors {
    fn default() -> Self {
        Self {
            occupied_edge: [0.98, 0.50, 0.12, 0.94],
            occupied_grid: [0.98, 0.50, 0.12, 0.34],
            halo_edge: [0.18, 0.78, 0.96, 0.86],
            halo_grid: [0.18, 0.78, 0.96, 0.30],
            free_edge: [0.70, 0.70, 0.74, 0.72],
            free_grid: [0.58, 0.58, 0.62, 0.26],
        }
    }
}

impl GpuChunkOverlayColors {
    fn from_active_overlay(params: &ActiveOverlayParams) -> Self {
        let colors = &params.0.colors;
        Self {
            occupied_edge: [
                colors.chunk_overlay_occupied_edge.r,
                colors.chunk_overlay_occupied_edge.g,
                colors.chunk_overlay_occupied_edge.b,
                colors.chunk_overlay_occupied_edge.a,
            ],
            occupied_grid: [
                colors.chunk_overlay_occupied_grid.r,
                colors.chunk_overlay_occupied_grid.g,
                colors.chunk_overlay_occupied_grid.b,
                colors.chunk_overlay_occupied_grid.a,
            ],
            halo_edge: [
                colors.chunk_overlay_halo_edge.r,
                colors.chunk_overlay_halo_edge.g,
                colors.chunk_overlay_halo_edge.b,
                colors.chunk_overlay_halo_edge.a,
            ],
            halo_grid: [
                colors.chunk_overlay_halo_grid.r,
                colors.chunk_overlay_halo_grid.g,
                colors.chunk_overlay_halo_grid.b,
                colors.chunk_overlay_halo_grid.a,
            ],
            free_edge: [
                colors.chunk_overlay_free_edge.r,
                colors.chunk_overlay_free_edge.g,
                colors.chunk_overlay_free_edge.b,
                colors.chunk_overlay_free_edge.a,
            ],
            free_grid: [
                colors.chunk_overlay_free_grid.r,
                colors.chunk_overlay_free_grid.g,
                colors.chunk_overlay_free_grid.b,
                colors.chunk_overlay_free_grid.a,
            ],
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct ChunkPhysicsOverlayGpuLabel;

#[derive(Default)]
pub(super) struct ChunkPhysicsOverlayGpuNode;

#[derive(Resource)]
pub(super) struct ChunkPhysicsOverlayGpuPipeline {
    pub bind_group_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
    pub fallback_params_buf: Buffer,
    pub fallback_chunk_meta_buf: Buffer,
    pub fallback_overlay_colors_buf: Buffer,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct ChunkPhysicsOverlayPipelineKey {
    target_format: TextureFormat,
    sample_count: u32,
}

impl SpecializedRenderPipeline for ChunkPhysicsOverlayGpuPipeline {
    type Key = ChunkPhysicsOverlayPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("chunk_physics_overlay_gpu_pipeline".into()),
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
pub(super) struct ViewChunkPhysicsOverlayGpuPipeline {
    pub id: CachedRenderPipelineId,
}

pub(super) fn init_chunk_physics_overlay_gpu_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "chunk_physics_overlay_gpu_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer_sized(
                    false,
                    core::num::NonZeroU64::new(size_of::<GpuMpmParams>() as u64),
                ),
                storage_buffer_read_only_sized(false, None),
                uniform_buffer_sized(
                    false,
                    core::num::NonZeroU64::new(size_of::<GpuChunkOverlayColors>() as u64),
                ),
            ),
        ),
    );

    let fallback_params_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("chunk_physics_overlay_fallback_params"),
        size: size_of::<GpuMpmParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fallback_chunk_meta_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("chunk_physics_overlay_fallback_chunk_meta"),
        size: size_of::<GpuChunkMeta>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fallback_overlay_colors_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("chunk_physics_overlay_fallback_colors"),
        size: size_of::<GpuChunkOverlayColors>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(ChunkPhysicsOverlayGpuPipeline {
        bind_group_layout,
        shader: asset_server.load(CHUNK_PHYSICS_OVERLAY_SHADER_PATH),
        fallback_params_buf,
        fallback_chunk_meta_buf,
        fallback_overlay_colors_buf,
    });
}

pub(super) fn prepare_chunk_physics_overlay_gpu_pipeline(
    mut commands: Commands,
    pipeline: Res<ChunkPhysicsOverlayGpuPipeline>,
    queue: Res<RenderQueue>,
    overlay_params: Option<Res<ActiveOverlayParams>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ChunkPhysicsOverlayGpuPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(Entity, &ViewTarget, Option<&Msaa>)>,
) {
    let overlay_colors = overlay_params
        .as_deref()
        .map(GpuChunkOverlayColors::from_active_overlay)
        .unwrap_or_default();
    queue.write_buffer(
        &pipeline.fallback_overlay_colors_buf,
        0,
        bytemuck::bytes_of(&overlay_colors),
    );

    for (entity, view_target, msaa) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            ChunkPhysicsOverlayPipelineKey {
                target_format: view_target.main_texture_format(),
                sample_count: msaa.map_or(1, Msaa::samples),
            },
        );
        commands
            .entity(entity)
            .insert(ViewChunkPhysicsOverlayGpuPipeline { id });
    }
}

impl ViewNode for ChunkPhysicsOverlayGpuNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static ViewChunkPhysicsOverlayGpuPipeline,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, view_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let chunk_overlay_enabled = world
            .get_resource::<TileOverlayState>()
            .map(|s| s.enabled)
            .unwrap_or(false);
        let physics_overlay_enabled = world
            .get_resource::<PhysicsAreaOverlayState>()
            .map(|s| s.enabled)
            .unwrap_or(false);
        if !chunk_overlay_enabled && !physics_overlay_enabled {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(view_pipeline.id) else {
            return Ok(());
        };
        let overlay_pipeline = world.resource::<ChunkPhysicsOverlayGpuPipeline>();
        let overlay_colors_binding = overlay_pipeline.fallback_overlay_colors_buf.as_entire_binding();

        let (params_binding, chunk_meta_binding, chunk_count) =
            if let Some(buffers) = world.get_resource::<MpmGpuBuffers>() {
                (
                    buffers.params_buf.as_entire_binding(),
                    buffers.chunk_meta_buf.as_entire_binding(),
                    buffers.active_chunk_count,
                )
            } else {
                (
                    overlay_pipeline.fallback_params_buf.as_entire_binding(),
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
            "chunk_physics_overlay_gpu_bind_group",
            &pipeline_cache.get_bind_group_layout(&overlay_pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                view_binding,
                params_binding,
                chunk_meta_binding,
                overlay_colors_binding,
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("chunk_physics_overlay_gpu_pass"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..6, 0..chunk_count);
        Ok(())
    }
}
