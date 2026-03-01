use bevy::asset::AssetServer;
use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::ecs::query::QueryItem;
use bevy::log::warn;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraph, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode,
    ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, uniform_buffer, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendComponent,
    BlendFactor, BlendOperation, BlendState, Buffer, BufferDescriptor, BufferUsages,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, MultisampleState,
    PipelineCache, PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bytemuck::{Pod, Zeroable, cast_slice};

use crate::physics::material::TerrainMaterial;
use crate::physics::state::SimUpdateSet;
use crate::physics::world::constants::{
    CELL_SIZE_M, CHUNK_SIZE_I32, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X,
    WORLD_MIN_CHUNK_Y,
};
use crate::physics::world::terrain::{TerrainCell, TerrainWorld};

use super::water_dot_gpu::WaterDotGpuLabel;

const TERRAIN_DOT_SHADER_PATH: &str = "shaders/render/terrain_dot_gpu.wgsl";
const TERRAIN_GRID_PADDING_CELLS: i32 = 0;
const TERRAIN_DOTS_PER_CELL: u32 = 8;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct TerrainDotGpuLabel;

#[derive(Default)]
struct TerrainDotGpuNode;

#[derive(Clone, Copy, Debug)]
struct TerrainCellLayout {
    width: u32,
    height: u32,
    origin: IVec2,
}

impl TerrainCellLayout {
    fn cell_count(&self) -> u32 {
        self.width * self.height
    }
}

fn terrain_cell_layout() -> TerrainCellLayout {
    let min_cell = IVec2::new(
        WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32 - TERRAIN_GRID_PADDING_CELLS,
        WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32 - TERRAIN_GRID_PADDING_CELLS,
    );
    let max_cell_exclusive = IVec2::new(
        (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 + TERRAIN_GRID_PADDING_CELLS,
        (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 + TERRAIN_GRID_PADDING_CELLS,
    );
    let span = (max_cell_exclusive - min_cell).max(IVec2::ONE);

    TerrainCellLayout {
        width: span.x as u32,
        height: span.y as u32,
        origin: min_cell,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainDotParams {
    origin_x: f32,
    origin_y: f32,
    cell_size_m: f32,
    dot_size_m: f32,
    width: u32,
    height: u32,
    palette_seed: u32,
    dots_per_cell: u32,
}

const _: () = assert!(std::mem::size_of::<TerrainDotParams>() == 32);

#[derive(Resource)]
struct TerrainDotGpuResources {
    layout: TerrainCellLayout,
    params_buf: Buffer,
    cells_buf: Buffer,
}

#[derive(Resource)]
struct TerrainDotGpuPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct TerrainDotPipelineKey {
    target_format: TextureFormat,
    sample_count: u32,
}

impl SpecializedRenderPipeline for TerrainDotGpuPipeline {
    type Key = TerrainDotPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("terrain_dot_gpu_pipeline".into()),
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
struct ViewTerrainDotGpuPipeline {
    id: CachedRenderPipelineId,
}

#[derive(Resource, Default)]
pub struct TerrainDotUploadRequest {
    upload_cells: bool,
    initialized: bool,
    width: u32,
    height: u32,
    origin_x: i32,
    origin_y: i32,
    cells: Vec<u32>,
}

impl Clone for TerrainDotUploadRequest {
    fn clone(&self) -> Self {
        Self {
            upload_cells: self.upload_cells,
            initialized: self.initialized,
            width: self.width,
            height: self.height,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            cells: self.cells.clone(),
        }
    }
}

impl ExtractResource for TerrainDotUploadRequest {
    type Source = TerrainDotUploadRequest;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

fn encode_terrain_cell(cell: TerrainCell) -> u32 {
    match cell {
        TerrainCell::Empty => 0,
        TerrainCell::Solid {
            material: TerrainMaterial::Stone,
            ..
        } => 1,
        TerrainCell::Solid {
            material: TerrainMaterial::Soil,
            ..
        } => 2,
        TerrainCell::Solid {
            material: TerrainMaterial::Sand,
            ..
        } => 3,
    }
}

fn prepare_terrain_dot_upload_request(
    terrain: Res<TerrainWorld>,
    mut request: ResMut<TerrainDotUploadRequest>,
) {
    if request.initialized && !terrain.is_changed() {
        request.upload_cells = false;
        return;
    }

    let layout = terrain_cell_layout();
    let cell_count = layout.cell_count() as usize;
    request.cells.resize(cell_count, 0);

    for y in 0..layout.height as i32 {
        for x in 0..layout.width as i32 {
            let global = layout.origin + IVec2::new(x, y);
            let idx = y as usize * layout.width as usize + x as usize;
            request.cells[idx] = encode_terrain_cell(terrain.get_cell_or_generated(global));
        }
    }

    request.width = layout.width;
    request.height = layout.height;
    request.origin_x = layout.origin.x;
    request.origin_y = layout.origin.y;
    request.upload_cells = true;
    request.initialized = true;
}

fn init_terrain_dot_gpu_resources(mut commands: Commands, render_device: Res<RenderDevice>) {
    let layout = terrain_cell_layout();

    let params = TerrainDotParams {
        origin_x: layout.origin.x as f32 * CELL_SIZE_M,
        origin_y: layout.origin.y as f32 * CELL_SIZE_M,
        cell_size_m: CELL_SIZE_M,
        dot_size_m: CELL_SIZE_M / TERRAIN_DOTS_PER_CELL as f32,
        width: layout.width,
        height: layout.height,
        palette_seed: 0x5EED_7163,
        dots_per_cell: TERRAIN_DOTS_PER_CELL,
    };

    let params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_dot_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );

    let cells_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("terrain_dot_cells"),
        size: layout.cell_count() as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(TerrainDotGpuResources {
        layout,
        params_buf,
        cells_buf,
    });
}

fn init_terrain_dot_gpu_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "terrain_dot_gpu_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer_sized(false, core::num::NonZeroU64::new(32)),
                storage_buffer_read_only_sized(false, None),
            ),
        ),
    );

    commands.insert_resource(TerrainDotGpuPipeline {
        bind_group_layout,
        shader: asset_server.load(TERRAIN_DOT_SHADER_PATH),
    });
}

fn prepare_terrain_dot_uploads(
    upload: Res<TerrainDotUploadRequest>,
    resources: Res<TerrainDotGpuResources>,
    queue: Res<RenderQueue>,
) {
    if !upload.upload_cells || upload.cells.is_empty() {
        return;
    }
    if upload.width != resources.layout.width
        || upload.height != resources.layout.height
        || upload.origin_x != resources.layout.origin.x
        || upload.origin_y != resources.layout.origin.y
    {
        warn!(
            "terrain_dot_gpu: upload layout mismatch (upload {}x{} @ {},{} vs gpu {}x{} @ {},{})",
            upload.width,
            upload.height,
            upload.origin_x,
            upload.origin_y,
            resources.layout.width,
            resources.layout.height,
            resources.layout.origin.x,
            resources.layout.origin.y,
        );
        return;
    }

    queue.write_buffer(&resources.cells_buf, 0, cast_slice(&upload.cells));
}

fn prepare_terrain_dot_gpu_pipeline(
    mut commands: Commands,
    pipeline: Res<TerrainDotGpuPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TerrainDotGpuPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(Entity, &ViewTarget, Option<&Msaa>)>,
) {
    for (entity, view_target, msaa) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            TerrainDotPipelineKey {
                target_format: view_target.main_texture_format(),
                sample_count: msaa.map_or(1, Msaa::samples),
            },
        );
        commands
            .entity(entity)
            .insert(ViewTerrainDotGpuPipeline { id });
    }
}

impl ViewNode for TerrainDotGpuNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static ViewTerrainDotGpuPipeline,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, view_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(resources) = world.get_resource::<TerrainDotGpuResources>() else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(view_pipeline.id) else {
            static PIPELINE_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !PIPELINE_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                warn!("terrain_dot_gpu: render pipeline is not ready");
            }
            return Ok(());
        };

        let view_uniforms_resource = world.resource::<ViewUniforms>();
        let Some(view_binding) = view_uniforms_resource.uniforms.binding() else {
            static VIEW_UNIFORM_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !VIEW_UNIFORM_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                warn!("terrain_dot_gpu: view uniform binding unavailable");
            }
            return Ok(());
        };

        let render_pipeline = world.resource::<TerrainDotGpuPipeline>();
        let bind_group = render_context.render_device().create_bind_group(
            "terrain_dot_gpu_bind_group",
            &pipeline_cache.get_bind_group_layout(&render_pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                view_binding,
                resources.params_buf.as_entire_binding(),
                resources.cells_buf.as_entire_binding(),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("terrain_dot_gpu_pass"),
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

fn link_terrain_and_water_graph(mut graph: ResMut<RenderGraph>) {
    let _ = graph.try_add_node_edge(TerrainDotGpuLabel, WaterDotGpuLabel);
}

pub struct TerrainDotGpuPlugin;

impl Plugin for TerrainDotGpuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainDotUploadRequest>()
            .add_systems(
                Update,
                prepare_terrain_dot_upload_request.in_set(SimUpdateSet::Rendering),
            )
            .add_plugins(ExtractResourcePlugin::<TerrainDotUploadRequest>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SpecializedRenderPipelines<TerrainDotGpuPipeline>>()
            .add_systems(
                RenderStartup,
                (
                    init_terrain_dot_gpu_resources,
                    init_terrain_dot_gpu_pipeline,
                    link_terrain_and_water_graph,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_terrain_dot_uploads,
                    prepare_terrain_dot_gpu_pipeline,
                )
                    .in_set(RenderSystems::Prepare),
            )
            .add_render_graph_node::<ViewNodeRunner<TerrainDotGpuNode>>(Core2d, TerrainDotGpuLabel)
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::MainTransparentPass,
                    TerrainDotGpuLabel,
                    Node2d::EndMainPass,
                ),
            );
    }
}
