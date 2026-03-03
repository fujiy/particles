/// GPU-resident terrain Near-cache pipeline.
///
/// Architecture (initial version — ring buffer / Far cache in follow-up):
///   Main graph:   `TerrainNearUpdateLabel` (compute) → writes R16Uint material-ID texture
///   Core2d graph: `TerrainComposeLabel`    (fragment ViewNode) → reads texture → screen
///
/// Data flow:
///   CPU TerrainWorld → `TerrainNearUpdateRequest` (main world, changes only on terrain edit)
///     → ExtractResource → render world
///     → `prepare_terrain_near_uploads` (writes override_buf, sets dirty flag)
///     → `TerrainNearUpdateNode` (compute, dispatches only when dirty)
///     → `near_texture` (R16Uint, GPU-resident, persistent between frames)
///     → `TerrainComposeNode` (fragment ViewNode, always reads cached texture)
///     → `clear_terrain_near_dirty` (Cleanup) → dirty = false until next terrain change
use bevy::asset::AssetServer;
use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::ecs::query::QueryItem;
use bevy::log::warn;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode,
    ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, texture_2d, texture_storage_2d, uniform_buffer,
    uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    BindingResource, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer,
    BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor,
    ComputePipelineDescriptor, Extent3d, FragmentState, MultisampleState, PipelineCache,
    PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    SpecializedRenderPipeline, SpecializedRenderPipelines, StorageTextureAccess, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, VertexState,
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

const TERRAIN_NEAR_UPDATE_SHADER_PATH: &str = "shaders/render/terrain_near_update.wgsl";
const TERRAIN_COMPOSE_SHADER_PATH: &str = "shaders/render/terrain_compose.wgsl";
const TERRAIN_DOTS_PER_CELL: u32 = 8;
const NEAR_UPDATE_WORKGROUP: u32 = 8;

// ── Render labels ─────────────────────────────────────────────────────────────

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct TerrainNearUpdateLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct TerrainComposeLabel;

// ── Cell layout (fixed world bounds) ─────────────────────────────────────────

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
        WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32,
        WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32,
    );
    let max_cell_exclusive = IVec2::new(
        (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32,
        (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32,
    );
    let span = (max_cell_exclusive - min_cell).max(IVec2::ONE);
    TerrainCellLayout {
        width: span.x as u32,
        height: span.y as u32,
        origin: min_cell,
    }
}

// ── GPU uniform structs ───────────────────────────────────────────────────────

/// Uniform for `TerrainNearUpdate` compute pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainNearParams {
    override_origin_x: i32,
    override_origin_y: i32,
    override_width: u32,
    override_height: u32,
    cache_origin_x: i32,
    cache_origin_y: i32,
    _pad0: u32,
    _pad1: u32,
}
const _: () = assert!(std::mem::size_of::<TerrainNearParams>() == 32);

/// Uniform for `TerrainCompose` fragment pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainComposeParams {
    origin_x: f32,
    origin_y: f32,
    cell_size_m: f32,
    dot_size_m: f32,
    cache_origin_x: i32,
    cache_origin_y: i32,
    palette_seed: u32,
    dots_per_cell: u32,
}
const _: () = assert!(std::mem::size_of::<TerrainComposeParams>() == 32);

// ── GPU resources (render world) ──────────────────────────────────────────────

#[derive(Resource)]
struct TerrainNearGpuResources {
    layout: TerrainCellLayout,
    /// R16Uint persistent texture: material IDs at cell resolution.
    near_texture: Texture,
    near_texture_view: TextureView,
    /// Storage buffer: one u32 per world cell with encoded material ID.
    override_buf: Buffer,
    /// Uniform buffer for compute pass (TerrainNearParams).
    near_params_buf: Buffer,
    /// Uniform buffer for fragment pass (TerrainComposeParams).
    compose_params_buf: Buffer,
    /// How many more frames to keep the dirty flag set, used to retry when the
    /// pipeline is still compiling on the first frame data arrives.
    pending_dispatch_frames: u32,
}

/// Set `true` when override data needs to be dispatched via compute; cleared in Cleanup.
#[derive(Resource, Default)]
struct TerrainNearCacheDirty(bool);

// ── Compute (TerrainNearUpdate) pipeline & node ───────────────────────────────

#[derive(Resource)]
struct TerrainNearUpdatePipeline {
    /// Stored for bind-group creation via `pipeline_cache.get_bind_group_layout`.
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
}

#[derive(Default)]
struct TerrainNearUpdateNode;

impl Node for TerrainNearUpdateNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if !world
            .get_resource::<TerrainNearCacheDirty>()
            .is_some_and(|d| d.0)
        {
            return Ok(());
        }
        let Some(resources) = world.get_resource::<TerrainNearGpuResources>() else {
            return Ok(());
        };
        let Some(pipeline_res) = world.get_resource::<TerrainNearUpdatePipeline>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline_id) else {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                match pipeline_cache.get_compute_pipeline_state(pipeline_res.pipeline_id) {
                    CachedPipelineState::Err(e) => {
                        warn!("terrain_gpu: near_update pipeline error: {e}")
                    }
                    s => warn!("terrain_gpu: near_update pipeline not ready: {s:?}"),
                }
            }
            return Ok(());
        };

        let layout = pipeline_cache.get_bind_group_layout(&pipeline_res.bind_group_layout);
        let bind_group = render_context.render_device().create_bind_group(
            "terrain_near_update_bg",
            &layout,
            &BindGroupEntries::sequential((
                resources.near_params_buf.as_entire_binding(),
                resources.override_buf.as_entire_binding(),
                BindingResource::TextureView(&resources.near_texture_view),
            )),
        );

        let w = resources.layout.width.div_ceil(NEAR_UPDATE_WORKGROUP);
        let h = resources.layout.height.div_ceil(NEAR_UPDATE_WORKGROUP);

        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("terrain_near_update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(w, h, 1);

        Ok(())
    }
}

// ── Fragment (TerrainCompose) pipeline & node ─────────────────────────────────

#[derive(Resource)]
struct TerrainComposePipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct TerrainComposePipelineKey {
    target_format: TextureFormat,
    sample_count: u32,
}

impl SpecializedRenderPipeline for TerrainComposePipeline {
    type Key = TerrainComposePipelineKey;
    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("terrain_compose_pipeline".into()),
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
struct ViewTerrainComposePipeline {
    id: CachedRenderPipelineId,
}

#[derive(Default)]
struct TerrainComposeNode;

impl ViewNode for TerrainComposeNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static ViewTerrainComposePipeline,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, view_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(resources) = world.get_resource::<TerrainNearGpuResources>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(view_pipeline.id) else {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                warn!("terrain_gpu: compose pipeline not ready");
            }
            return Ok(());
        };
        let view_uniforms = world.resource::<ViewUniforms>();
        let Some(view_binding) = view_uniforms.uniforms.binding() else {
            return Ok(());
        };
        let compose_pipeline = world.resource::<TerrainComposePipeline>();
        let bind_group = render_context.render_device().create_bind_group(
            "terrain_compose_bg",
            &pipeline_cache.get_bind_group_layout(&compose_pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                view_binding,
                resources.compose_params_buf.as_entire_binding(),
                BindingResource::TextureView(&resources.near_texture_view),
            )),
        );

        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("terrain_compose_pass"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        pass.draw(0..6, 0..1);

        Ok(())
    }
}

// ── CPU-side request (main world → render world via ExtractResource) ──────────

#[derive(Resource, Default)]
pub struct TerrainNearUpdateRequest {
    dirty: bool,
    initialized: bool,
    width: u32,
    height: u32,
    origin_x: i32,
    origin_y: i32,
    cells: Vec<u32>,
}

impl Clone for TerrainNearUpdateRequest {
    fn clone(&self) -> Self {
        Self {
            dirty: self.dirty,
            initialized: self.initialized,
            width: self.width,
            height: self.height,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            cells: self.cells.clone(),
        }
    }
}

impl ExtractResource for TerrainNearUpdateRequest {
    type Source = TerrainNearUpdateRequest;
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

pub fn prepare_terrain_near_update_request(
    terrain: Res<TerrainWorld>,
    mut request: ResMut<TerrainNearUpdateRequest>,
) {
    if request.initialized && !terrain.is_changed() {
        request.dirty = false;
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
    request.dirty = true;
    request.initialized = true;
}

// ── Render startup systems ────────────────────────────────────────────────────

fn init_terrain_near_gpu_resources(mut commands: Commands, render_device: Res<RenderDevice>) {
    let layout = terrain_cell_layout();

    let near_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("terrain_near_cache"),
        size: Extent3d {
            width: layout.width,
            height: layout.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R16Uint,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let near_texture_view = near_texture.create_view(&TextureViewDescriptor::default());

    let override_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("terrain_near_override"),
        size: layout.cell_count() as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let near_params = TerrainNearParams {
        override_origin_x: layout.origin.x,
        override_origin_y: layout.origin.y,
        override_width: layout.width,
        override_height: layout.height,
        cache_origin_x: layout.origin.x,
        cache_origin_y: layout.origin.y,
        _pad0: 0,
        _pad1: 0,
    };
    let near_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_near_params"),
            contents: bytemuck::bytes_of(&near_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );

    let compose_params = TerrainComposeParams {
        origin_x: layout.origin.x as f32 * CELL_SIZE_M,
        origin_y: layout.origin.y as f32 * CELL_SIZE_M,
        cell_size_m: CELL_SIZE_M,
        dot_size_m: CELL_SIZE_M / TERRAIN_DOTS_PER_CELL as f32,
        cache_origin_x: layout.origin.x,
        cache_origin_y: layout.origin.y,
        palette_seed: 0x5EED_7163,
        dots_per_cell: TERRAIN_DOTS_PER_CELL,
    };
    let compose_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_compose_params"),
            contents: bytemuck::bytes_of(&compose_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );

    commands.insert_resource(TerrainNearGpuResources {
        layout,
        near_texture,
        near_texture_view,
        override_buf,
        near_params_buf,
        compose_params_buf,
        pending_dispatch_frames: 0,
    });
    commands.insert_resource(TerrainNearCacheDirty(false));
}

fn init_terrain_near_update_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    // The bind_group_layout is stored in the resource for use in the node.
    // A separate (identical) descriptor is passed to ComputePipelineDescriptor.layout.
    let make_layout = || {
        BindGroupLayoutDescriptor::new(
            "terrain_near_update_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, core::num::NonZeroU64::new(32)),
                    storage_buffer_read_only_sized(false, None),
                    texture_storage_2d(TextureFormat::R16Uint, StorageTextureAccess::WriteOnly),
                ),
            ),
        )
    };

    let shader = asset_server.load(TERRAIN_NEAR_UPDATE_SHADER_PATH);
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("terrain_near_update_compute".into()),
        layout: vec![make_layout()],
        push_constant_ranges: vec![],
        shader,
        shader_defs: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(TerrainNearUpdatePipeline {
        bind_group_layout: make_layout(),
        pipeline_id,
    });
}

fn init_terrain_compose_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "terrain_compose_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer_sized(false, core::num::NonZeroU64::new(32)),
                texture_2d(TextureSampleType::Uint),
            ),
        ),
    );
    commands.insert_resource(TerrainComposePipeline {
        bind_group_layout,
        shader: asset_server.load(TERRAIN_COMPOSE_SHADER_PATH),
    });
}

fn link_terrain_and_water_graph(mut graph: ResMut<RenderGraph>) {
    // Add Core2d edge: TerrainCompose → WaterDotGpu (terrain renders before water).
    if let Some(core2d) = graph.get_sub_graph_mut(Core2d) {
        let _ = core2d.try_add_node_edge(TerrainComposeLabel, WaterDotGpuLabel);
    }
}

// ── Render prepare / cleanup systems ─────────────────────────────────────────

fn prepare_terrain_near_uploads(
    upload: Res<TerrainNearUpdateRequest>,
    mut resources: ResMut<TerrainNearGpuResources>,
    queue: Res<RenderQueue>,
    mut dirty: ResMut<TerrainNearCacheDirty>,
) {
    if upload.dirty && !upload.cells.is_empty() {
        if upload.width != resources.layout.width || upload.height != resources.layout.height {
            warn!(
                "terrain_gpu: layout mismatch (upload {}×{}, gpu {}×{})",
                upload.width, upload.height, resources.layout.width, resources.layout.height,
            );
        } else {
            queue.write_buffer(&resources.override_buf, 0, cast_slice(&upload.cells));
            // Allow up to 32 frames of retry in case the pipeline is still compiling.
            resources.pending_dispatch_frames = 32;
        }
    }
    if resources.pending_dispatch_frames > 0 {
        dirty.0 = true;
        resources.pending_dispatch_frames -= 1;
    }
}

fn prepare_terrain_compose_pipeline(
    mut commands: Commands,
    pipeline: Res<TerrainComposePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TerrainComposePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(Entity, &ViewTarget, Option<&Msaa>)>,
) {
    for (entity, view_target, msaa) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            TerrainComposePipelineKey {
                target_format: view_target.main_texture_format(),
                sample_count: msaa.map_or(1, Msaa::samples),
            },
        );
        match pipeline_cache.get_render_pipeline_state(id) {
            CachedPipelineState::Err(e) => {
                static COMPOSE_ERR_WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !COMPOSE_ERR_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    warn!("terrain_gpu: compose pipeline error: {e}");
                }
            }
            _ => {}
        }
        commands
            .entity(entity)
            .insert(ViewTerrainComposePipeline { id });
    }
}

/// Clear the dirty flag after the render graph (and thus the compute pass) has run.
fn clear_terrain_near_dirty(mut dirty: ResMut<TerrainNearCacheDirty>) {
    dirty.0 = false;
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct TerrainGpuPlugin;

impl Plugin for TerrainGpuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainNearUpdateRequest>()
            .add_systems(
                Update,
                prepare_terrain_near_update_request.in_set(SimUpdateSet::Rendering),
            )
            .add_plugins(ExtractResourcePlugin::<TerrainNearUpdateRequest>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SpecializedRenderPipelines<TerrainComposePipeline>>()
            .add_systems(
                RenderStartup,
                (
                    init_terrain_near_gpu_resources,
                    init_terrain_near_update_pipeline,
                    init_terrain_compose_pipeline,
                    link_terrain_and_water_graph,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_terrain_near_uploads,
                    prepare_terrain_compose_pipeline,
                )
                    .in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                clear_terrain_near_dirty.in_set(RenderSystems::Cleanup),
            )
            // TerrainComposeLabel: ViewNode in Core2d graph.
            .add_render_graph_node::<ViewNodeRunner<TerrainComposeNode>>(
                Core2d,
                TerrainComposeLabel,
            )
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::MainTransparentPass,
                    TerrainComposeLabel,
                    Node2d::EndMainPass,
                ),
            );

        // TerrainNearUpdateLabel: non-view compute node in main render graph.
        {
            let mut main_graph = render_app.world_mut().resource_mut::<RenderGraph>();
            main_graph.add_node(TerrainNearUpdateLabel, TerrainNearUpdateNode);
            main_graph.add_node_edge(TerrainNearUpdateLabel, CameraDriverLabel);
        }
    }
}
