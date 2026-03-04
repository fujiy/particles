/// GPU-resident terrain Near-cache pipeline.
///
/// Architecture (initial version — ring buffer / Far cache in follow-up):
///   Main graph:   `TerrainNearUpdateLabel` (compute) → writes R16Uint material-ID texture
///   Core2d graph: `TerrainComposeLabel`    (fragment ViewNode) → reads texture → screen
///
/// Data flow:
///   CPU TerrainWorld → `TerrainNearUpdateRequest` (main world, dirty cells on pan/edit)
///     → ExtractResource → render world
///     → `prepare_terrain_near_uploads` (writes dirty-cell SSBO, sets dirty flag)
///     → `TerrainNearUpdateNode` (compute, dispatches only when dirty)
///     → `near_texture` (R16Uint, GPU-resident, persistent between frames)
///     → `TerrainComposeNode` (fragment ViewNode, always reads cached texture)
///     → `clear_terrain_near_dirty` (Cleanup) → dirty = false until next terrain change
use std::collections::VecDeque;

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
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BindingResource,
    BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferDescriptor,
    BufferUsages, CachedComputePipelineId, CachedPipelineState, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
    FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPassDescriptor,
    RenderPipelineDescriptor, ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines,
    StorageTextureAccess, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::window::PrimaryWindow;
use bytemuck::{Pod, Zeroable, cast_slice};

use crate::camera_controller::MainCamera;
use crate::physics::material::TerrainMaterial;
use crate::physics::state::SimUpdateSet;
use crate::physics::world::constants::CELL_SIZE_M;
use crate::physics::world::terrain::{TerrainCell, TerrainWorld};

use super::{TerrainRenderDiagnostics, water_dot_gpu::WaterDotGpuLabel};

const TERRAIN_NEAR_UPDATE_SHADER_PATH: &str = "shaders/render/terrain_near_update.wgsl";
const TERRAIN_GEN_SHADER_PATH: &str = "shaders/render/terrain_gen.wgsl";
const TERRAIN_COMPOSE_SHADER_PATH: &str = "shaders/render/terrain_compose.wgsl";
const TERRAIN_DOTS_PER_CELL: u32 = 8;
const NEAR_UPDATE_WORKGROUP: u32 = 64;
const MAX_DISPATCH_GROUPS_X: u32 = 65_535;
const MAX_DIRTY_CELLS_PER_DISPATCH: u32 = NEAR_UPDATE_WORKGROUP * MAX_DISPATCH_GROUPS_X;
const TERRAIN_OVERRIDE_NONE: u32 = 0xFFFF;
// Temporary debug policy: treat terrain as read-only generated field.
// This disables CPU override sampling (loaded/edited cells) to avoid seams caused by
// mixed CPU-generated and GPU-generated terrain while REND-GPU-01 is in progress.
const TERRAIN_OVERRIDE_UPLOAD_ENABLED: bool = false;
const NEAR_MARGIN_FACTOR: f32 = 1.35;
const NEAR_QUALITY_DIVISOR: f32 = 1.0;
const NEAR_TEX_MIN_CELLS: u32 = 1;
const NEAR_TEX_MAX_CELLS: u32 = 2048;
const DEFAULT_SCREEN_SIZE_PX: UVec2 = UVec2::new(1280, 720);
const DEFAULT_CAMERA_VIEWPORT_HEIGHT_M: f32 = 14.0;

#[derive(Clone, Copy, Debug)]
pub struct TerrainDirtyTile {
    pub _world_origin: IVec2,
    pub _extent_cells: UVec2,
}

#[derive(Resource, Debug)]
pub struct TerrainCacheState {
    pub cache_origin_world: IVec2,
    pub ring_offset: IVec2,
    pub lod_k: i32,
    pub near_dirty_queue: VecDeque<TerrainDirtyTile>,
    pub far_dirty_queue: VecDeque<TerrainDirtyTile>,
    pub near_extent_cells: UVec2,
}

impl Default for TerrainCacheState {
    fn default() -> Self {
        Self {
            cache_origin_world: IVec2::ZERO,
            ring_offset: IVec2::ZERO,
            lod_k: 0,
            near_dirty_queue: VecDeque::new(),
            far_dirty_queue: VecDeque::new(),
            near_extent_cells: compute_near_cache_extent(DEFAULT_SCREEN_SIZE_PX),
        }
    }
}

fn compute_near_cache_extent(screen_size_px: UVec2) -> UVec2 {
    let size = screen_size_px.max(UVec2::splat(1)).as_vec2();
    let aspect = size.x / size.y;
    let viewport_world = Vec2::new(
        DEFAULT_CAMERA_VIEWPORT_HEIGHT_M * aspect,
        DEFAULT_CAMERA_VIEWPORT_HEIGHT_M,
    );
    compute_near_cache_extent_from_viewport_world(viewport_world)
}

fn compute_near_cache_extent_from_viewport_world(viewport_world_m: Vec2) -> UVec2 {
    let viewport_world_m = viewport_world_m.max(Vec2::splat(CELL_SIZE_M.max(1e-6)));
    let viewport_cells = viewport_world_m / CELL_SIZE_M;
    let scaled = viewport_cells * (NEAR_MARGIN_FACTOR / NEAR_QUALITY_DIVISOR);
    let mut extent = UVec2::new(
        scaled
            .x
            .ceil()
            .max(NEAR_TEX_MIN_CELLS as f32)
            .min(NEAR_TEX_MAX_CELLS as f32) as u32,
        scaled
            .y
            .ceil()
            .max(NEAR_TEX_MIN_CELLS as f32)
            .min(NEAR_TEX_MAX_CELLS as f32) as u32,
    );
    let area = u64::from(extent.x) * u64::from(extent.y);
    let max_area = u64::from(MAX_DIRTY_CELLS_PER_DISPATCH);
    if area > max_area {
        if extent.x >= extent.y {
            extent.x = (MAX_DIRTY_CELLS_PER_DISPATCH / extent.y.max(1)).max(1);
        } else {
            extent.y = (MAX_DIRTY_CELLS_PER_DISPATCH / extent.x.max(1)).max(1);
        }
    }
    extent
}

fn compute_near_cache_extent_for_projection(
    screen_size_px: UVec2,
    projection: &Projection,
) -> UVec2 {
    let Projection::Orthographic(ortho) = projection else {
        return compute_near_cache_extent(screen_size_px);
    };
    let viewport_world = (ortho.area.max - ortho.area.min).abs();
    if viewport_world.x <= 0.0 || viewport_world.y <= 0.0 {
        return compute_near_cache_extent(screen_size_px);
    }
    compute_near_cache_extent_from_viewport_world(viewport_world)
}

fn normalize_ring_offset(ring_offset: IVec2, extent: UVec2) -> IVec2 {
    IVec2::new(
        ring_offset.x.rem_euclid(extent.x.max(1) as i32),
        ring_offset.y.rem_euclid(extent.y.max(1) as i32),
    )
}

#[allow(dead_code)]
fn world_cell_to_texture_index(
    world_cell: IVec2,
    cache_origin_world: IVec2,
    ring_offset: IVec2,
    cache_extent: UVec2,
) -> Option<usize> {
    if cache_extent.x == 0 || cache_extent.y == 0 {
        return None;
    }

    let logical = world_cell - cache_origin_world;
    if logical.x < 0
        || logical.y < 0
        || logical.x >= cache_extent.x as i32
        || logical.y >= cache_extent.y as i32
    {
        return None;
    }

    let ring = normalize_ring_offset(ring_offset, cache_extent);
    let tx = (logical.x + ring.x).rem_euclid(cache_extent.x as i32) as usize;
    let ty = (logical.y + ring.y).rem_euclid(cache_extent.y as i32) as usize;
    Some(ty * cache_extent.x as usize + tx)
}

fn saturating_u32_to_i32(v: u32) -> i32 {
    i32::try_from(v).unwrap_or(i32::MAX)
}

// ── Render labels ─────────────────────────────────────────────────────────────

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct TerrainNearUpdateLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct TerrainComposeLabel;

// ── GPU uniform structs ───────────────────────────────────────────────────────

/// Uniform for `TerrainNearUpdate` compute pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainNearParams {
    cache_origin_x: i32,
    cache_origin_y: i32,
    cache_width: u32,
    cache_height: u32,
    ring_offset_x: i32,
    ring_offset_y: i32,
    generation_enabled: u32,
    override_none: u32,
    dirty_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
const _: () = assert!(std::mem::size_of::<TerrainNearParams>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainDirtyCellGpu {
    world_x: i32,
    world_y: i32,
    override_material: u32,
    _pad0: u32,
}
const _: () = assert!(std::mem::size_of::<TerrainDirtyCellGpu>() == 16);

/// Uniform for `TerrainCompose` fragment pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainComposeParams {
    cell_size_m: f32,
    dot_size_m: f32,
    cache_origin_x: i32,
    cache_origin_y: i32,
    palette_seed: u32,
    dots_per_cell: u32,
    ring_offset_x: i32,
    ring_offset_y: i32,
}
const _: () = assert!(std::mem::size_of::<TerrainComposeParams>() == 32);

// ── GPU resources (render world) ──────────────────────────────────────────────

struct TerrainNearGpuCache {
    /// R16Uint persistent texture: material IDs at cell resolution.
    _near_texture: Texture,
    near_texture_view: TextureView,
    extent_cells: UVec2,
}

#[derive(Resource)]
struct TerrainNearGpuResources {
    near_cache: TerrainNearGpuCache,
    /// Storage buffer: dirty world cells to update in the current dispatch.
    dirty_cells_buf: Buffer,
    dirty_cells_capacity: u32,
    /// Uniform buffer for compute pass (TerrainNearParams).
    near_params_buf: Buffer,
    /// Uniform buffer for fragment pass (TerrainComposeParams).
    compose_params_buf: Buffer,
    /// How many more frames to keep the dirty flag set, used to retry when the
    /// pipeline is still compiling on the first frame data arrives.
    pending_dispatch_frames: u32,
    pending_dispatch_cells: u32,
}

/// Set `true` when override data needs to be dispatched via compute; cleared in Cleanup.
#[derive(Resource, Default)]
struct TerrainNearCacheDirty(bool);

fn make_near_cache(render_device: &RenderDevice, extent: UVec2) -> TerrainNearGpuCache {
    let near_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("terrain_near_cache"),
        size: Extent3d {
            width: extent.x,
            height: extent.y,
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
    TerrainNearGpuCache {
        _near_texture: near_texture,
        near_texture_view,
        extent_cells: extent,
    }
}

fn make_dirty_cells_buffer(render_device: &RenderDevice, dirty_cells_capacity: u32) -> Buffer {
    let cap = dirty_cells_capacity.max(1);
    render_device.create_buffer(&BufferDescriptor {
        label: Some("terrain_near_dirty_cells"),
        size: cap as u64 * std::mem::size_of::<TerrainDirtyCellGpu>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// ── Compute (TerrainNearUpdate) pipeline & node ───────────────────────────────

#[derive(Resource)]
struct TerrainNearUpdatePipeline {
    /// Stored for bind-group creation via `pipeline_cache.get_bind_group_layout`.
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
    _terrain_gen_shader: Handle<Shader>,
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
                resources.dirty_cells_buf.as_entire_binding(),
                BindingResource::TextureView(&resources.near_cache.near_texture_view),
            )),
        );

        let dirty_count = resources.pending_dispatch_cells;
        if dirty_count == 0 {
            return Ok(());
        }
        let w = dirty_count.div_ceil(NEAR_UPDATE_WORKGROUP);

        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("terrain_near_update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(w, 1, 1);

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
                BindingResource::TextureView(&resources.near_cache.near_texture_view),
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
    terrain_render_version: u64,
    generation_enabled: bool,
    width: u32,
    height: u32,
    origin_x: i32,
    origin_y: i32,
    ring_offset_x: i32,
    ring_offset_y: i32,
    dirty_cells: Vec<TerrainDirtyCellGpu>,
}

impl Clone for TerrainNearUpdateRequest {
    fn clone(&self) -> Self {
        Self {
            dirty: self.dirty,
            initialized: self.initialized,
            terrain_render_version: self.terrain_render_version,
            generation_enabled: self.generation_enabled,
            width: self.width,
            height: self.height,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            ring_offset_x: self.ring_offset_x,
            ring_offset_y: self.ring_offset_y,
            dirty_cells: self.dirty_cells.clone(),
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

fn override_material_or_none(terrain: &TerrainWorld, global_cell: IVec2) -> u32 {
    if !TERRAIN_OVERRIDE_UPLOAD_ENABLED {
        return TERRAIN_OVERRIDE_NONE;
    }
    terrain
        .get_loaded_or_overridden_cell(global_cell)
        .map(encode_terrain_cell)
        .unwrap_or(TERRAIN_OVERRIDE_NONE)
}

fn append_dirty_rect_cells(
    terrain: &TerrainWorld,
    dirty_cells: &mut Vec<TerrainDirtyCellGpu>,
    rect_origin: IVec2,
    rect_extent: UVec2,
) {
    let rect_h = saturating_u32_to_i32(rect_extent.y).max(0);
    let rect_w = saturating_u32_to_i32(rect_extent.x).max(0);
    if rect_w == 0 || rect_h == 0 {
        return;
    }
    dirty_cells.reserve(rect_extent.x as usize * rect_extent.y as usize);
    for y in 0..rect_h {
        for x in 0..rect_w {
            let global = rect_origin + IVec2::new(x, y);
            dirty_cells.push(TerrainDirtyCellGpu {
                world_x: global.x,
                world_y: global.y,
                override_material: override_material_or_none(terrain, global),
                _pad0: 0,
            });
        }
    }
}

pub fn prepare_terrain_near_update_request(
    terrain: Res<TerrainWorld>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut request: ResMut<TerrainNearUpdateRequest>,
    mut cache_state: ResMut<TerrainCacheState>,
    mut diagnostics: ResMut<TerrainRenderDiagnostics>,
    camera_q: Query<(&Transform, &Projection), With<MainCamera>>,
) {
    diagnostics.terrain_generation_eval_count_frame = 0;
    diagnostics.terrain_generation_origin_delta_x_frame = 0;
    diagnostics.terrain_generation_origin_delta_y_frame = 0;
    diagnostics.terrain_generation_full_refresh_frame = false;
    diagnostics.terrain_generation_full_refresh_reason_bits = 0;

    const FULL_REFRESH_INIT: u32 = 1 << 0;
    const FULL_REFRESH_TERRAIN_CHANGED: u32 = 1 << 1;
    const FULL_REFRESH_EXTENT_CHANGED: u32 = 1 << 2;
    const FULL_REFRESH_NO_OVERLAP: u32 = 1 << 3;

    let window_size_px = windows
        .single()
        .map(|w| {
            UVec2::new(
                w.width().max(1.0).round() as u32,
                w.height().max(1.0).round() as u32,
            )
        })
        .unwrap_or(DEFAULT_SCREEN_SIZE_PX);
    let (camera_pos, near_extent) = camera_q
        .single()
        .map(|(transform, projection)| {
            (
                transform.translation.xy(),
                compute_near_cache_extent_for_projection(window_size_px, projection),
            )
        })
        .unwrap_or((Vec2::ZERO, compute_near_cache_extent(window_size_px)));
    let near_extent_i = IVec2::new(
        saturating_u32_to_i32(near_extent.x),
        saturating_u32_to_i32(near_extent.y),
    );
    let extent_changed = near_extent != cache_state.near_extent_cells;
    if extent_changed {
        cache_state.near_extent_cells = near_extent;
        cache_state.ring_offset = IVec2::ZERO;
        cache_state.near_dirty_queue.clear();
        cache_state.far_dirty_queue.clear();
    } else {
        cache_state.ring_offset = normalize_ring_offset(cache_state.ring_offset, near_extent);
    }

    // Compute cache origin centred on camera position.
    let camera_cell = IVec2::new(
        (camera_pos.x / CELL_SIZE_M).floor() as i32,
        (camera_pos.y / CELL_SIZE_M).floor() as i32,
    );
    let new_origin = camera_cell - IVec2::new(near_extent_i.x / 2, near_extent_i.y / 2);
    let old_origin = IVec2::new(request.origin_x, request.origin_y);
    let origin_delta = if request.initialized {
        new_origin - old_origin
    } else {
        IVec2::ZERO
    };
    diagnostics.terrain_generation_origin_delta_x_frame = origin_delta.x;
    diagnostics.terrain_generation_origin_delta_y_frame = origin_delta.y;

    let origin_changed = origin_delta != IVec2::ZERO;

    let terrain_changed = request.terrain_render_version != terrain.terrain_render_version();
    if request.initialized && !terrain_changed && !origin_changed && !extent_changed {
        request.dirty = false;
        return;
    }

    // Full refresh is needed only when the new view no longer overlaps the previous cache,
    // or when terrain contents / cache extent changed.
    let no_overlap = request.initialized
        && (origin_delta.x.abs() >= near_extent_i.x || origin_delta.y.abs() >= near_extent_i.y);
    let full_refresh = !request.initialized || terrain_changed || extent_changed || no_overlap;
    let mut full_refresh_reason_bits = 0u32;
    if !request.initialized {
        full_refresh_reason_bits |= FULL_REFRESH_INIT;
    }
    if terrain_changed {
        full_refresh_reason_bits |= FULL_REFRESH_TERRAIN_CHANGED;
    }
    if extent_changed {
        full_refresh_reason_bits |= FULL_REFRESH_EXTENT_CHANGED;
    }
    if no_overlap {
        full_refresh_reason_bits |= FULL_REFRESH_NO_OVERLAP;
    }
    diagnostics.terrain_generation_full_refresh_frame = full_refresh;
    diagnostics.terrain_generation_full_refresh_reason_bits = full_refresh_reason_bits;

    if full_refresh {
        cache_state.ring_offset = IVec2::ZERO;
    } else {
        cache_state.ring_offset =
            normalize_ring_offset(cache_state.ring_offset + origin_delta, near_extent);
    }

    cache_state.cache_origin_world = new_origin;
    cache_state.lod_k = 0;
    cache_state.near_dirty_queue.clear();
    cache_state.far_dirty_queue.clear();

    let mut dirty_cells = Vec::new();
    if full_refresh {
        append_dirty_rect_cells(terrain.as_ref(), &mut dirty_cells, new_origin, near_extent);
        cache_state.near_dirty_queue.push_back(TerrainDirtyTile {
            _world_origin: new_origin,
            _extent_cells: near_extent,
        });
    } else {
        let dx = origin_delta.x;
        let dy = origin_delta.y;

        if dx > 0 {
            let w = dx.min(near_extent_i.x) as u32;
            let origin = IVec2::new(new_origin.x + near_extent_i.x - dx, new_origin.y);
            let extent = UVec2::new(w, near_extent.y);
            append_dirty_rect_cells(terrain.as_ref(), &mut dirty_cells, origin, extent);
            cache_state.near_dirty_queue.push_back(TerrainDirtyTile {
                _world_origin: origin,
                _extent_cells: extent,
            });
        } else if dx < 0 {
            let w = (-dx).min(near_extent_i.x) as u32;
            let origin = new_origin;
            let extent = UVec2::new(w, near_extent.y);
            append_dirty_rect_cells(terrain.as_ref(), &mut dirty_cells, origin, extent);
            cache_state.near_dirty_queue.push_back(TerrainDirtyTile {
                _world_origin: origin,
                _extent_cells: extent,
            });
        }

        let mut strip_x0 = new_origin.x;
        let mut strip_x1 = new_origin.x + near_extent_i.x;
        if dx > 0 {
            strip_x1 -= dx.min(near_extent_i.x);
        } else if dx < 0 {
            strip_x0 += (-dx).min(near_extent_i.x);
        }
        let strip_width = (strip_x1 - strip_x0).max(0) as u32;

        if strip_width > 0 {
            if dy > 0 {
                let h = dy.min(near_extent_i.y) as u32;
                let origin = IVec2::new(strip_x0, new_origin.y + near_extent_i.y - dy);
                let extent = UVec2::new(strip_width, h);
                append_dirty_rect_cells(terrain.as_ref(), &mut dirty_cells, origin, extent);
                cache_state.near_dirty_queue.push_back(TerrainDirtyTile {
                    _world_origin: origin,
                    _extent_cells: extent,
                });
            } else if dy < 0 {
                let h = (-dy).min(near_extent_i.y) as u32;
                let origin = IVec2::new(strip_x0, new_origin.y);
                let extent = UVec2::new(strip_width, h);
                append_dirty_rect_cells(terrain.as_ref(), &mut dirty_cells, origin, extent);
                cache_state.near_dirty_queue.push_back(TerrainDirtyTile {
                    _world_origin: origin,
                    _extent_cells: extent,
                });
            }
        }
    }

    if dirty_cells.is_empty() {
        request.dirty = false;
        return;
    }

    diagnostics.terrain_generation_eval_count_frame = dirty_cells.len() as u32;

    request.generation_enabled = terrain.generation_enabled();
    request.terrain_render_version = terrain.terrain_render_version();
    request.width = near_extent.x;
    request.height = near_extent.y;
    request.origin_x = new_origin.x;
    request.origin_y = new_origin.y;
    request.ring_offset_x = cache_state.ring_offset.x;
    request.ring_offset_y = cache_state.ring_offset.y;
    request.dirty_cells = dirty_cells;
    request.dirty = true;
    request.initialized = true;
}

// ── Render startup systems ────────────────────────────────────────────────────

fn init_terrain_near_gpu_resources(mut commands: Commands, render_device: Res<RenderDevice>) {
    let near_extent = compute_near_cache_extent(DEFAULT_SCREEN_SIZE_PX);
    let near_cache = make_near_cache(&render_device, near_extent);
    let dirty_cells_capacity = near_extent.x.saturating_mul(near_extent.y).max(1);
    let dirty_cells_buf = make_dirty_cells_buffer(&render_device, dirty_cells_capacity);

    // Initial params use origin (0, 0); overwritten immediately on first frame.
    let near_params = TerrainNearParams {
        cache_origin_x: 0,
        cache_origin_y: 0,
        cache_width: near_extent.x,
        cache_height: near_extent.y,
        ring_offset_x: 0,
        ring_offset_y: 0,
        generation_enabled: 1,
        override_none: TERRAIN_OVERRIDE_NONE,
        dirty_count: 0,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let near_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_near_params"),
            contents: bytemuck::bytes_of(&near_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );

    let compose_params = TerrainComposeParams {
        cell_size_m: CELL_SIZE_M,
        dot_size_m: CELL_SIZE_M / TERRAIN_DOTS_PER_CELL as f32,
        cache_origin_x: 0,
        cache_origin_y: 0,
        palette_seed: 0x5EED_7163,
        dots_per_cell: TERRAIN_DOTS_PER_CELL,
        ring_offset_x: 0,
        ring_offset_y: 0,
    };
    let compose_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_compose_params"),
            contents: bytemuck::bytes_of(&compose_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );

    commands.insert_resource(TerrainNearGpuResources {
        near_cache,
        dirty_cells_buf,
        dirty_cells_capacity,
        near_params_buf,
        compose_params_buf,
        pending_dispatch_frames: 0,
        pending_dispatch_cells: 0,
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
                    uniform_buffer_sized(false, core::num::NonZeroU64::new(48)),
                    storage_buffer_read_only_sized(false, None),
                    texture_storage_2d(TextureFormat::R16Uint, StorageTextureAccess::WriteOnly),
                ),
            ),
        )
    };

    let terrain_gen_shader = asset_server.load(TERRAIN_GEN_SHADER_PATH);
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
        _terrain_gen_shader: terrain_gen_shader,
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
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    near_pipeline: Res<TerrainNearUpdatePipeline>,
    mut dirty: ResMut<TerrainNearCacheDirty>,
) {
    if upload.dirty && !upload.dirty_cells.is_empty() {
        let requested_extent = UVec2::new(upload.width.max(1), upload.height.max(1));
        if resources.near_cache.extent_cells != requested_extent {
            resources.near_cache = make_near_cache(&render_device, requested_extent);
        }
        let dirty_count_full = upload.dirty_cells.len() as u32;
        let dirty_count = dirty_count_full.min(MAX_DIRTY_CELLS_PER_DISPATCH);
        if dirty_count_full > dirty_count {
            static DIRTY_CLAMP_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !DIRTY_CLAMP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                warn!(
                    "terrain_gpu: dirty cell count {} exceeds dispatch limit {}, clamping",
                    dirty_count_full, MAX_DIRTY_CELLS_PER_DISPATCH
                );
            }
        }
        if dirty_count > resources.dirty_cells_capacity {
            let new_capacity = dirty_count.next_power_of_two();
            resources.dirty_cells_buf = make_dirty_cells_buffer(&render_device, new_capacity);
            resources.dirty_cells_capacity = new_capacity;
        }

        // Upload dirty world-cell list.
        queue.write_buffer(
            &resources.dirty_cells_buf,
            0,
            cast_slice(&upload.dirty_cells[..dirty_count as usize]),
        );

        // Update compute-pass uniform.
        let near_params = TerrainNearParams {
            cache_origin_x: upload.origin_x,
            cache_origin_y: upload.origin_y,
            cache_width: upload.width,
            cache_height: upload.height,
            ring_offset_x: upload.ring_offset_x,
            ring_offset_y: upload.ring_offset_y,
            generation_enabled: u32::from(upload.generation_enabled),
            override_none: TERRAIN_OVERRIDE_NONE,
            dirty_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(
            &resources.near_params_buf,
            0,
            bytemuck::bytes_of(&near_params),
        );

        // Update fragment-pass uniform: cache origin for texture lookup.
        let compose_params = TerrainComposeParams {
            cell_size_m: CELL_SIZE_M,
            dot_size_m: CELL_SIZE_M / TERRAIN_DOTS_PER_CELL as f32,
            cache_origin_x: upload.origin_x,
            cache_origin_y: upload.origin_y,
            palette_seed: 0x5EED_7163,
            dots_per_cell: TERRAIN_DOTS_PER_CELL,
            ring_offset_x: upload.ring_offset_x,
            ring_offset_y: upload.ring_offset_y,
        };
        queue.write_buffer(
            &resources.compose_params_buf,
            0,
            bytemuck::bytes_of(&compose_params),
        );

        // Retry only while pipeline is still compiling; otherwise dispatch once.
        let pipeline_ready = pipeline_cache
            .get_compute_pipeline(near_pipeline.pipeline_id)
            .is_some();
        resources.pending_dispatch_frames = if pipeline_ready { 1 } else { 32 };
        resources.pending_dispatch_cells = dirty_count;
    }
    if resources.pending_dispatch_frames > 0 && resources.pending_dispatch_cells > 0 {
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
            .init_resource::<TerrainCacheState>()
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

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_SCREEN_SIZE_PX, MAX_DIRTY_CELLS_PER_DISPATCH, TerrainCacheState,
        compute_near_cache_extent, compute_near_cache_extent_from_viewport_world,
        world_cell_to_texture_index,
    };
    use bevy::math::{IVec2, UVec2, Vec2};

    #[test]
    fn near_extent_uses_default_camera_viewport_formula() {
        let extent = compute_near_cache_extent(DEFAULT_SCREEN_SIZE_PX);
        assert_eq!(extent, UVec2::new(135, 76));
    }

    #[test]
    fn world_cell_to_texture_index_applies_ring_wrap() {
        let idx = world_cell_to_texture_index(
            IVec2::new(11, 21),
            IVec2::new(10, 20),
            IVec2::new(3, 2),
            UVec2::new(8, 4),
        );
        assert_eq!(idx, Some(28));
    }

    #[test]
    fn cache_state_default_initializes_near_extent() {
        let state = TerrainCacheState::default();
        assert_eq!(state.near_extent_cells, UVec2::new(135, 76));
    }

    #[test]
    fn near_extent_is_clamped_to_dispatch_limit() {
        let extent = compute_near_cache_extent_from_viewport_world(Vec2::splat(1_000_000.0));
        let area = u64::from(extent.x) * u64::from(extent.y);
        assert!(area <= u64::from(MAX_DIRTY_CELLS_PER_DISPATCH));
    }
}
