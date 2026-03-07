/// GPU-resident terrain Near-cache pipeline.
///
/// Architecture:
///   Main graph:   `TerrainNearUpdateLabel` (compute) → writes R16Uint material-ID texture
///                 `TerrainFarUpdateLabel`  (compute) → aggregates Near into RGBA8Uint Far cache
///   Core2d graph: `TerrainComposeLabel`    (fragment ViewNode) → reads texture → screen
///
/// Data flow:
///   CPU TerrainWorld → `TerrainNearUpdateRequest` (main world, dirty cells on pan/edit)
///     → ExtractResource → render world
///     → `prepare_terrain_near_uploads` (writes dirty-cell SSBO, sets dirty flag)
///     → `TerrainNearUpdateNode` (compute, dispatches only when dirty)
///     → `near_texture` (R16Uint, GPU-resident, persistent between frames)
///     → `TerrainFarUpdateNode`  (compute, full refresh on Near updates)
///     → `TerrainComposeNode` (fragment ViewNode, always reads cached texture)
///     → `clear_terrain_cache_dirty` (Cleanup) → dirty = false until next terrain change
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

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
    storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_storage_2d,
    uniform_buffer, uniform_buffer_sized,
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
use crate::params::ActiveRenderParams;
use crate::physics::material::TerrainMaterial;
use crate::physics::state::SimUpdateSet;
use crate::physics::world::constants::CELL_SIZE_M;
use crate::physics::world::terrain::{CHUNK_SIZE, CHUNK_SIZE_I32, TerrainCell, TerrainWorld};

use super::{TerrainRenderDiagnostics, water_dot_gpu::WaterDotGpuLabel};

const TERRAIN_NEAR_UPDATE_SHADER_PATH: &str = "shaders/render/terrain_near_update.wgsl";
const TERRAIN_OVERRIDE_APPLY_SHADER_PATH: &str = "shaders/render/terrain_override_apply.wgsl";
const TERRAIN_FAR_UPDATE_SHADER_PATH: &str = "shaders/render/terrain_far_update.wgsl";
const TERRAIN_GEN_SHADER_PATH: &str = "shaders/render/terrain_gen.wgsl";
const TERRAIN_CHUNK_GENERATE_SHADER_PATH: &str = "shaders/render/terrain_chunk_generate.wgsl";
const TERRAIN_COMPOSE_SHADER_PATH: &str = "shaders/render/terrain_compose.wgsl";
const TERRAIN_DOTS_PER_CELL: u32 = 8;
const NEAR_UPDATE_WORKGROUP: u32 = 64;
const OVERRIDE_APPLY_WORKGROUP: u32 = 64;
const CHUNK_GENERATE_WORKGROUP: u32 = 64;
const FAR_UPDATE_WORKGROUP_X: u32 = 8;
const FAR_UPDATE_WORKGROUP_Y: u32 = 8;
const FAR_LOD_OFFSET: i32 = 3;
const FAR_MIN_DOWNSAMPLE: u32 = 1;
const FAR_MARGIN_FACTOR: f32 = 2.5;
const FAR_QUALITY_DIVISOR: f32 = 2.0;
const FAR_QUALITY_HEADROOM: f32 = 1.1;
const BACK_MIN_SCREEN_RESOLUTION_DIVISOR: f32 = 2.0;
const BACK_SCALE_MULTIPLIER: f32 = 2.0;
const BACK_ATMOSPHERE_TINT: f32 = 0.35;
const SKY_COLOR_R: f32 = 164.0;
const SKY_COLOR_G: f32 = 208.0;
const SKY_COLOR_B: f32 = 255.0;
const MAX_DISPATCH_GROUPS_X: u32 = 65_535;
const MAX_DIRTY_CELLS_PER_DISPATCH: u32 = NEAR_UPDATE_WORKGROUP * MAX_DISPATCH_GROUPS_X;
const MAX_OVERRIDE_RUNS_PER_FRAME: usize = 8_192;
const MAX_OVERRIDE_RUNS_PER_DISPATCH: u32 = MAX_OVERRIDE_RUNS_PER_FRAME as u32;
const TERRAIN_OVERRIDE_NONE: u32 = 0xFFFF;
const NEAR_MARGIN_FACTOR: f32 = 1.35;
const NEAR_QUALITY_DIVISOR: f32 = 1.0;
const NEAR_TEX_MIN_CELLS: u32 = 1;
const NEAR_TEX_MAX_CELLS: u32 = 2048;
const FAR_TEX_MIN_CELLS: u32 = 1;
const FAR_TEX_MAX_CELLS: u32 = 4096;
const DEFAULT_SCREEN_SIZE_PX: UVec2 = UVec2::new(1280, 720);
const DEFAULT_CAMERA_VIEWPORT_HEIGHT_M: f32 = 14.0;
const TERRAIN_CHUNK_CELLS: usize = CHUNK_SIZE * CHUNK_SIZE;

#[derive(Clone, Copy, Debug)]
struct TerrainRuntimeSettings {
    near_margin_factor: f32,
    near_quality_divisor: f32,
    far_margin_factor: f32,
    far_quality_divisor: f32,
    far_quality_headroom: f32,
    far_min_downsample: u32,
    back_min_screen_resolution_divisor: f32,
    back_scale_multiplier: f32,
    back_atmosphere_tint: f32,
    sky_color_r: f32,
    sky_color_g: f32,
    sky_color_b: f32,
    terrain_palette_seed: u32,
}

impl Default for TerrainRuntimeSettings {
    fn default() -> Self {
        Self {
            near_margin_factor: NEAR_MARGIN_FACTOR,
            near_quality_divisor: NEAR_QUALITY_DIVISOR,
            far_margin_factor: FAR_MARGIN_FACTOR,
            far_quality_divisor: FAR_QUALITY_DIVISOR,
            far_quality_headroom: FAR_QUALITY_HEADROOM,
            far_min_downsample: FAR_MIN_DOWNSAMPLE,
            back_min_screen_resolution_divisor: BACK_MIN_SCREEN_RESOLUTION_DIVISOR,
            back_scale_multiplier: BACK_SCALE_MULTIPLIER,
            back_atmosphere_tint: BACK_ATMOSPHERE_TINT,
            sky_color_r: SKY_COLOR_R,
            sky_color_g: SKY_COLOR_G,
            sky_color_b: SKY_COLOR_B,
            terrain_palette_seed: 0x5EED_7163,
        }
    }
}

fn terrain_runtime_settings_from_active(
    active: Option<&ActiveRenderParams>,
) -> TerrainRuntimeSettings {
    let mut settings = TerrainRuntimeSettings::default();
    let Some(active) = active else {
        return settings;
    };
    let terrain = &active.0.terrain;
    settings.near_margin_factor = terrain.lod.near_margin_factor;
    settings.near_quality_divisor = terrain.lod.near_quality_divisor;
    settings.far_margin_factor = terrain.lod.far_margin_factor;
    settings.far_quality_divisor = terrain.lod.far_quality_divisor;
    settings.far_quality_headroom = terrain.lod.far_quality_headroom;
    settings.far_min_downsample = terrain.lod.far_min_downsample.max(1);
    settings.back_min_screen_resolution_divisor =
        terrain.back.min_screen_resolution_divisor.max(1.0);
    settings.back_scale_multiplier = terrain.back.scale_multiplier.max(0.25);
    settings.back_atmosphere_tint = terrain.back.atmosphere_tint.clamp(0.0, 1.0);
    settings.sky_color_r = terrain.sky_color.r.clamp(0.0, 255.0);
    settings.sky_color_g = terrain.sky_color.g.clamp(0.0, 255.0);
    settings.sky_color_b = terrain.sky_color.b.clamp(0.0, 255.0);
    settings.terrain_palette_seed = terrain.palette_seed;
    settings
}

#[derive(Clone, Copy, Debug)]
struct TerrainDirtyTile {
    pub _world_origin: IVec2,
    pub _extent_cells: UVec2,
}

#[derive(Resource, Debug)]
struct TerrainCacheState {
    pub cache_origin_world: IVec2,
    pub ring_offset: IVec2,
    pub lod_k: i32,
    pub near_dirty_queue: VecDeque<TerrainDirtyTile>,
    pub far_dirty_queue: VecDeque<TerrainDirtyTile>,
    pub near_extent_cells: UVec2,
    override_run_queue: VecDeque<TerrainOverrideRunGpu>,
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
            override_run_queue: VecDeque::new(),
        }
    }
}

fn default_viewport_world(screen_size_px: UVec2) -> Vec2 {
    let size = screen_size_px.max(UVec2::splat(1)).as_vec2();
    let aspect = size.x / size.y;
    Vec2::new(
        DEFAULT_CAMERA_VIEWPORT_HEIGHT_M * aspect,
        DEFAULT_CAMERA_VIEWPORT_HEIGHT_M,
    )
}

fn projection_viewport_world(projection: &Projection, viewport_size_px: Vec2) -> Option<Vec2> {
    let Projection::Orthographic(ortho) = projection else {
        return None;
    };
    let width = viewport_size_px.x.max(1.0);
    let height = viewport_size_px.y.max(1.0);
    let (projection_width, projection_height) = match ortho.scaling_mode {
        bevy::camera::ScalingMode::WindowSize => (width, height),
        bevy::camera::ScalingMode::AutoMin {
            min_width,
            min_height,
        } => {
            if width * min_height > min_width * height {
                (width * min_height / height, min_height)
            } else {
                (min_width, height * min_width / width)
            }
        }
        bevy::camera::ScalingMode::AutoMax {
            max_width,
            max_height,
        } => {
            if width * max_height < max_width * height {
                (width * max_height / height, max_height)
            } else {
                (max_width, height * max_width / width)
            }
        }
        bevy::camera::ScalingMode::FixedVertical { viewport_height } => {
            (width * viewport_height / height, viewport_height)
        }
        bevy::camera::ScalingMode::FixedHorizontal { viewport_width } => {
            (viewport_width, height * viewport_width / width)
        }
        bevy::camera::ScalingMode::Fixed { width, height } => (width, height),
    };
    let viewport_world =
        Vec2::new(projection_width, projection_height) * ortho.scale.abs().max(1.0e-6);
    if viewport_world.x <= 0.0 || viewport_world.y <= 0.0 {
        return None;
    }
    Some(viewport_world)
}

fn near_target_extent_cells_from_viewport_world_with_settings(
    viewport_world_m: Vec2,
    settings: TerrainRuntimeSettings,
) -> Vec2 {
    let viewport_world_m = viewport_world_m.max(Vec2::splat(CELL_SIZE_M.max(1e-6)));
    let viewport_cells = viewport_world_m / CELL_SIZE_M;
    viewport_cells * (settings.near_margin_factor / settings.near_quality_divisor.max(1e-6))
}

fn compute_near_cache_extent_with_settings(
    screen_size_px: UVec2,
    settings: TerrainRuntimeSettings,
) -> UVec2 {
    compute_near_cache_extent_from_viewport_world_with_settings(
        default_viewport_world(screen_size_px),
        settings,
    )
}

fn compute_near_cache_extent(screen_size_px: UVec2) -> UVec2 {
    compute_near_cache_extent_with_settings(screen_size_px, TerrainRuntimeSettings::default())
}

fn compute_near_cache_extent_from_viewport_world_with_settings(
    viewport_world_m: Vec2,
    settings: TerrainRuntimeSettings,
) -> UVec2 {
    let scaled =
        near_target_extent_cells_from_viewport_world_with_settings(viewport_world_m, settings);
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

#[cfg(test)]
fn compute_near_cache_extent_from_viewport_world(viewport_world_m: Vec2) -> UVec2 {
    compute_near_cache_extent_from_viewport_world_with_settings(
        viewport_world_m,
        TerrainRuntimeSettings::default(),
    )
}

fn compute_far_cache_base_extent_with_settings(
    screen_size_px: UVec2,
    settings: TerrainRuntimeSettings,
) -> UVec2 {
    let screen = screen_size_px.max(UVec2::splat(1)).as_vec2();
    let scaled = screen
        * (settings.far_margin_factor * settings.far_quality_headroom
            / settings.far_quality_divisor.max(1e-6));
    UVec2::new(
        scaled
            .x
            .ceil()
            .max(FAR_TEX_MIN_CELLS as f32)
            .min(FAR_TEX_MAX_CELLS as f32) as u32,
        scaled
            .y
            .ceil()
            .max(FAR_TEX_MIN_CELLS as f32)
            .min(FAR_TEX_MAX_CELLS as f32) as u32,
    )
}

#[cfg(test)]
fn compute_far_cache_base_extent(screen_size_px: UVec2) -> UVec2 {
    compute_far_cache_base_extent_with_settings(screen_size_px, TerrainRuntimeSettings::default())
}

fn ceil_to_power_of_two(v: u32) -> u32 {
    if v <= 1 {
        return 1;
    }
    v.checked_next_power_of_two().unwrap_or(1 << 31)
}

fn compute_far_downsample_for_viewport_with_settings(
    viewport_world_m: Vec2,
    far_extent: UVec2,
    settings: TerrainRuntimeSettings,
) -> u32 {
    let viewport_world_m = viewport_world_m.max(Vec2::splat(CELL_SIZE_M.max(1e-6)));
    let viewport_cells = viewport_world_m / CELL_SIZE_M;
    let far_extent = far_extent.max(UVec2::splat(1)).as_vec2();
    let required_x = (viewport_cells.x * settings.far_margin_factor / far_extent.x)
        .ceil()
        .max(1.0) as u32;
    let required_y = (viewport_cells.y * settings.far_margin_factor / far_extent.y)
        .ceil()
        .max(1.0) as u32;
    let required = required_x.max(required_y).max(1);
    ceil_to_power_of_two(required.max(settings.far_min_downsample.max(1)))
}

#[cfg(test)]
fn compute_far_downsample_for_viewport(viewport_world_m: Vec2, far_extent: UVec2) -> u32 {
    compute_far_downsample_for_viewport_with_settings(
        viewport_world_m,
        far_extent,
        TerrainRuntimeSettings::default(),
    )
}

fn compute_back_display_scale(
    front_mpp_cells_per_px: f32,
    settings: TerrainRuntimeSettings,
) -> f32 {
    let front_mpp_cells_per_px = front_mpp_cells_per_px.max(1.0e-6);
    let scale_excess = settings.back_scale_multiplier.max(0.0)
        / (settings.back_min_screen_resolution_divisor.max(1.0) * front_mpp_cells_per_px);
    1.0 + scale_excess
}

fn compute_back_downsample_for_viewport_with_settings(
    viewport_world_m: Vec2,
    back_extent: UVec2,
    settings: TerrainRuntimeSettings,
) -> u32 {
    // Back downsample follows the same rule as Far (power-of-two coverage),
    // but is computed independently to keep the path decoupled.
    compute_far_downsample_for_viewport_with_settings(viewport_world_m, back_extent, settings)
}

fn compute_far_origin_world(camera_cell: IVec2, far_extent: UVec2, far_downsample: u32) -> IVec2 {
    let downsample = far_downsample.max(1);
    let downsample_i = saturating_u32_to_i32(downsample).max(1);
    let world_extent = IVec2::new(
        saturating_u32_to_i32(far_extent.x.saturating_mul(downsample)),
        saturating_u32_to_i32(far_extent.y.saturating_mul(downsample)),
    );
    let origin = camera_cell - IVec2::new(world_extent.x / 2, world_extent.y / 2);
    IVec2::new(
        origin.x.div_euclid(downsample_i) * downsample_i,
        origin.y.div_euclid(downsample_i) * downsample_i,
    )
}

fn compute_near_lod_k_from_viewport_world(viewport_world_m: Vec2) -> i32 {
    let viewport_world_height = viewport_world_m.y.abs().max(1e-6);
    let ratio = (viewport_world_height / DEFAULT_CAMERA_VIEWPORT_HEIGHT_M.max(1e-6)).max(1.0);
    ratio.log2().floor() as i32
}

fn compute_near_cache_extent_for_projection(
    screen_size_px: UVec2,
    projection: &Projection,
    window_size_px: Vec2,
    settings: TerrainRuntimeSettings,
) -> UVec2 {
    projection_viewport_world(projection, window_size_px)
        .map(|viewport| {
            compute_near_cache_extent_from_viewport_world_with_settings(viewport, settings)
        })
        .unwrap_or_else(|| compute_near_cache_extent_with_settings(screen_size_px, settings))
}

fn near_covers_viewport_with_margin(
    near_extent: UVec2,
    viewport_world_m: Vec2,
    settings: TerrainRuntimeSettings,
) -> bool {
    let target =
        near_target_extent_cells_from_viewport_world_with_settings(viewport_world_m, settings);
    near_extent.x as f32 + 0.5 >= target.x.ceil() && near_extent.y as f32 + 0.5 >= target.y.ceil()
}

fn normalize_ring_offset(ring_offset: IVec2, extent: UVec2) -> IVec2 {
    IVec2::new(
        ring_offset.x.rem_euclid(extent.x.max(1) as i32),
        ring_offset.y.rem_euclid(extent.y.max(1) as i32),
    )
}

#[cfg(test)]
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

#[derive(Clone, Copy, Debug)]
struct TerrainGeneratedChunkInflight {
    request_id: u64,
    chunk_coord: IVec2,
}

#[derive(Resource, Debug, Default)]
pub struct TerrainGeneratedChunkCache {
    generated_chunks: HashMap<IVec2, [u16; TERRAIN_CHUNK_CELLS]>,
    request_queue: VecDeque<IVec2>,
    pending_chunks: HashSet<IVec2>,
    inflight: Option<TerrainGeneratedChunkInflight>,
    next_request_id: u64,
}

impl TerrainGeneratedChunkCache {
    pub fn material_ids_for_chunk(
        &self,
        chunk_coord: IVec2,
    ) -> Option<&[u16; TERRAIN_CHUNK_CELLS]> {
        self.generated_chunks.get(&chunk_coord)
    }

    pub fn enqueue_chunk_request(&mut self, chunk_coord: IVec2) {
        if self.generated_chunks.contains_key(&chunk_coord)
            || self.pending_chunks.contains(&chunk_coord)
            || self.inflight.map(|req| req.chunk_coord) == Some(chunk_coord)
        {
            return;
        }
        self.request_queue.push_back(chunk_coord);
        self.pending_chunks.insert(chunk_coord);
    }

    pub fn enqueue_prefetch_square(&mut self, center_chunk: IVec2, radius_chunks: i32) {
        let r = radius_chunks.max(0);
        for cy in (center_chunk.y - r)..=(center_chunk.y + r) {
            for cx in (center_chunk.x - r)..=(center_chunk.x + r) {
                self.enqueue_chunk_request(IVec2::new(cx, cy));
            }
        }
    }

    pub fn cached_chunk_coords(&self) -> Vec<IVec2> {
        let mut coords: Vec<_> = self.generated_chunks.keys().copied().collect();
        coords.sort_by_key(|coord| (coord.y, coord.x));
        coords
    }

    fn begin_next_request(&mut self) -> Option<(u64, IVec2)> {
        if self.inflight.is_some() {
            return None;
        }
        while let Some(chunk_coord) = self.request_queue.pop_front() {
            if self.generated_chunks.contains_key(&chunk_coord) {
                self.pending_chunks.remove(&chunk_coord);
                continue;
            }
            let request_id = self.next_request_id;
            self.next_request_id = self.next_request_id.wrapping_add(1);
            self.inflight = Some(TerrainGeneratedChunkInflight {
                request_id,
                chunk_coord,
            });
            return Some((request_id, chunk_coord));
        }
        None
    }

    fn finish_request(&mut self, request_id: u64, chunk_coord: IVec2) -> bool {
        let Some(inflight) = self.inflight else {
            return false;
        };
        if inflight.request_id != request_id || inflight.chunk_coord != chunk_coord {
            return false;
        }
        self.inflight = None;
        self.pending_chunks.remove(&chunk_coord);
        true
    }

    fn cache_generated_chunk(&mut self, chunk_coord: IVec2, cells: [u16; TERRAIN_CHUNK_CELLS]) {
        self.generated_chunks.insert(chunk_coord, cells);
    }
}

#[derive(Resource, Clone, Debug, Default)]
struct TerrainGeneratedChunkReadbackRequest {
    active: bool,
    request_id: u64,
    chunk_coord: IVec2,
    generation_enabled: bool,
}

impl ExtractResource for TerrainGeneratedChunkReadbackRequest {
    type Source = TerrainGeneratedChunkReadbackRequest;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[derive(Clone, Debug)]
struct TerrainGeneratedChunkReadbackPayload {
    request_id: u64,
    chunk_coord: IVec2,
    cells: Vec<u16>,
}

#[derive(Resource, Clone, Default)]
struct TerrainGeneratedChunkReadbackResult {
    inner: Arc<Mutex<Option<TerrainGeneratedChunkReadbackPayload>>>,
}

impl TerrainGeneratedChunkReadbackResult {
    fn take(&self) -> Option<TerrainGeneratedChunkReadbackPayload> {
        if let Ok(mut guard) = self.inner.lock() {
            guard.take()
        } else {
            None
        }
    }

    fn store(&self, payload: TerrainGeneratedChunkReadbackPayload) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(payload);
        }
    }
}

// ── Render labels ─────────────────────────────────────────────────────────────

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TerrainNearUpdateLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TerrainFarUpdateLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TerrainOverrideApplyLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TerrainBackUpdateLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TerrainComposeLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TerrainChunkGenerateLabel;

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

/// Uniform for `TerrainOverrideApply` compute pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainOverrideParams {
    cache_origin_x: i32,
    cache_origin_y: i32,
    cache_width: u32,
    cache_height: u32,
    ring_offset_x: i32,
    ring_offset_y: i32,
    override_none: u32,
    run_count: u32,
    chunk_size_i32: i32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
const _: () = assert!(std::mem::size_of::<TerrainOverrideParams>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainOverrideRunGpu {
    chunk_x: i32,
    chunk_y: i32,
    start_index: u32,
    run_length: u32,
    material: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
const _: () = assert!(std::mem::size_of::<TerrainOverrideRunGpu>() == 32);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainFarOverrideHashEntry {
    chunk_x: i32,
    chunk_y: i32,
    chunk_index: i32,
    _pad0: i32,
}
const _: () = assert!(std::mem::size_of::<TerrainFarOverrideHashEntry>() == 16);

/// Uniform for `TerrainFarUpdate` compute pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainFarParams {
    far_origin_x: i32,
    far_origin_y: i32,
    far_width: u32,
    far_height: u32,
    far_downsample: u32,
    generation_enabled: u32,
    near_origin_x: i32,
    near_origin_y: i32,
    near_width: u32,
    near_height: u32,
    ring_offset_x: i32,
    ring_offset_y: i32,
    override_none: u32,
    override_hash_mask: u32,
    override_hash_len: u32,
    override_chunk_count: u32,
    override_chunk_size_i32: i32,
    near_cache_enabled: u32,
    _pad0: u32,
    _pad1: u32,
}
const _: () = assert!(std::mem::size_of::<TerrainFarParams>() == 80);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainChunkGenerateParams {
    chunk_x: i32,
    chunk_y: i32,
    chunk_size_i32: i32,
    generation_enabled: u32,
}
const _: () = assert!(std::mem::size_of::<TerrainChunkGenerateParams>() == 16);

/// Uniform for `TerrainCompose` fragment pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TerrainComposeParams {
    cell_size_m: f32,
    dot_size_m: f32,
    back_min_screen_resolution_divisor: f32,
    back_scale_multiplier: f32,
    front_mpp_cells_per_px: f32,
    back_atmosphere_tint: f32,
    far_origin_x: i32,
    far_origin_y: i32,
    back_origin_x: i32,
    back_origin_y: i32,
    near_origin_x: i32,
    near_origin_y: i32,
    palette_seed: u32,
    dots_per_cell: u32,
    ring_offset_x: i32,
    ring_offset_y: i32,
    near_enabled: u32,
    far_downsample: u32,
    back_downsample: u32,
    _pad1: u32,
    sky_color_r: f32,
    sky_color_g: f32,
    sky_color_b: f32,
    _pad2: f32,
}
const _: () = assert!(std::mem::size_of::<TerrainComposeParams>() == 96);

// ── GPU resources (render world) ──────────────────────────────────────────────

pub(crate) struct TerrainNearGpuCache {
    /// R16Uint persistent texture: material IDs at cell resolution.
    _near_texture: Texture,
    pub(crate) near_texture_view: TextureView,
    extent_cells: UVec2,
}

struct TerrainFarGpuCache {
    /// RGBA8Uint persistent texture:
    /// R = top1 material id, G = top1 weight, B = solid fraction, A = unused.
    _far_texture: Texture,
    far_texture_view: TextureView,
    extent_cells: UVec2,
}

struct TerrainBackGpuCache {
    /// RGBA8Uint persistent texture:
    /// R = top1 material id, G = top1 weight, B = solid fraction, A = unused.
    _back_texture: Texture,
    back_texture_view: TextureView,
    extent_cells: UVec2,
}

#[derive(Resource)]
pub(crate) struct TerrainNearGpuResources {
    pub(crate) near_cache: TerrainNearGpuCache,
    pub(crate) override_cache: TerrainNearGpuCache,
    far_cache: TerrainFarGpuCache,
    back_cache: TerrainBackGpuCache,
    /// Storage buffer: dirty world cells to update in the current dispatch.
    dirty_cells_buf: Buffer,
    dirty_cells_capacity: u32,
    /// Storage buffer: RLE-compressed terrain override runs.
    override_runs_buf: Buffer,
    override_runs_capacity: u32,
    /// Storage buffer: hash table mapping chunk coord -> override chunk index.
    far_override_hash_buf: Buffer,
    far_override_hash_capacity: u32,
    /// Storage buffer: packed per-chunk override cell ids (`CHUNK_SIZE * CHUNK_SIZE`).
    far_override_cells_buf: Buffer,
    far_override_cells_capacity: u32,
    /// Uniform buffer for compute pass (TerrainNearParams).
    near_params_buf: Buffer,
    /// Uniform buffer for override apply pass (TerrainOverrideParams).
    override_params_buf: Buffer,
    /// Uniform buffer for far compute pass (TerrainFarParams).
    far_params_buf: Buffer,
    /// Uniform buffer for back compute pass (TerrainFarParams).
    back_params_buf: Buffer,
    /// Uniform buffer for fragment pass (TerrainComposeParams).
    pub(crate) compose_params_buf: Buffer,
    /// How many more frames to keep the dirty flag set, used to retry when the
    /// pipeline is still compiling on the first frame data arrives.
    pending_dispatch_frames: u32,
    pending_dispatch_cells: u32,
    pending_override_dispatch_frames: u32,
    pending_override_dispatch_runs: u32,
    pending_far_dispatch_frames: u32,
    pending_far_dispatch_width: u32,
    pending_far_dispatch_height: u32,
    pending_back_dispatch_frames: u32,
    pending_back_dispatch_width: u32,
    pending_back_dispatch_height: u32,
}

#[derive(Resource)]
struct TerrainChunkGenerateGpuResources {
    params_buf: Buffer,
    generated_buf: Buffer,
    readback_buf: Buffer,
}

/// Set `true` when override data needs to be dispatched via compute; cleared in Cleanup.
#[derive(Resource, Default)]
struct TerrainNearCacheDirty(bool);

/// Set `true` when compressed override runs need application; cleared in Cleanup.
#[derive(Resource, Default)]
struct TerrainOverrideCacheDirty(bool);

/// Set `true` when far cache needs refresh via compute; cleared in Cleanup.
#[derive(Resource, Default)]
struct TerrainFarCacheDirty(bool);

/// Set `true` when back cache needs refresh via compute; cleared in Cleanup.
#[derive(Resource, Default)]
struct TerrainBackCacheDirty(bool);

#[derive(Resource, Default)]
struct TerrainChunkGenerateDirty(bool);

#[derive(Resource)]
struct TerrainChunkReadbackState {
    mapped_ready: Arc<AtomicBool>,
    inner: Mutex<TerrainChunkReadbackStateInner>,
}

#[derive(Default)]
struct TerrainChunkReadbackStateInner {
    mapped: bool,
    pending_dispatch: Option<TerrainGeneratedChunkInflight>,
    pending_map: Option<TerrainGeneratedChunkInflight>,
    mapped_meta: Option<TerrainGeneratedChunkInflight>,
}

impl Default for TerrainChunkReadbackState {
    fn default() -> Self {
        Self {
            mapped_ready: Arc::new(AtomicBool::new(false)),
            inner: Mutex::new(TerrainChunkReadbackStateInner::default()),
        }
    }
}

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

fn make_override_cache(render_device: &RenderDevice, extent: UVec2) -> TerrainNearGpuCache {
    let override_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("terrain_near_override_cache"),
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
    let override_texture_view = override_texture.create_view(&TextureViewDescriptor::default());
    TerrainNearGpuCache {
        _near_texture: override_texture,
        near_texture_view: override_texture_view,
        extent_cells: extent,
    }
}

fn make_far_cache(render_device: &RenderDevice, far_extent: UVec2) -> TerrainFarGpuCache {
    let far_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("terrain_far_cache"),
        size: Extent3d {
            width: far_extent.x,
            height: far_extent.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Uint,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let far_texture_view = far_texture.create_view(&TextureViewDescriptor::default());
    TerrainFarGpuCache {
        _far_texture: far_texture,
        far_texture_view,
        extent_cells: far_extent,
    }
}

fn make_back_cache(render_device: &RenderDevice, back_extent: UVec2) -> TerrainBackGpuCache {
    let back_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("terrain_back_cache"),
        size: Extent3d {
            width: back_extent.x,
            height: back_extent.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Uint,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let back_texture_view = back_texture.create_view(&TextureViewDescriptor::default());
    TerrainBackGpuCache {
        _back_texture: back_texture,
        back_texture_view,
        extent_cells: back_extent,
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

fn make_override_runs_buffer(render_device: &RenderDevice, runs_capacity: u32) -> Buffer {
    let cap = runs_capacity.max(1);
    render_device.create_buffer(&BufferDescriptor {
        label: Some("terrain_override_runs"),
        size: cap as u64 * std::mem::size_of::<TerrainOverrideRunGpu>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_far_override_hash_buffer(render_device: &RenderDevice, capacity: u32) -> Buffer {
    let cap = capacity.max(1);
    render_device.create_buffer(&BufferDescriptor {
        label: Some("terrain_far_override_hash"),
        size: cap as u64 * std::mem::size_of::<TerrainFarOverrideHashEntry>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_far_override_cells_buffer(render_device: &RenderDevice, capacity: u32) -> Buffer {
    let cap = capacity.max(1);
    render_device.create_buffer(&BufferDescriptor {
        label: Some("terrain_far_override_cells"),
        size: cap as u64 * std::mem::size_of::<u32>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_chunk_generated_buffer(render_device: &RenderDevice) -> Buffer {
    render_device.create_buffer(&BufferDescriptor {
        label: Some("terrain_chunk_generated_materials"),
        size: (TERRAIN_CHUNK_CELLS * std::mem::size_of::<u32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn make_chunk_readback_buffer(render_device: &RenderDevice) -> Buffer {
    render_device.create_buffer(&BufferDescriptor {
        label: Some("terrain_chunk_generated_readback"),
        size: (TERRAIN_CHUNK_CELLS * std::mem::size_of::<u32>()) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
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
                BindingResource::TextureView(&resources.override_cache.near_texture_view),
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

#[derive(Resource)]
struct TerrainFarUpdatePipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
    _terrain_gen_shader: Handle<Shader>,
}

#[derive(Resource)]
struct TerrainOverrideApplyPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
}

#[derive(Resource)]
struct TerrainChunkGeneratePipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline_id: CachedComputePipelineId,
    _terrain_gen_shader: Handle<Shader>,
}

#[derive(Default)]
struct TerrainOverrideApplyNode;

impl Node for TerrainOverrideApplyNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if !world
            .get_resource::<TerrainOverrideCacheDirty>()
            .is_some_and(|d| d.0)
        {
            return Ok(());
        }
        let Some(resources) = world.get_resource::<TerrainNearGpuResources>() else {
            return Ok(());
        };
        let Some(pipeline_res) = world.get_resource::<TerrainOverrideApplyPipeline>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline_id) else {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                match pipeline_cache.get_compute_pipeline_state(pipeline_res.pipeline_id) {
                    CachedPipelineState::Err(e) => {
                        warn!("terrain_gpu: override_apply pipeline error: {e}")
                    }
                    s => warn!("terrain_gpu: override_apply pipeline not ready: {s:?}"),
                }
            }
            return Ok(());
        };

        let run_count = resources.pending_override_dispatch_runs;
        if run_count == 0 {
            return Ok(());
        }
        let layout = pipeline_cache.get_bind_group_layout(&pipeline_res.bind_group_layout);
        let bind_group = render_context.render_device().create_bind_group(
            "terrain_override_apply_bg",
            &layout,
            &BindGroupEntries::sequential((
                resources.override_params_buf.as_entire_binding(),
                resources.override_runs_buf.as_entire_binding(),
                BindingResource::TextureView(&resources.override_cache.near_texture_view),
            )),
        );

        let dispatch = run_count.div_ceil(OVERRIDE_APPLY_WORKGROUP);
        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("terrain_override_apply"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
        Ok(())
    }
}

#[derive(Default)]
struct TerrainChunkGenerateNode;

impl Node for TerrainChunkGenerateNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if !world
            .get_resource::<TerrainChunkGenerateDirty>()
            .is_some_and(|d| d.0)
        {
            return Ok(());
        }
        let Some(resources) = world.get_resource::<TerrainChunkGenerateGpuResources>() else {
            return Ok(());
        };
        let Some(pipeline_res) = world.get_resource::<TerrainChunkGeneratePipeline>() else {
            return Ok(());
        };
        let readback_state = world.resource::<TerrainChunkReadbackState>();
        let Some(meta) = readback_state
            .inner
            .lock()
            .ok()
            .and_then(|mut inner| inner.pending_dispatch.take())
        else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline_id) else {
            if let Ok(mut inner) = readback_state.inner.lock() {
                inner.pending_dispatch = Some(meta);
            }
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                match pipeline_cache.get_compute_pipeline_state(pipeline_res.pipeline_id) {
                    CachedPipelineState::Err(e) => {
                        warn!("terrain_gpu: chunk_generate pipeline error: {e}")
                    }
                    s => warn!("terrain_gpu: chunk_generate pipeline not ready: {s:?}"),
                }
            }
            return Ok(());
        };

        let layout = pipeline_cache.get_bind_group_layout(&pipeline_res.bind_group_layout);
        let bind_group = render_context.render_device().create_bind_group(
            "terrain_chunk_generate_bg",
            &layout,
            &BindGroupEntries::sequential((
                resources.params_buf.as_entire_binding(),
                resources.generated_buf.as_entire_binding(),
            )),
        );

        let cell_count = TERRAIN_CHUNK_CELLS as u32;
        let dispatch = cell_count.div_ceil(CHUNK_GENERATE_WORKGROUP);
        let byte_size = (TERRAIN_CHUNK_CELLS * std::mem::size_of::<u32>()) as u64;
        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("terrain_chunk_generate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch, 1, 1);
        drop(pass);
        encoder.copy_buffer_to_buffer(
            &resources.generated_buf,
            0,
            &resources.readback_buf,
            0,
            byte_size,
        );
        if let Ok(mut inner) = readback_state.inner.lock() {
            inner.pending_map = Some(meta);
        }
        Ok(())
    }
}

#[derive(Default)]
struct TerrainFarUpdateNode;

impl Node for TerrainFarUpdateNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if !world
            .get_resource::<TerrainFarCacheDirty>()
            .is_some_and(|d| d.0)
        {
            return Ok(());
        }
        let Some(resources) = world.get_resource::<TerrainNearGpuResources>() else {
            return Ok(());
        };
        let Some(pipeline_res) = world.get_resource::<TerrainFarUpdatePipeline>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline_id) else {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                match pipeline_cache.get_compute_pipeline_state(pipeline_res.pipeline_id) {
                    CachedPipelineState::Err(e) => {
                        warn!("terrain_gpu: far_update pipeline error: {e}")
                    }
                    s => warn!("terrain_gpu: far_update pipeline not ready: {s:?}"),
                }
            }
            return Ok(());
        };

        let w = resources.pending_far_dispatch_width;
        let h = resources.pending_far_dispatch_height;
        if w == 0 || h == 0 {
            return Ok(());
        }

        let layout = pipeline_cache.get_bind_group_layout(&pipeline_res.bind_group_layout);
        let bind_group = render_context.render_device().create_bind_group(
            "terrain_far_update_bg",
            &layout,
            &BindGroupEntries::sequential((
                resources.far_params_buf.as_entire_binding(),
                BindingResource::TextureView(&resources.far_cache.far_texture_view),
                BindingResource::TextureView(&resources.near_cache.near_texture_view),
                BindingResource::TextureView(&resources.override_cache.near_texture_view),
                resources.far_override_hash_buf.as_entire_binding(),
                resources.far_override_cells_buf.as_entire_binding(),
            )),
        );

        let dispatch_x = w.div_ceil(FAR_UPDATE_WORKGROUP_X);
        let dispatch_y = h.div_ceil(FAR_UPDATE_WORKGROUP_Y);
        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("terrain_far_update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

        Ok(())
    }
}

#[derive(Default)]
struct TerrainBackUpdateNode;

impl Node for TerrainBackUpdateNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if !world
            .get_resource::<TerrainBackCacheDirty>()
            .is_some_and(|d| d.0)
        {
            return Ok(());
        }
        let Some(resources) = world.get_resource::<TerrainNearGpuResources>() else {
            return Ok(());
        };
        let Some(pipeline_res) = world.get_resource::<TerrainFarUpdatePipeline>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.pipeline_id) else {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                match pipeline_cache.get_compute_pipeline_state(pipeline_res.pipeline_id) {
                    CachedPipelineState::Err(e) => {
                        warn!("terrain_gpu: back_update pipeline error: {e}")
                    }
                    s => warn!("terrain_gpu: back_update pipeline not ready: {s:?}"),
                }
            }
            return Ok(());
        };

        let w = resources.pending_back_dispatch_width;
        let h = resources.pending_back_dispatch_height;
        if w == 0 || h == 0 {
            return Ok(());
        }

        let layout = pipeline_cache.get_bind_group_layout(&pipeline_res.bind_group_layout);
        let bind_group = render_context.render_device().create_bind_group(
            "terrain_back_update_bg",
            &layout,
            &BindGroupEntries::sequential((
                resources.back_params_buf.as_entire_binding(),
                BindingResource::TextureView(&resources.back_cache.back_texture_view),
                BindingResource::TextureView(&resources.near_cache.near_texture_view),
                BindingResource::TextureView(&resources.override_cache.near_texture_view),
                resources.far_override_hash_buf.as_entire_binding(),
                resources.far_override_cells_buf.as_entire_binding(),
            )),
        );

        let dispatch_x = w.div_ceil(FAR_UPDATE_WORKGROUP_X);
        let dispatch_y = h.div_ceil(FAR_UPDATE_WORKGROUP_Y);
        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("terrain_back_update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

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
                BindingResource::TextureView(&resources.override_cache.near_texture_view),
                BindingResource::TextureView(&resources.far_cache.far_texture_view),
                BindingResource::TextureView(&resources.back_cache.back_texture_view),
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
struct TerrainNearUpdateRequest {
    dirty: bool,
    override_dirty: bool,
    far_override_dirty: bool,
    far_refresh_dirty: bool,
    back_refresh_dirty: bool,
    initialized: bool,
    terrain_render_version: u64,
    near_lod_k: i32,
    far_lod_k: i32,
    near_render_enabled: bool,
    far_downsample: u32,
    far_width: u32,
    far_height: u32,
    far_origin_x: i32,
    far_origin_y: i32,
    back_downsample: u32,
    back_width: u32,
    back_height: u32,
    back_origin_x: i32,
    back_origin_y: i32,
    back_min_screen_resolution_divisor: f32,
    back_scale_multiplier: f32,
    front_mpp_cells_per_px: f32,
    back_atmosphere_tint: f32,
    sky_color_r: f32,
    sky_color_g: f32,
    sky_color_b: f32,
    palette_seed: u32,
    generation_enabled: bool,
    width: u32,
    height: u32,
    origin_x: i32,
    origin_y: i32,
    ring_offset_x: i32,
    ring_offset_y: i32,
    dirty_cells: Vec<TerrainDirtyCellGpu>,
    override_runs: Vec<TerrainOverrideRunGpu>,
    far_override_hash_mask: u32,
    far_override_hash_len: u32,
    far_override_chunk_count: u32,
    far_override_hash_entries: Vec<TerrainFarOverrideHashEntry>,
    far_override_cells: Vec<u32>,
}

impl Clone for TerrainNearUpdateRequest {
    fn clone(&self) -> Self {
        Self {
            dirty: self.dirty,
            override_dirty: self.override_dirty,
            far_override_dirty: self.far_override_dirty,
            far_refresh_dirty: self.far_refresh_dirty,
            back_refresh_dirty: self.back_refresh_dirty,
            initialized: self.initialized,
            terrain_render_version: self.terrain_render_version,
            near_lod_k: self.near_lod_k,
            far_lod_k: self.far_lod_k,
            near_render_enabled: self.near_render_enabled,
            far_downsample: self.far_downsample,
            far_width: self.far_width,
            far_height: self.far_height,
            far_origin_x: self.far_origin_x,
            far_origin_y: self.far_origin_y,
            back_downsample: self.back_downsample,
            back_width: self.back_width,
            back_height: self.back_height,
            back_origin_x: self.back_origin_x,
            back_origin_y: self.back_origin_y,
            back_min_screen_resolution_divisor: self.back_min_screen_resolution_divisor,
            back_scale_multiplier: self.back_scale_multiplier,
            front_mpp_cells_per_px: self.front_mpp_cells_per_px,
            back_atmosphere_tint: self.back_atmosphere_tint,
            sky_color_r: self.sky_color_r,
            sky_color_g: self.sky_color_g,
            sky_color_b: self.sky_color_b,
            palette_seed: self.palette_seed,
            generation_enabled: self.generation_enabled,
            width: self.width,
            height: self.height,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            ring_offset_x: self.ring_offset_x,
            ring_offset_y: self.ring_offset_y,
            dirty_cells: self.dirty_cells.clone(),
            override_runs: self.override_runs.clone(),
            far_override_hash_mask: self.far_override_hash_mask,
            far_override_hash_len: self.far_override_hash_len,
            far_override_chunk_count: self.far_override_chunk_count,
            far_override_hash_entries: self.far_override_hash_entries.clone(),
            far_override_cells: self.far_override_cells.clone(),
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

fn override_material_or_none(_terrain: &TerrainWorld, _global_cell: IVec2) -> u32 {
    TERRAIN_OVERRIDE_NONE
}

fn chunk_local_to_index(local_cell: IVec2) -> usize {
    (local_cell.y as usize) * crate::physics::world::terrain::CHUNK_SIZE + (local_cell.x as usize)
}

fn append_override_chunk_runs(
    terrain: &TerrainWorld,
    chunk_coord: IVec2,
    out_runs: &mut VecDeque<TerrainOverrideRunGpu>,
) {
    let Some(overrides) = terrain.override_chunk_cells(chunk_coord) else {
        return;
    };
    let mut encoded = [TERRAIN_OVERRIDE_NONE;
        crate::physics::world::terrain::CHUNK_SIZE * crate::physics::world::terrain::CHUNK_SIZE];
    for (local_cell, cell) in overrides.iter_overrides() {
        encoded[chunk_local_to_index(local_cell)] = encode_terrain_cell(cell);
    }

    let mut run_start = 0usize;
    while run_start < encoded.len() {
        let material = encoded[run_start];
        let mut run_len = 1usize;
        while run_start + run_len < encoded.len()
            && encoded[run_start + run_len] == material
            && run_len < u32::MAX as usize
        {
            run_len += 1;
        }
        out_runs.push_back(TerrainOverrideRunGpu {
            chunk_x: chunk_coord.x,
            chunk_y: chunk_coord.y,
            start_index: run_start as u32,
            run_length: run_len as u32,
            material,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        });
        run_start += run_len;
    }
}

fn append_override_chunk_dense_materials(
    terrain: &TerrainWorld,
    chunk_coord: IVec2,
    out_cells: &mut Vec<u32>,
) {
    let mut encoded = [TERRAIN_OVERRIDE_NONE;
        crate::physics::world::terrain::CHUNK_SIZE * crate::physics::world::terrain::CHUNK_SIZE];
    if let Some(overrides) = terrain.override_chunk_cells(chunk_coord) {
        for (local_cell, cell) in overrides.iter_overrides() {
            encoded[chunk_local_to_index(local_cell)] = encode_terrain_cell(cell);
        }
    }
    out_cells.extend_from_slice(&encoded);
}

fn far_override_chunk_hash(chunk_coord: IVec2) -> u32 {
    let x = chunk_coord.x as u32;
    let y = chunk_coord.y as u32;
    let mut h = x.wrapping_mul(0x9E37_79B1u32) ^ y.wrapping_mul(0x85EB_CA77u32).rotate_left(16);
    h ^= h >> 16;
    h
}

fn build_far_override_hash_entries(
    chunks: &[IVec2],
) -> (Vec<TerrainFarOverrideHashEntry>, u32, u32) {
    let hash_len = if chunks.is_empty() {
        1
    } else {
        (chunks.len() as u32).saturating_mul(2).next_power_of_two()
    };
    let hash_mask = hash_len.saturating_sub(1);
    let mut entries = vec![
        TerrainFarOverrideHashEntry {
            chunk_x: 0,
            chunk_y: 0,
            chunk_index: -1,
            _pad0: 0,
        };
        hash_len as usize
    ];
    for (index, chunk) in chunks.iter().copied().enumerate() {
        let mut slot = (far_override_chunk_hash(chunk) & hash_mask) as usize;
        loop {
            if entries[slot].chunk_index < 0 {
                entries[slot] = TerrainFarOverrideHashEntry {
                    chunk_x: chunk.x,
                    chunk_y: chunk.y,
                    chunk_index: index as i32,
                    _pad0: 0,
                };
                break;
            }
            slot = (slot + 1) & hash_mask as usize;
        }
    }
    (entries, hash_mask, hash_len)
}

fn append_override_chunks_for_rect(
    terrain: &TerrainWorld,
    rect_origin: IVec2,
    rect_extent: UVec2,
    out_chunks: &mut HashSet<IVec2>,
) {
    if rect_extent.x == 0 || rect_extent.y == 0 {
        return;
    }
    let max_cell = rect_origin
        + IVec2::new(
            saturating_u32_to_i32(rect_extent.x).saturating_sub(1),
            saturating_u32_to_i32(rect_extent.y).saturating_sub(1),
        );
    let min_chunk = IVec2::new(
        rect_origin
            .x
            .div_euclid(crate::physics::world::terrain::CHUNK_SIZE_I32),
        rect_origin
            .y
            .div_euclid(crate::physics::world::terrain::CHUNK_SIZE_I32),
    );
    let max_chunk = IVec2::new(
        max_cell
            .x
            .div_euclid(crate::physics::world::terrain::CHUNK_SIZE_I32),
        max_cell
            .y
            .div_euclid(crate::physics::world::terrain::CHUNK_SIZE_I32),
    );
    for cy in min_chunk.y..=max_chunk.y {
        for cx in min_chunk.x..=max_chunk.x {
            let chunk = IVec2::new(cx, cy);
            if terrain.override_chunk_cells(chunk).is_some() {
                out_chunks.insert(chunk);
            }
        }
    }
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

fn prepare_terrain_near_update_request(
    mut terrain: ResMut<TerrainWorld>,
    render_params: Res<ActiveRenderParams>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut request: ResMut<TerrainNearUpdateRequest>,
    mut cache_state: ResMut<TerrainCacheState>,
    mut diagnostics: ResMut<TerrainRenderDiagnostics>,
    camera_q: Query<(&Transform, &Projection), With<MainCamera>>,
) {
    let settings = terrain_runtime_settings_from_active(Some(&render_params));
    diagnostics.terrain_generation_eval_count_frame = 0;
    diagnostics.terrain_generation_origin_delta_x_frame = 0;
    diagnostics.terrain_generation_origin_delta_y_frame = 0;
    diagnostics.terrain_generation_full_refresh_frame = false;
    diagnostics.terrain_generation_full_refresh_reason_bits = 0;
    diagnostics.terrain_override_runs_frame = 0;
    diagnostics.terrain_override_cells_frame = 0;
    diagnostics.terrain_override_pending_runs = cache_state.override_run_queue.len() as u32;
    diagnostics.terrain_override_budget_completion_frame =
        if diagnostics.terrain_override_pending_runs == 0 {
            1.0
        } else {
            0.0
        };

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
    let (camera_pos, viewport_world_m, near_extent, near_lod_k) = camera_q
        .single()
        .map(|(transform, projection)| {
            let viewport_world = projection_viewport_world(
                projection,
                window_size_px.max(UVec2::splat(1)).as_vec2(),
            )
            .unwrap_or_else(|| default_viewport_world(window_size_px));
            (
                transform.translation.xy(),
                viewport_world,
                compute_near_cache_extent_for_projection(
                    window_size_px,
                    projection,
                    window_size_px.max(UVec2::splat(1)).as_vec2(),
                    settings,
                ),
                compute_near_lod_k_from_viewport_world(viewport_world),
            )
        })
        .unwrap_or((
            Vec2::ZERO,
            default_viewport_world(window_size_px),
            compute_near_cache_extent_with_settings(window_size_px, settings),
            0,
        ));
    let front_mpp_cells_per_px =
        (viewport_world_m.y / CELL_SIZE_M.max(1e-6)) / (window_size_px.y.max(1) as f32);
    let back_scale = compute_back_display_scale(front_mpp_cells_per_px, settings);
    let far_extent = compute_far_cache_base_extent_with_settings(window_size_px, settings);
    let far_downsample =
        compute_far_downsample_for_viewport_with_settings(viewport_world_m, far_extent, settings);
    let back_extent = far_extent;
    let back_downsample = compute_back_downsample_for_viewport_with_settings(
        viewport_world_m * back_scale.max(1.0),
        back_extent,
        settings,
    );
    let far_lod_k = near_lod_k + FAR_LOD_OFFSET;
    let near_render_enabled =
        near_covers_viewport_with_margin(near_extent, viewport_world_m, settings);
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
    let far_origin = compute_far_origin_world(camera_cell, far_extent, far_downsample);
    let back_origin = compute_far_origin_world(camera_cell, back_extent, back_downsample);
    let old_origin = IVec2::new(request.origin_x, request.origin_y);
    let origin_delta = if request.initialized {
        new_origin - old_origin
    } else {
        IVec2::ZERO
    };
    diagnostics.terrain_generation_origin_delta_x_frame = origin_delta.x;
    diagnostics.terrain_generation_origin_delta_y_frame = origin_delta.y;

    let origin_changed = origin_delta != IVec2::ZERO;
    let far_origin_changed =
        request.initialized && IVec2::new(request.far_origin_x, request.far_origin_y) != far_origin;
    let far_extent_changed = request.initialized
        && (request.far_width != far_extent.x || request.far_height != far_extent.y);
    let back_origin_changed = request.initialized
        && IVec2::new(request.back_origin_x, request.back_origin_y) != back_origin;
    let back_extent_changed = request.initialized
        && (request.back_width != back_extent.x || request.back_height != back_extent.y);

    let terrain_changed = request.terrain_render_version != terrain.terrain_render_version();
    let near_lod_changed = request.initialized && request.near_lod_k != near_lod_k;
    let far_lod_changed = request.initialized && request.far_lod_k != far_lod_k;
    let far_downsample_changed = request.initialized && request.far_downsample != far_downsample;
    let back_downsample_changed = request.initialized && request.back_downsample != back_downsample;
    let back_min_resolution_divisor_changed = request.initialized
        && (request.back_min_screen_resolution_divisor
            - settings.back_min_screen_resolution_divisor)
            .abs()
            > 1e-6;
    let back_scale_multiplier_changed = request.initialized
        && (request.back_scale_multiplier - settings.back_scale_multiplier).abs() > 1e-6;
    let front_mpp_changed = request.initialized
        && (request.front_mpp_cells_per_px - front_mpp_cells_per_px).abs() > 1e-6;
    let back_atmosphere_tint_changed = request.initialized
        && (request.back_atmosphere_tint - settings.back_atmosphere_tint).abs() > 1e-6;
    let sky_color_changed = request.initialized
        && ((request.sky_color_r - settings.sky_color_r).abs() > 1e-6
            || (request.sky_color_g - settings.sky_color_g).abs() > 1e-6
            || (request.sky_color_b - settings.sky_color_b).abs() > 1e-6);
    let palette_seed_changed =
        request.initialized && request.palette_seed != settings.terrain_palette_seed;
    let near_mode_changed =
        request.initialized && request.near_render_enabled != near_render_enabled;
    if request.initialized
        && !terrain_changed
        && !origin_changed
        && !extent_changed
        && !far_origin_changed
        && !far_extent_changed
        && !back_origin_changed
        && !back_extent_changed
        && !near_lod_changed
        && !far_lod_changed
        && !far_downsample_changed
        && !back_downsample_changed
        && !back_min_resolution_divisor_changed
        && !back_scale_multiplier_changed
        && !front_mpp_changed
        && !back_atmosphere_tint_changed
        && !sky_color_changed
        && !palette_seed_changed
        && !near_mode_changed
        && cache_state.override_run_queue.is_empty()
    {
        request.dirty = false;
        request.override_dirty = false;
        request.far_override_dirty = false;
        request.far_refresh_dirty = false;
        request.back_refresh_dirty = false;
        request.override_runs.clear();
        return;
    }

    // Full refresh is needed only when the new view no longer overlaps the previous cache,
    // or when terrain contents / cache extent changed.
    let no_overlap = request.initialized
        && (origin_delta.x.abs() >= near_extent_i.x || origin_delta.y.abs() >= near_extent_i.y);
    let full_refresh = near_render_enabled
        && (!request.initialized
            || terrain_changed
            || extent_changed
            || no_overlap
            || near_mode_changed);
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
    if !full_refresh {
        full_refresh_reason_bits = 0;
    }
    diagnostics.terrain_generation_full_refresh_frame = full_refresh;
    diagnostics.terrain_generation_full_refresh_reason_bits = full_refresh_reason_bits;

    if !near_render_enabled || full_refresh {
        cache_state.ring_offset = IVec2::ZERO;
    } else {
        cache_state.ring_offset =
            normalize_ring_offset(cache_state.ring_offset + origin_delta, near_extent);
    }

    cache_state.cache_origin_world = new_origin;
    cache_state.lod_k = near_lod_k;
    cache_state.near_dirty_queue.clear();
    cache_state.far_dirty_queue.clear();

    let mut dirty_cells = Vec::new();
    if near_render_enabled {
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
    }

    let near_dirty = !dirty_cells.is_empty();
    if near_dirty {
        diagnostics.terrain_generation_eval_count_frame = dirty_cells.len() as u32;
    }

    let dirty_override_chunks = terrain.take_dirty_override_chunks();
    let near_max = new_origin + near_extent_i - IVec2::ONE;
    let mut override_chunks = HashSet::new();
    for dirty_tile in &cache_state.near_dirty_queue {
        append_override_chunks_for_rect(
            terrain.as_ref(),
            dirty_tile._world_origin,
            dirty_tile._extent_cells,
            &mut override_chunks,
        );
    }
    for chunk in dirty_override_chunks {
        let chunk_min = chunk * crate::physics::world::terrain::CHUNK_SIZE_I32;
        let chunk_max =
            chunk_min + IVec2::splat(crate::physics::world::terrain::CHUNK_SIZE_I32 - 1);
        let intersects = chunk_min.x <= near_max.x
            && chunk_max.x >= new_origin.x
            && chunk_min.y <= near_max.y
            && chunk_max.y >= new_origin.y;
        if intersects {
            override_chunks.insert(chunk);
        }
    }
    let mut override_chunk_vec: Vec<_> = override_chunks.into_iter().collect();
    override_chunk_vec.sort_by_key(|coord| (coord.y, coord.x));
    for chunk in override_chunk_vec {
        append_override_chunk_runs(terrain.as_ref(), chunk, &mut cache_state.override_run_queue);
    }

    let mut override_runs = Vec::new();
    let mut override_cells_frame = 0u32;
    while override_runs.len() < MAX_OVERRIDE_RUNS_PER_FRAME {
        let Some(run) = cache_state.override_run_queue.pop_front() else {
            break;
        };
        override_cells_frame = override_cells_frame.saturating_add(run.run_length);
        override_runs.push(run);
    }
    diagnostics.terrain_override_runs_frame = override_runs.len() as u32;
    diagnostics.terrain_override_cells_frame = override_cells_frame;
    diagnostics.terrain_override_pending_runs = cache_state.override_run_queue.len() as u32;
    diagnostics.terrain_override_runs_total = diagnostics
        .terrain_override_runs_total
        .saturating_add(diagnostics.terrain_override_runs_frame as u64);
    diagnostics.terrain_override_cells_total = diagnostics
        .terrain_override_cells_total
        .saturating_add(diagnostics.terrain_override_cells_frame as u64);
    let override_total = diagnostics
        .terrain_override_runs_frame
        .saturating_add(diagnostics.terrain_override_pending_runs);
    diagnostics.terrain_override_budget_completion_frame = if override_total == 0 {
        1.0
    } else {
        diagnostics.terrain_override_runs_frame as f32 / override_total as f32
    };
    let override_dirty = !override_runs.is_empty();
    let far_override_dirty = !request.initialized || terrain_changed;
    let mut far_override_hash_mask = request.far_override_hash_mask;
    let mut far_override_hash_len = request.far_override_hash_len.max(1);
    let mut far_override_chunk_count = request.far_override_chunk_count;
    if far_override_dirty {
        let override_chunk_coords = terrain.override_chunk_coords();
        let mut far_override_cells = Vec::with_capacity(
            override_chunk_coords.len()
                * crate::physics::world::terrain::CHUNK_SIZE
                * crate::physics::world::terrain::CHUNK_SIZE,
        );
        for &chunk in &override_chunk_coords {
            append_override_chunk_dense_materials(terrain.as_ref(), chunk, &mut far_override_cells);
        }
        let (far_override_hash_entries, hash_mask, hash_len) =
            build_far_override_hash_entries(&override_chunk_coords);
        far_override_hash_mask = hash_mask;
        far_override_hash_len = hash_len.max(1);
        far_override_chunk_count = override_chunk_coords.len() as u32;
        request.far_override_hash_entries = far_override_hash_entries;
        request.far_override_cells = far_override_cells;
    }

    request.generation_enabled = terrain.generation_enabled();
    request.terrain_render_version = terrain.terrain_render_version();
    request.near_lod_k = near_lod_k;
    request.far_lod_k = far_lod_k;
    request.near_render_enabled = near_render_enabled;
    request.far_downsample = far_downsample;
    request.far_width = far_extent.x;
    request.far_height = far_extent.y;
    request.far_origin_x = far_origin.x;
    request.far_origin_y = far_origin.y;
    request.back_downsample = back_downsample;
    request.back_width = back_extent.x;
    request.back_height = back_extent.y;
    request.back_origin_x = back_origin.x;
    request.back_origin_y = back_origin.y;
    request.back_min_screen_resolution_divisor = settings.back_min_screen_resolution_divisor;
    request.back_scale_multiplier = settings.back_scale_multiplier;
    request.front_mpp_cells_per_px = front_mpp_cells_per_px;
    request.back_atmosphere_tint = settings.back_atmosphere_tint;
    request.sky_color_r = settings.sky_color_r;
    request.sky_color_g = settings.sky_color_g;
    request.sky_color_b = settings.sky_color_b;
    request.palette_seed = settings.terrain_palette_seed;
    request.width = near_extent.x;
    request.height = near_extent.y;
    request.origin_x = new_origin.x;
    request.origin_y = new_origin.y;
    request.ring_offset_x = cache_state.ring_offset.x;
    request.ring_offset_y = cache_state.ring_offset.y;
    request.dirty_cells = dirty_cells;
    request.override_runs = override_runs;
    request.dirty = near_dirty;
    request.override_dirty = override_dirty;
    request.far_override_dirty = far_override_dirty;
    request.far_override_hash_mask = far_override_hash_mask;
    request.far_override_hash_len = far_override_hash_len;
    request.far_override_chunk_count = far_override_chunk_count;
    request.far_refresh_dirty = !request.initialized
        || terrain_changed
        || far_downsample_changed
        || far_origin_changed
        || far_extent_changed;
    request.back_refresh_dirty = !request.initialized
        || terrain_changed
        || back_downsample_changed
        || back_origin_changed
        || back_extent_changed;
    request.initialized = true;
}

fn drive_generated_chunk_readback_requests(
    mut cache: ResMut<TerrainGeneratedChunkCache>,
    terrain_world: Res<TerrainWorld>,
    mut request: ResMut<TerrainGeneratedChunkReadbackRequest>,
) {
    if cache.inflight.is_some() {
        request.active = false;
        return;
    }
    let Some((request_id, chunk_coord)) = cache.begin_next_request() else {
        request.active = false;
        return;
    };
    if terrain_world.chunk(chunk_coord).is_some() {
        let _ = cache.finish_request(request_id, chunk_coord);
        request.active = false;
        return;
    }
    request.active = true;
    request.request_id = request_id;
    request.chunk_coord = chunk_coord;
    request.generation_enabled = terrain_world.generation_enabled();
}

fn apply_generated_chunk_readback_results(
    mut terrain_world: ResMut<TerrainWorld>,
    mut cache: ResMut<TerrainGeneratedChunkCache>,
    readback_result: Res<TerrainGeneratedChunkReadbackResult>,
) {
    let Some(payload) = readback_result.take() else {
        return;
    };
    if payload.cells.len() != TERRAIN_CHUNK_CELLS {
        return;
    }
    if !cache.finish_request(payload.request_id, payload.chunk_coord) {
        return;
    }
    let mut cells = [0u16; TERRAIN_CHUNK_CELLS];
    cells.copy_from_slice(&payload.cells);
    cache.cache_generated_chunk(payload.chunk_coord, cells);
    if terrain_world.chunk(payload.chunk_coord).is_none() {
        terrain_world.load_generated_chunk_from_material_ids(payload.chunk_coord, &cells);
    }
}

// ── Render startup systems ────────────────────────────────────────────────────

fn init_terrain_near_gpu_resources(mut commands: Commands, render_device: Res<RenderDevice>) {
    let settings = TerrainRuntimeSettings::default();
    let near_extent = compute_near_cache_extent(DEFAULT_SCREEN_SIZE_PX);
    let default_viewport_world = default_viewport_world(DEFAULT_SCREEN_SIZE_PX);
    let far_extent = compute_far_cache_base_extent_with_settings(DEFAULT_SCREEN_SIZE_PX, settings);
    let front_mpp_cells_per_px =
        (default_viewport_world.y / CELL_SIZE_M.max(1e-6)) / (DEFAULT_SCREEN_SIZE_PX.y as f32);
    let back_scale = compute_back_display_scale(front_mpp_cells_per_px, settings);
    let far_downsample = compute_far_downsample_for_viewport_with_settings(
        default_viewport_world,
        far_extent,
        settings,
    );
    let back_extent = far_extent;
    let back_downsample = compute_back_downsample_for_viewport_with_settings(
        default_viewport_world * back_scale.max(1.0),
        back_extent,
        settings,
    );
    let near_cache = make_near_cache(&render_device, near_extent);
    let override_cache = make_override_cache(&render_device, near_extent);
    let far_cache = make_far_cache(&render_device, far_extent);
    let back_cache = make_back_cache(&render_device, back_extent);
    let dirty_cells_capacity = near_extent.x.saturating_mul(near_extent.y).max(1);
    let dirty_cells_buf = make_dirty_cells_buffer(&render_device, dirty_cells_capacity);
    let override_runs_capacity = MAX_OVERRIDE_RUNS_PER_DISPATCH.max(1);
    let override_runs_buf = make_override_runs_buffer(&render_device, override_runs_capacity);
    let far_override_hash_capacity = 1;
    let far_override_hash_buf =
        make_far_override_hash_buffer(&render_device, far_override_hash_capacity);
    let far_override_cells_capacity = 1;
    let far_override_cells_buf =
        make_far_override_cells_buffer(&render_device, far_override_cells_capacity);
    let far_origin = compute_far_origin_world(IVec2::ZERO, far_extent, far_downsample);
    let back_origin = compute_far_origin_world(IVec2::ZERO, back_extent, back_downsample);

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
    let override_params = TerrainOverrideParams {
        cache_origin_x: 0,
        cache_origin_y: 0,
        cache_width: near_extent.x,
        cache_height: near_extent.y,
        ring_offset_x: 0,
        ring_offset_y: 0,
        override_none: TERRAIN_OVERRIDE_NONE,
        run_count: 0,
        chunk_size_i32: crate::physics::world::terrain::CHUNK_SIZE_I32,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    let override_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_override_params"),
            contents: bytemuck::bytes_of(&override_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );

    let far_params = TerrainFarParams {
        far_origin_x: far_origin.x,
        far_origin_y: far_origin.y,
        far_width: far_cache.extent_cells.x,
        far_height: far_cache.extent_cells.y,
        far_downsample,
        generation_enabled: 1,
        near_origin_x: 0,
        near_origin_y: 0,
        near_width: near_cache.extent_cells.x,
        near_height: near_cache.extent_cells.y,
        ring_offset_x: 0,
        ring_offset_y: 0,
        override_none: TERRAIN_OVERRIDE_NONE,
        override_hash_mask: 0,
        override_hash_len: 1,
        override_chunk_count: 0,
        override_chunk_size_i32: CHUNK_SIZE_I32,
        near_cache_enabled: 1,
        _pad0: 0,
        _pad1: 0,
    };
    let far_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_far_params"),
            contents: bytemuck::bytes_of(&far_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );
    let back_params = TerrainFarParams {
        far_origin_x: back_origin.x,
        far_origin_y: back_origin.y,
        far_width: back_cache.extent_cells.x,
        far_height: back_cache.extent_cells.y,
        far_downsample: back_downsample,
        generation_enabled: 1,
        near_origin_x: 0,
        near_origin_y: 0,
        near_width: near_cache.extent_cells.x,
        near_height: near_cache.extent_cells.y,
        ring_offset_x: 0,
        ring_offset_y: 0,
        override_none: TERRAIN_OVERRIDE_NONE,
        override_hash_mask: 0,
        override_hash_len: 1,
        override_chunk_count: 0,
        override_chunk_size_i32: CHUNK_SIZE_I32,
        near_cache_enabled: 1,
        _pad0: 0,
        _pad1: 0,
    };
    let back_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_back_params"),
            contents: bytemuck::bytes_of(&back_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );

    let compose_params = TerrainComposeParams {
        cell_size_m: CELL_SIZE_M,
        dot_size_m: CELL_SIZE_M / TERRAIN_DOTS_PER_CELL as f32,
        back_min_screen_resolution_divisor: settings.back_min_screen_resolution_divisor,
        back_scale_multiplier: settings.back_scale_multiplier,
        front_mpp_cells_per_px: 1.0,
        back_atmosphere_tint: settings.back_atmosphere_tint,
        far_origin_x: far_origin.x,
        far_origin_y: far_origin.y,
        back_origin_x: back_origin.x,
        back_origin_y: back_origin.y,
        near_origin_x: 0,
        near_origin_y: 0,
        palette_seed: settings.terrain_palette_seed,
        dots_per_cell: TERRAIN_DOTS_PER_CELL,
        ring_offset_x: 0,
        ring_offset_y: 0,
        near_enabled: 1,
        far_downsample,
        back_downsample,
        _pad1: 0,
        sky_color_r: settings.sky_color_r,
        sky_color_g: settings.sky_color_g,
        sky_color_b: settings.sky_color_b,
        _pad2: 0.0,
    };
    let compose_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_compose_params"),
            contents: bytemuck::bytes_of(&compose_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );
    let chunk_params = TerrainChunkGenerateParams {
        chunk_x: 0,
        chunk_y: 0,
        chunk_size_i32: CHUNK_SIZE_I32,
        generation_enabled: 1,
    };
    let chunk_params_buf = render_device.create_buffer_with_data(
        &bevy::render::render_resource::BufferInitDescriptor {
            label: Some("terrain_chunk_generate_params"),
            contents: bytemuck::bytes_of(&chunk_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        },
    );
    let chunk_generated_buf = make_chunk_generated_buffer(&render_device);
    let chunk_readback_buf = make_chunk_readback_buffer(&render_device);

    commands.insert_resource(TerrainNearGpuResources {
        near_cache,
        override_cache,
        far_cache,
        back_cache,
        dirty_cells_buf,
        dirty_cells_capacity,
        override_runs_buf,
        override_runs_capacity,
        far_override_hash_buf,
        far_override_hash_capacity,
        far_override_cells_buf,
        far_override_cells_capacity,
        near_params_buf,
        override_params_buf,
        far_params_buf,
        back_params_buf,
        compose_params_buf,
        pending_dispatch_frames: 0,
        pending_dispatch_cells: 0,
        pending_override_dispatch_frames: 0,
        pending_override_dispatch_runs: 0,
        pending_far_dispatch_frames: 0,
        pending_far_dispatch_width: 0,
        pending_far_dispatch_height: 0,
        pending_back_dispatch_frames: 0,
        pending_back_dispatch_width: 0,
        pending_back_dispatch_height: 0,
    });
    commands.insert_resource(TerrainNearCacheDirty(false));
    commands.insert_resource(TerrainOverrideCacheDirty(false));
    commands.insert_resource(TerrainFarCacheDirty(false));
    commands.insert_resource(TerrainBackCacheDirty(false));
    commands.insert_resource(TerrainChunkGenerateGpuResources {
        params_buf: chunk_params_buf,
        generated_buf: chunk_generated_buf,
        readback_buf: chunk_readback_buf,
    });
    commands.insert_resource(TerrainChunkGenerateDirty(false));
    commands.insert_resource(TerrainChunkReadbackState::default());
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

fn init_terrain_far_update_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let make_layout = || {
        BindGroupLayoutDescriptor::new(
            "terrain_far_update_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, core::num::NonZeroU64::new(80)),
                    texture_storage_2d(TextureFormat::Rgba8Uint, StorageTextureAccess::WriteOnly),
                    texture_2d(TextureSampleType::Uint),
                    texture_2d(TextureSampleType::Uint),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                ),
            ),
        )
    };
    let terrain_gen_shader = asset_server.load(TERRAIN_GEN_SHADER_PATH);
    let shader = asset_server.load(TERRAIN_FAR_UPDATE_SHADER_PATH);
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("terrain_far_update_compute".into()),
        layout: vec![make_layout()],
        push_constant_ranges: vec![],
        shader,
        shader_defs: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(TerrainFarUpdatePipeline {
        bind_group_layout: make_layout(),
        pipeline_id,
        _terrain_gen_shader: terrain_gen_shader,
    });
}

fn init_terrain_override_apply_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let make_layout = || {
        BindGroupLayoutDescriptor::new(
            "terrain_override_apply_layout",
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
    let shader = asset_server.load(TERRAIN_OVERRIDE_APPLY_SHADER_PATH);
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("terrain_override_apply_compute".into()),
        layout: vec![make_layout()],
        push_constant_ranges: vec![],
        shader,
        shader_defs: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(TerrainOverrideApplyPipeline {
        bind_group_layout: make_layout(),
        pipeline_id,
    });
}

fn init_terrain_chunk_generate_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let make_layout = || {
        BindGroupLayoutDescriptor::new(
            "terrain_chunk_generate_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, core::num::NonZeroU64::new(16)),
                    storage_buffer_sized(false, None),
                ),
            ),
        )
    };
    let terrain_gen_shader = asset_server.load(TERRAIN_GEN_SHADER_PATH);
    let shader = asset_server.load(TERRAIN_CHUNK_GENERATE_SHADER_PATH);
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("terrain_chunk_generate_compute".into()),
        layout: vec![make_layout()],
        push_constant_ranges: vec![],
        shader,
        shader_defs: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: false,
    });
    commands.insert_resource(TerrainChunkGeneratePipeline {
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
                uniform_buffer_sized(
                    false,
                    core::num::NonZeroU64::new(std::mem::size_of::<TerrainComposeParams>() as u64),
                ),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Uint),
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
    // Add Core2d edge: TerrainCompose → WaterDotGpu.
    // Water pass performs terrain-occlusion discard so terrain remains visually in front.
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
    override_pipeline: Res<TerrainOverrideApplyPipeline>,
    far_pipeline: Res<TerrainFarUpdatePipeline>,
    mut dirty: ResMut<TerrainNearCacheDirty>,
    mut override_dirty: ResMut<TerrainOverrideCacheDirty>,
    mut far_dirty: ResMut<TerrainFarCacheDirty>,
    mut back_dirty: ResMut<TerrainBackCacheDirty>,
) {
    let near_upload_requested = upload.dirty && !upload.dirty_cells.is_empty();
    let override_upload_requested = upload.override_dirty && !upload.override_runs.is_empty();
    let far_override_upload_requested = upload.far_override_dirty;
    let far_refresh_requested = upload.far_refresh_dirty;
    let back_refresh_requested = upload.back_refresh_dirty;
    let requested_far_downsample = upload.far_downsample.max(1);
    let requested_back_downsample = upload.back_downsample.max(1);
    if near_upload_requested
        || override_upload_requested
        || far_override_upload_requested
        || far_refresh_requested
        || back_refresh_requested
    {
        let requested_extent = UVec2::new(upload.width.max(1), upload.height.max(1));
        let requested_far_extent = UVec2::new(upload.far_width.max(1), upload.far_height.max(1));
        let requested_back_extent = UVec2::new(upload.back_width.max(1), upload.back_height.max(1));
        if near_upload_requested && resources.near_cache.extent_cells != requested_extent {
            resources.near_cache = make_near_cache(&render_device, requested_extent);
            resources.override_cache = make_override_cache(&render_device, requested_extent);
        }
        if resources.far_cache.extent_cells != requested_far_extent {
            resources.far_cache = make_far_cache(&render_device, requested_far_extent);
        }
        if resources.back_cache.extent_cells != requested_back_extent {
            resources.back_cache = make_back_cache(&render_device, requested_back_extent);
        }

        if near_upload_requested {
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

            // Retry only while pipeline is still compiling; otherwise dispatch once.
            let near_pipeline_ready = pipeline_cache
                .get_compute_pipeline(near_pipeline.pipeline_id)
                .is_some();
            resources.pending_dispatch_frames = if near_pipeline_ready { 1 } else { 32 };
            resources.pending_dispatch_cells = dirty_count;
        }

        if override_upload_requested {
            let run_count_full = upload.override_runs.len() as u32;
            let run_count = run_count_full.min(MAX_OVERRIDE_RUNS_PER_DISPATCH);
            if run_count_full > run_count {
                static RUN_CLAMP_WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !RUN_CLAMP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    warn!(
                        "terrain_gpu: override run count {} exceeds dispatch limit {}, clamping",
                        run_count_full, MAX_OVERRIDE_RUNS_PER_DISPATCH
                    );
                }
            }
            if run_count > resources.override_runs_capacity {
                let new_capacity = run_count.next_power_of_two();
                resources.override_runs_buf =
                    make_override_runs_buffer(&render_device, new_capacity);
                resources.override_runs_capacity = new_capacity;
            }
            queue.write_buffer(
                &resources.override_runs_buf,
                0,
                cast_slice(&upload.override_runs[..run_count as usize]),
            );
            let override_params = TerrainOverrideParams {
                cache_origin_x: upload.origin_x,
                cache_origin_y: upload.origin_y,
                cache_width: upload.width,
                cache_height: upload.height,
                ring_offset_x: upload.ring_offset_x,
                ring_offset_y: upload.ring_offset_y,
                override_none: TERRAIN_OVERRIDE_NONE,
                run_count,
                chunk_size_i32: crate::physics::world::terrain::CHUNK_SIZE_I32,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(
                &resources.override_params_buf,
                0,
                bytemuck::bytes_of(&override_params),
            );
            let override_pipeline_ready = pipeline_cache
                .get_compute_pipeline(override_pipeline.pipeline_id)
                .is_some();
            resources.pending_override_dispatch_frames =
                if override_pipeline_ready { 1 } else { 32 };
            resources.pending_override_dispatch_runs = run_count;
        }

        if far_override_upload_requested {
            let hash_count = upload.far_override_hash_entries.len() as u32;
            if hash_count > resources.far_override_hash_capacity {
                let new_capacity = hash_count.next_power_of_two();
                resources.far_override_hash_buf =
                    make_far_override_hash_buffer(&render_device, new_capacity);
                resources.far_override_hash_capacity = new_capacity;
            }
            if hash_count > 0 {
                queue.write_buffer(
                    &resources.far_override_hash_buf,
                    0,
                    cast_slice(&upload.far_override_hash_entries),
                );
            }
            let cell_count = upload.far_override_cells.len() as u32;
            if cell_count > resources.far_override_cells_capacity {
                let new_capacity = cell_count.next_power_of_two();
                resources.far_override_cells_buf =
                    make_far_override_cells_buffer(&render_device, new_capacity);
                resources.far_override_cells_capacity = new_capacity;
            }
            if cell_count > 0 {
                queue.write_buffer(
                    &resources.far_override_cells_buf,
                    0,
                    cast_slice(&upload.far_override_cells),
                );
            }
        }

        // Update far-aggregation params and request full far refresh.
        let far_params = TerrainFarParams {
            far_origin_x: upload.far_origin_x,
            far_origin_y: upload.far_origin_y,
            far_width: resources.far_cache.extent_cells.x,
            far_height: resources.far_cache.extent_cells.y,
            far_downsample: requested_far_downsample,
            generation_enabled: u32::from(upload.generation_enabled),
            near_origin_x: upload.origin_x,
            near_origin_y: upload.origin_y,
            near_width: upload.width,
            near_height: upload.height,
            ring_offset_x: upload.ring_offset_x,
            ring_offset_y: upload.ring_offset_y,
            override_none: TERRAIN_OVERRIDE_NONE,
            override_hash_mask: upload.far_override_hash_mask,
            override_hash_len: upload.far_override_hash_len,
            override_chunk_count: upload.far_override_chunk_count,
            override_chunk_size_i32: CHUNK_SIZE_I32,
            near_cache_enabled: u32::from(upload.near_render_enabled),
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(
            &resources.far_params_buf,
            0,
            bytemuck::bytes_of(&far_params),
        );
        let back_params = TerrainFarParams {
            far_origin_x: upload.back_origin_x,
            far_origin_y: upload.back_origin_y,
            far_width: resources.back_cache.extent_cells.x,
            far_height: resources.back_cache.extent_cells.y,
            far_downsample: requested_back_downsample,
            generation_enabled: u32::from(upload.generation_enabled),
            near_origin_x: upload.origin_x,
            near_origin_y: upload.origin_y,
            near_width: upload.width,
            near_height: upload.height,
            ring_offset_x: upload.ring_offset_x,
            ring_offset_y: upload.ring_offset_y,
            override_none: TERRAIN_OVERRIDE_NONE,
            override_hash_mask: upload.far_override_hash_mask,
            override_hash_len: upload.far_override_hash_len,
            override_chunk_count: upload.far_override_chunk_count,
            override_chunk_size_i32: CHUNK_SIZE_I32,
            near_cache_enabled: u32::from(upload.near_render_enabled),
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(
            &resources.back_params_buf,
            0,
            bytemuck::bytes_of(&back_params),
        );

        if far_refresh_requested {
            let far_pipeline_ready = pipeline_cache
                .get_compute_pipeline(far_pipeline.pipeline_id)
                .is_some();
            resources.pending_far_dispatch_frames = if far_pipeline_ready { 1 } else { 32 };
            resources.pending_far_dispatch_width = resources.far_cache.extent_cells.x;
            resources.pending_far_dispatch_height = resources.far_cache.extent_cells.y;
        }
        if back_refresh_requested {
            let back_pipeline_ready = pipeline_cache
                .get_compute_pipeline(far_pipeline.pipeline_id)
                .is_some();
            resources.pending_back_dispatch_frames = if back_pipeline_ready { 1 } else { 32 };
            resources.pending_back_dispatch_width = resources.back_cache.extent_cells.x;
            resources.pending_back_dispatch_height = resources.back_cache.extent_cells.y;
        }
    }
    // Keep compose uniforms in sync even on frames without cache refresh.
    let compose_params = TerrainComposeParams {
        cell_size_m: CELL_SIZE_M,
        dot_size_m: CELL_SIZE_M / TERRAIN_DOTS_PER_CELL as f32,
        back_min_screen_resolution_divisor: upload.back_min_screen_resolution_divisor,
        back_scale_multiplier: upload.back_scale_multiplier,
        front_mpp_cells_per_px: upload.front_mpp_cells_per_px,
        back_atmosphere_tint: upload.back_atmosphere_tint,
        far_origin_x: upload.far_origin_x,
        far_origin_y: upload.far_origin_y,
        back_origin_x: upload.back_origin_x,
        back_origin_y: upload.back_origin_y,
        near_origin_x: upload.origin_x,
        near_origin_y: upload.origin_y,
        palette_seed: upload.palette_seed,
        dots_per_cell: TERRAIN_DOTS_PER_CELL,
        ring_offset_x: upload.ring_offset_x,
        ring_offset_y: upload.ring_offset_y,
        near_enabled: u32::from(upload.near_render_enabled),
        far_downsample: requested_far_downsample,
        back_downsample: requested_back_downsample,
        _pad1: 0,
        sky_color_r: upload.sky_color_r,
        sky_color_g: upload.sky_color_g,
        sky_color_b: upload.sky_color_b,
        _pad2: 0.0,
    };
    queue.write_buffer(
        &resources.compose_params_buf,
        0,
        bytemuck::bytes_of(&compose_params),
    );
    if resources.pending_dispatch_frames > 0 && resources.pending_dispatch_cells > 0 {
        dirty.0 = true;
        resources.pending_dispatch_frames -= 1;
    }
    if resources.pending_override_dispatch_frames > 0
        && resources.pending_override_dispatch_runs > 0
    {
        override_dirty.0 = true;
        resources.pending_override_dispatch_frames -= 1;
    }
    if resources.pending_far_dispatch_frames > 0
        && resources.pending_far_dispatch_width > 0
        && resources.pending_far_dispatch_height > 0
    {
        far_dirty.0 = true;
        resources.pending_far_dispatch_frames -= 1;
    }
    if resources.pending_back_dispatch_frames > 0
        && resources.pending_back_dispatch_width > 0
        && resources.pending_back_dispatch_height > 0
    {
        back_dirty.0 = true;
        resources.pending_back_dispatch_frames -= 1;
    }
}

fn prepare_terrain_chunk_generate_uploads(
    request: Res<TerrainGeneratedChunkReadbackRequest>,
    resources: Res<TerrainChunkGenerateGpuResources>,
    queue: Res<RenderQueue>,
    readback_state: Res<TerrainChunkReadbackState>,
    mut dirty: ResMut<TerrainChunkGenerateDirty>,
) {
    let Ok(mut inner) = readback_state.inner.lock() else {
        return;
    };
    if inner.pending_dispatch.is_some() {
        dirty.0 = true;
        return;
    }
    if !request.active {
        return;
    }
    if inner.mapped || inner.pending_map.is_some() || inner.pending_dispatch.is_some() {
        return;
    }
    let params = TerrainChunkGenerateParams {
        chunk_x: request.chunk_coord.x,
        chunk_y: request.chunk_coord.y,
        chunk_size_i32: CHUNK_SIZE_I32,
        generation_enabled: u32::from(request.generation_enabled),
    };
    queue.write_buffer(&resources.params_buf, 0, bytemuck::bytes_of(&params));
    inner.pending_dispatch = Some(TerrainGeneratedChunkInflight {
        request_id: request.request_id,
        chunk_coord: request.chunk_coord,
    });
    dirty.0 = true;
}

fn readback_generated_chunks(
    resources: Res<TerrainChunkGenerateGpuResources>,
    state: Res<TerrainChunkReadbackState>,
    result: Res<TerrainGeneratedChunkReadbackResult>,
) {
    let byte_size = (TERRAIN_CHUNK_CELLS * std::mem::size_of::<u32>()) as u64;
    let ready = state
        .mapped_ready
        .load(std::sync::atomic::Ordering::Acquire);
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    if inner.mapped {
        if !ready {
            return;
        }
        let Some(meta) = inner.mapped_meta.take() else {
            return;
        };
        let slice = resources.readback_buf.slice(..byte_size);
        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(data.as_ref());
        let values: Vec<u16> = words
            .iter()
            .take(TERRAIN_CHUNK_CELLS)
            .map(|v| *v as u16)
            .collect();
        drop(data);
        resources.readback_buf.unmap();
        state
            .mapped_ready
            .store(false, std::sync::atomic::Ordering::Release);
        inner.mapped = false;
        result.store(TerrainGeneratedChunkReadbackPayload {
            request_id: meta.request_id,
            chunk_coord: meta.chunk_coord,
            cells: values,
        });
        return;
    }
    let Some(meta) = inner.pending_map.take() else {
        return;
    };
    let slice = resources.readback_buf.slice(..byte_size);
    let flag = state.mapped_ready.clone();
    slice.map_async(wgpu::MapMode::Read, move |map_result| {
        if map_result.is_ok() {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
    });
    inner.mapped = true;
    inner.mapped_meta = Some(meta);
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

/// Clear dirty flags after render graph compute passes have run.
fn clear_terrain_cache_dirty(
    mut near_dirty: ResMut<TerrainNearCacheDirty>,
    mut override_dirty: ResMut<TerrainOverrideCacheDirty>,
    mut far_dirty: ResMut<TerrainFarCacheDirty>,
    mut back_dirty: ResMut<TerrainBackCacheDirty>,
    mut chunk_dirty: ResMut<TerrainChunkGenerateDirty>,
) {
    near_dirty.0 = false;
    override_dirty.0 = false;
    far_dirty.0 = false;
    back_dirty.0 = false;
    chunk_dirty.0 = false;
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct TerrainGpuPlugin;

impl Plugin for TerrainGpuPlugin {
    fn build(&self, app: &mut App) {
        let chunk_readback = TerrainGeneratedChunkReadbackResult::default();
        let chunk_readback_for_render = chunk_readback.clone();
        app.init_resource::<TerrainNearUpdateRequest>()
            .init_resource::<TerrainCacheState>()
            .init_resource::<TerrainGeneratedChunkCache>()
            .init_resource::<TerrainGeneratedChunkReadbackRequest>()
            .insert_resource(chunk_readback)
            .add_systems(
                Update,
                prepare_terrain_near_update_request.in_set(SimUpdateSet::Rendering),
            )
            .add_systems(
                Update,
                (
                    apply_generated_chunk_readback_results,
                    drive_generated_chunk_readback_requests,
                )
                    .chain()
                    .in_set(SimUpdateSet::Interaction),
            )
            .add_plugins(ExtractResourcePlugin::<TerrainNearUpdateRequest>::default())
            .add_plugins(ExtractResourcePlugin::<TerrainGeneratedChunkReadbackRequest>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(chunk_readback_for_render)
            .init_resource::<SpecializedRenderPipelines<TerrainComposePipeline>>()
            .add_systems(
                RenderStartup,
                (
                    init_terrain_near_gpu_resources,
                    init_terrain_near_update_pipeline,
                    init_terrain_override_apply_pipeline,
                    init_terrain_chunk_generate_pipeline,
                    init_terrain_far_update_pipeline,
                    init_terrain_compose_pipeline,
                    link_terrain_and_water_graph,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_terrain_near_uploads,
                    prepare_terrain_chunk_generate_uploads,
                    prepare_terrain_compose_pipeline,
                )
                    .in_set(RenderSystems::Prepare),
            )
            .add_systems(
                Render,
                (readback_generated_chunks, clear_terrain_cache_dirty)
                    .chain()
                    .in_set(RenderSystems::Cleanup),
            )
            // TerrainComposeLabel: ViewNode in Core2d graph.
            .add_render_graph_node::<ViewNodeRunner<TerrainComposeNode>>(
                Core2d,
                TerrainComposeLabel,
            )
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::StartMainPass,
                    TerrainComposeLabel,
                    Node2d::MainTransparentPass,
                ),
            );

        // TerrainNearUpdateLabel / TerrainOverrideApplyLabel / TerrainFarUpdateLabel / TerrainBackUpdateLabel:
        // non-view compute nodes in main render graph.
        {
            let mut main_graph = render_app.world_mut().resource_mut::<RenderGraph>();
            main_graph.add_node(TerrainNearUpdateLabel, TerrainNearUpdateNode);
            main_graph.add_node(TerrainOverrideApplyLabel, TerrainOverrideApplyNode);
            main_graph.add_node(TerrainFarUpdateLabel, TerrainFarUpdateNode);
            main_graph.add_node(TerrainBackUpdateLabel, TerrainBackUpdateNode);
            main_graph.add_node(TerrainChunkGenerateLabel, TerrainChunkGenerateNode);
            main_graph.add_node_edge(TerrainNearUpdateLabel, TerrainOverrideApplyLabel);
            main_graph.add_node_edge(TerrainOverrideApplyLabel, TerrainFarUpdateLabel);
            main_graph.add_node_edge(TerrainFarUpdateLabel, TerrainBackUpdateLabel);
            main_graph.add_node_edge(TerrainBackUpdateLabel, TerrainChunkGenerateLabel);
            main_graph.add_node_edge(TerrainChunkGenerateLabel, CameraDriverLabel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CELL_SIZE_M, DEFAULT_SCREEN_SIZE_PX, FAR_MIN_DOWNSAMPLE, MAX_DIRTY_CELLS_PER_DISPATCH,
        TerrainCacheState, TerrainRuntimeSettings, compute_back_display_scale,
        compute_back_downsample_for_viewport_with_settings, compute_far_cache_base_extent,
        compute_far_downsample_for_viewport, compute_near_cache_extent,
        compute_near_cache_extent_from_viewport_world, default_viewport_world,
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

    #[test]
    fn far_extent_covers_viewport_with_margin() {
        let far = compute_far_cache_base_extent(DEFAULT_SCREEN_SIZE_PX);
        assert_eq!(far, UVec2::new(1760, 990));
    }

    #[test]
    fn far_downsample_tracks_viewport_coverage() {
        let far = compute_far_cache_base_extent(DEFAULT_SCREEN_SIZE_PX);
        let base_viewport = default_viewport_world(DEFAULT_SCREEN_SIZE_PX);
        assert_eq!(
            compute_far_downsample_for_viewport(base_viewport, far),
            FAR_MIN_DOWNSAMPLE
        );

        let zoomed_out_viewport = base_viewport * 27.0;
        assert_eq!(
            compute_far_downsample_for_viewport(zoomed_out_viewport, far),
            4
        );
    }

    #[test]
    fn back_downsample_converges_toward_far_rule_at_zoom_out() {
        let far = compute_far_cache_base_extent(DEFAULT_SCREEN_SIZE_PX);
        let zoomed_out_viewport = default_viewport_world(DEFAULT_SCREEN_SIZE_PX) * 27.0;
        let viewport_cells = zoomed_out_viewport / CELL_SIZE_M.max(1.0e-6);
        let front_mpp_cells_per_px = viewport_cells.y / (DEFAULT_SCREEN_SIZE_PX.y.max(1) as f32);
        let back_scale =
            compute_back_display_scale(front_mpp_cells_per_px, TerrainRuntimeSettings::default());
        let far_downsample = compute_far_downsample_for_viewport(zoomed_out_viewport, far);
        let back_downsample = compute_back_downsample_for_viewport_with_settings(
            zoomed_out_viewport * back_scale.max(1.0),
            far,
            TerrainRuntimeSettings::default(),
        );
        assert_eq!(far_downsample, 4);
        assert!(back_downsample >= far_downsample);

        let huge_zoomed_out_viewport = default_viewport_world(DEFAULT_SCREEN_SIZE_PX) * 512.0;
        let huge_cells = huge_zoomed_out_viewport / CELL_SIZE_M.max(1.0e-6);
        let huge_mpp = huge_cells.y / (DEFAULT_SCREEN_SIZE_PX.y.max(1) as f32);
        let huge_back_scale =
            compute_back_display_scale(huge_mpp, TerrainRuntimeSettings::default());
        let huge_far_downsample =
            compute_far_downsample_for_viewport(huge_zoomed_out_viewport, far);
        let huge_back_downsample = compute_back_downsample_for_viewport_with_settings(
            huge_zoomed_out_viewport * huge_back_scale.max(1.0),
            far,
            TerrainRuntimeSettings::default(),
        );
        assert_eq!(huge_back_downsample, huge_far_downsample);
    }

    #[test]
    fn back_downsample_expands_for_zoom_in_coverage() {
        let far = compute_far_cache_base_extent(DEFAULT_SCREEN_SIZE_PX);
        let zoomed_in_viewport = default_viewport_world(DEFAULT_SCREEN_SIZE_PX) * 0.45;
        let far_downsample = compute_far_downsample_for_viewport(zoomed_in_viewport, far);
        let viewport_cells = zoomed_in_viewport / CELL_SIZE_M.max(1.0e-6);
        let front_mpp_cells_per_px = viewport_cells.y / (DEFAULT_SCREEN_SIZE_PX.y.max(1) as f32);
        let back_scale =
            compute_back_display_scale(front_mpp_cells_per_px, TerrainRuntimeSettings::default());
        let back_downsample = compute_back_downsample_for_viewport_with_settings(
            zoomed_in_viewport * back_scale.max(1.0),
            far,
            TerrainRuntimeSettings::default(),
        );
        assert!(back_downsample >= far_downsample);
    }

    #[test]
    fn back_scale_floor_keeps_half_screen_resolution_target() {
        let settings = TerrainRuntimeSettings::default();
        let far = compute_far_cache_base_extent(DEFAULT_SCREEN_SIZE_PX);
        let zoomed_in_viewport = default_viewport_world(DEFAULT_SCREEN_SIZE_PX) * 0.45;
        let viewport_cells = zoomed_in_viewport / CELL_SIZE_M.max(1.0e-6);
        let front_mpp_cells_per_px = viewport_cells.y / (DEFAULT_SCREEN_SIZE_PX.y.max(1) as f32);
        let back_scale = compute_back_display_scale(front_mpp_cells_per_px, settings);
        let back_downsample = compute_back_downsample_for_viewport_with_settings(
            zoomed_in_viewport * back_scale.max(1.0),
            far,
            settings,
        );
        let back_texels_y = viewport_cells.y * back_scale / (back_downsample.max(1) as f32);
        let target_texels_y = (DEFAULT_SCREEN_SIZE_PX.y as f32)
            / settings.back_min_screen_resolution_divisor.max(1.0);
        assert!(back_texels_y + 1.0e-3 >= target_texels_y);
    }
}
