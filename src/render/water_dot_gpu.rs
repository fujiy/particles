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
    storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, uniform_buffer,
    uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntries, BindingResource, BlendComponent, BlendFactor, BlendOperation,
    BlendState, Buffer, BufferDescriptor, BufferUsages, CachedComputePipelineId,
    CachedPipelineState, CachedRenderPipelineId, ColorTargetState, ColorWrites,
    ComputePassDescriptor, ComputePipelineDescriptor, FragmentState, MultisampleState,
    PipelineCache, PrimitiveState, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat, TextureSampleType,
    VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::window::PrimaryWindow;
use bytemuck::{Pod, Zeroable};

use super::terrain_gpu::TerrainNearGpuResources;
use crate::camera_controller::MainCamera;
use crate::params::palette::{MaterialPalette4, PaletteColor};
use crate::params::{ActivePaletteParams, ActiveRenderParams};
use crate::physics::gpu_mpm::{MpmComputeLabel, gpu_resources::MpmGpuBuffers};
use crate::physics::profiler::{
    begin_gpu_pass_query, cpu_profile_span, end_gpu_pass_query, resolve_gpu_profiler_queries,
};
use crate::physics::world::constants::CELL_SIZE_M;

const WATER_DOT_RENDER_SHADER_PATH: &str = "shaders/render/water_dot_gpu_mainline.wgsl";
const WATER_DOT_PREPROCESS_SHADER_PATH: &str = "shaders/render/water_dot_preprocess.wgsl";
const DOTS_PER_CELL: u32 = 8;
const DOT_GRID_PADDING_CELLS: i32 = 32;
const DEFAULT_SPLAT_RADIUS_DOTS: f32 = 2.5;
const DEFAULT_BLUR_SIGMA_DOTS: f32 = 5.0;
const DEFAULT_BLUR_RADIUS_DOTS: u32 = 10;
const DEFAULT_DENSITY_ATOMIC_SCALE: f32 = 256.0;
const DEFAULT_DENSITY_THRESHOLD: f32 = 0.50;
const DEFAULT_WATER_PALETTE_SEED: u32 = 0xA53C_9E4D;
const WORKGROUP_SIZE: u32 = 64;
const MAX_COMPUTE_WORKGROUPS_PER_DIM: u32 = 65_535;
const MAX_DOT_COUNT: u64 = WORKGROUP_SIZE as u64 * MAX_COMPUTE_WORKGROUPS_PER_DIM as u64;
const DEFAULT_SCREEN_SIZE_PX: UVec2 = UVec2::new(1280, 720);
const DEFAULT_CAMERA_VIEWPORT_HEIGHT_M: f32 = 14.0;
const DOT_GRID_ORIGIN_SNAP_CELLS: i32 = 16;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct WaterDotPreprocessLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct WaterDotGpuLabel;

#[derive(Default)]
struct WaterDotPreprocessNode;

#[derive(Default)]
struct WaterDotGpuNode;

#[derive(Clone, Copy, Debug, PartialEq)]
struct DotGridLayout {
    width: u32,
    height: u32,
    origin: Vec2,
    dot_size_m: f32,
}

impl DotGridLayout {
    fn dot_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

fn default_viewport_world(screen_size_px: UVec2) -> Vec2 {
    let size = screen_size_px.max(UVec2::splat(1)).as_vec2();
    let aspect = size.x / size.y.max(1.0);
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

fn compute_world_locked_dot_grid_layout(camera_pos: Vec2, viewport_world_m: Vec2) -> DotGridLayout {
    let base_dot_size_m = CELL_SIZE_M / DOTS_PER_CELL as f32;
    let padding_world_m = Vec2::splat(DOT_GRID_PADDING_CELLS as f32 * CELL_SIZE_M);
    let snap_world_m = DOT_GRID_ORIGIN_SNAP_CELLS as f32 * CELL_SIZE_M;
    let required_span_world_m = (viewport_world_m.max(Vec2::splat(base_dot_size_m))
        + padding_world_m * 2.0
        + Vec2::splat(snap_world_m))
    .max(Vec2::splat(base_dot_size_m));
    let mut resolution_scale = 1u32;
    let mut dot_size_m = base_dot_size_m;
    let mut width = (required_span_world_m.x / dot_size_m).ceil().max(1.0) as u32;
    let mut height = (required_span_world_m.y / dot_size_m).ceil().max(1.0) as u32;
    while u64::from(width) * u64::from(height) > MAX_DOT_COUNT {
        resolution_scale = resolution_scale.saturating_add(1);
        dot_size_m = base_dot_size_m * resolution_scale as f32;
        width = (required_span_world_m.x / dot_size_m).ceil().max(1.0) as u32;
        height = (required_span_world_m.y / dot_size_m).ceil().max(1.0) as u32;
    }
    let min_world_m = camera_pos - viewport_world_m * 0.5 - padding_world_m;
    let origin = (min_world_m / snap_world_m).floor() * snap_world_m;
    DotGridLayout {
        width,
        height,
        origin,
        dot_size_m,
    }
}

fn dot_grid_layout() -> DotGridLayout {
    compute_world_locked_dot_grid_layout(Vec2::ZERO, default_viewport_world(DEFAULT_SCREEN_SIZE_PX))
}

#[derive(Resource, Clone)]
struct WaterDotLayoutRequest {
    layout: DotGridLayout,
}

impl Default for WaterDotLayoutRequest {
    fn default() -> Self {
        Self {
            layout: dot_grid_layout(),
        }
    }
}

impl ExtractResource for WaterDotLayoutRequest {
    type Source = WaterDotLayoutRequest;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WaterDotParams {
    origin_x: f32,
    origin_y: f32,
    dot_size_m: f32,
    density_threshold: f32,
    atomic_scale: f32,
    splat_radius_dots: f32,
    blur_sigma_dots: f32,
    _pad0: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    blur_radius_dots: u32,
    palette_seed: u32,
    _pad1: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<WaterDotParams>() == 64);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WaterDotPaletteParams {
    water: [[f32; 4]; 4],
    stone: [[f32; 4]; 4],
    soil: [[f32; 4]; 4],
    sand: [[f32; 4]; 4],
    grass: [[f32; 4]; 4],
}

const _: () = assert!(std::mem::size_of::<WaterDotPaletteParams>() == 320);

#[derive(Resource)]
struct WaterDotGpuResources {
    layout: DotGridLayout,
    capacity_dot_count: u64,
    params_buf: Buffer,
    palette_buf: Buffer,
    fallback_particle_buf: Buffer,
    density_atomic_water_buf: Buffer,
    density_atomic_stone_buf: Buffer,
    density_atomic_soil_buf: Buffer,
    density_atomic_sand_buf: Buffer,
    density_atomic_grass_buf: Buffer,
    blur_tmp_buf: Buffer,
    blurred_density_water_buf: Buffer,
    blurred_density_stone_buf: Buffer,
    blurred_density_soil_buf: Buffer,
    blurred_density_sand_buf: Buffer,
    blurred_density_grass_buf: Buffer,
}

impl WaterDotGpuResources {
    fn params_for(
        &self,
        particle_count: u32,
        active_render: Option<&ActiveRenderParams>,
    ) -> WaterDotParams {
        let (
            density_threshold,
            atomic_scale,
            splat_radius_dots,
            blur_sigma_dots,
            blur_radius_dots,
            palette_seed,
        ) = active_render
            .map(|r| {
                (
                    r.0.water.splat.density_threshold,
                    r.0.water.splat.atomic_scale,
                    r.0.water.splat.radius_dots,
                    r.0.water.splat.blur_sigma_dots,
                    r.0.water.splat.blur_radius_dots,
                    r.0.water.palette_seed,
                )
            })
            .unwrap_or((
                DEFAULT_DENSITY_THRESHOLD,
                DEFAULT_DENSITY_ATOMIC_SCALE,
                DEFAULT_SPLAT_RADIUS_DOTS,
                DEFAULT_BLUR_SIGMA_DOTS,
                DEFAULT_BLUR_RADIUS_DOTS,
                DEFAULT_WATER_PALETTE_SEED,
            ));
        WaterDotParams {
            origin_x: self.layout.origin.x,
            origin_y: self.layout.origin.y,
            dot_size_m: self.layout.dot_size_m,
            density_threshold,
            atomic_scale,
            splat_radius_dots,
            blur_sigma_dots,
            _pad0: 0.0,
            width: self.layout.width,
            height: self.layout.height,
            particle_count,
            blur_radius_dots,
            palette_seed,
            _pad1: [0; 3],
        }
    }

    fn update_layout(&mut self, render_device: &RenderDevice, layout: DotGridLayout) {
        let required_dot_count = layout.dot_count();
        if required_dot_count > self.capacity_dot_count {
            let byte_size = required_dot_count * 4;
            self.density_atomic_water_buf =
                create_storage_buffer(render_device, "water_dot_density_atomic_water", byte_size);
            self.density_atomic_stone_buf =
                create_storage_buffer(render_device, "water_dot_density_atomic_stone", byte_size);
            self.density_atomic_soil_buf =
                create_storage_buffer(render_device, "water_dot_density_atomic_soil", byte_size);
            self.density_atomic_sand_buf =
                create_storage_buffer(render_device, "water_dot_density_atomic_sand", byte_size);
            self.density_atomic_grass_buf =
                create_storage_buffer(render_device, "water_dot_density_atomic_grass", byte_size);
            self.blur_tmp_buf =
                create_storage_buffer(render_device, "water_dot_blur_tmp", byte_size);
            self.blurred_density_water_buf =
                create_storage_buffer(render_device, "water_dot_blurred_density_water", byte_size);
            self.blurred_density_stone_buf =
                create_storage_buffer(render_device, "water_dot_blurred_density_stone", byte_size);
            self.blurred_density_soil_buf =
                create_storage_buffer(render_device, "water_dot_blurred_density_soil", byte_size);
            self.blurred_density_sand_buf =
                create_storage_buffer(render_device, "water_dot_blurred_density_sand", byte_size);
            self.blurred_density_grass_buf =
                create_storage_buffer(render_device, "water_dot_blurred_density_grass", byte_size);
            self.capacity_dot_count = required_dot_count;
        }
        self.layout = layout;
    }
}

fn create_storage_buffer(render_device: &RenderDevice, label: &str, size: u64) -> Buffer {
    render_device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn update_water_dot_layout_request(
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Transform, &Projection), With<MainCamera>>,
    mut request: ResMut<WaterDotLayoutRequest>,
) {
    let window_size_px = windows
        .iter()
        .next()
        .map(|window| {
            UVec2::new(
                window.width().max(1.0).round() as u32,
                window.height().max(1.0).round() as u32,
            )
        })
        .unwrap_or(DEFAULT_SCREEN_SIZE_PX);
    let layout = cameras
        .iter()
        .next()
        .map(|(transform, projection)| {
            let viewport_world = projection_viewport_world(
                projection,
                window_size_px.max(UVec2::splat(1)).as_vec2(),
            )
            .unwrap_or_else(|| default_viewport_world(window_size_px));
            compute_world_locked_dot_grid_layout(transform.translation.xy(), viewport_world)
        })
        .unwrap_or_else(|| dot_grid_layout());
    if request.layout != layout {
        request.layout = layout;
    }
}

fn prepare_water_dot_layout(
    request: Res<WaterDotLayoutRequest>,
    render_device: Res<RenderDevice>,
    mut resources: ResMut<WaterDotGpuResources>,
) {
    resources.update_layout(&render_device, request.layout);
}

fn srgb_u8_to_linear(c: u8) -> f32 {
    let x = c as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn palette4_to_linear_rows(palette: &MaterialPalette4) -> [[f32; 4]; 4] {
    let to_row = |c: PaletteColor| -> [f32; 4] {
        [
            srgb_u8_to_linear(c.r),
            srgb_u8_to_linear(c.g),
            srgb_u8_to_linear(c.b),
            1.0,
        ]
    };
    [
        to_row(palette.colors[0]),
        to_row(palette.colors[1]),
        to_row(palette.colors[2]),
        to_row(palette.colors[3]),
    ]
}

fn palette_uniform_from_active(active: Option<&ActivePaletteParams>) -> WaterDotPaletteParams {
    let source = active.map(|p| &p.0).cloned().unwrap_or_default();
    WaterDotPaletteParams {
        water: palette4_to_linear_rows(&source.water),
        stone: palette4_to_linear_rows(&source.stone),
        soil: palette4_to_linear_rows(&source.soil),
        sand: palette4_to_linear_rows(&source.sand),
        grass: palette4_to_linear_rows(&source.grass),
    }
}

#[derive(Resource)]
struct WaterDotPreprocessPipelines {
    layout: BindGroupLayout,
    clear_pipeline: CachedComputePipelineId,
    splat_pipeline: CachedComputePipelineId,
    blur_x_water_pipeline: CachedComputePipelineId,
    blur_y_water_pipeline: CachedComputePipelineId,
    blur_x_stone_pipeline: CachedComputePipelineId,
    blur_y_stone_pipeline: CachedComputePipelineId,
    blur_x_soil_pipeline: CachedComputePipelineId,
    blur_y_soil_pipeline: CachedComputePipelineId,
    blur_x_sand_pipeline: CachedComputePipelineId,
    blur_y_sand_pipeline: CachedComputePipelineId,
    blur_x_grass_pipeline: CachedComputePipelineId,
    blur_y_grass_pipeline: CachedComputePipelineId,
}

#[derive(Resource)]
struct WaterDotGpuPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct WaterDotPipelineKey {
    target_format: TextureFormat,
    sample_count: u32,
}

impl SpecializedRenderPipeline for WaterDotGpuPipeline {
    type Key = WaterDotPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("water_dot_gpu_pipeline".into()),
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
struct ViewWaterDotGpuPipeline {
    id: CachedRenderPipelineId,
}

fn init_water_dot_gpu_compute_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = dot_grid_layout();
    let dot_count = layout.dot_count();

    let params_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_params"),
        size: 64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let palette_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_palette_params"),
        size: 320,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let fallback_particle_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_fallback_particles"),
        size: 72,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let density_atomic_water_buf = create_storage_buffer(
        &render_device,
        "water_dot_density_atomic_water",
        dot_count * 4,
    );
    let density_atomic_stone_buf = create_storage_buffer(
        &render_device,
        "water_dot_density_atomic_stone",
        dot_count * 4,
    );
    let density_atomic_soil_buf = create_storage_buffer(
        &render_device,
        "water_dot_density_atomic_soil",
        dot_count * 4,
    );
    let density_atomic_sand_buf = create_storage_buffer(
        &render_device,
        "water_dot_density_atomic_sand",
        dot_count * 4,
    );
    let density_atomic_grass_buf = create_storage_buffer(
        &render_device,
        "water_dot_density_atomic_grass",
        dot_count * 4,
    );
    let blur_tmp_buf = create_storage_buffer(&render_device, "water_dot_blur_tmp", dot_count * 4);
    let blurred_density_water_buf = create_storage_buffer(
        &render_device,
        "water_dot_blurred_density_water",
        dot_count * 4,
    );
    let blurred_density_stone_buf = create_storage_buffer(
        &render_device,
        "water_dot_blurred_density_stone",
        dot_count * 4,
    );
    let blurred_density_soil_buf = create_storage_buffer(
        &render_device,
        "water_dot_blurred_density_soil",
        dot_count * 4,
    );
    let blurred_density_sand_buf = create_storage_buffer(
        &render_device,
        "water_dot_blurred_density_sand",
        dot_count * 4,
    );
    let blurred_density_grass_buf = create_storage_buffer(
        &render_device,
        "water_dot_blurred_density_grass",
        dot_count * 4,
    );

    commands.insert_resource(WaterDotGpuResources {
        layout,
        capacity_dot_count: dot_count,
        params_buf,
        palette_buf,
        fallback_particle_buf,
        density_atomic_water_buf,
        density_atomic_stone_buf,
        density_atomic_soil_buf,
        density_atomic_sand_buf,
        density_atomic_grass_buf,
        blur_tmp_buf,
        blurred_density_water_buf,
        blurred_density_stone_buf,
        blurred_density_soil_buf,
        blurred_density_sand_buf,
        blurred_density_grass_buf,
    });

    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            uniform_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
        ),
    );
    let layout = render_device.create_bind_group_layout("water_dot_preprocess_layout", &*entries);

    let shader = asset_server.load(WATER_DOT_PREPROCESS_SHADER_PATH);
    let clear_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_clear".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("clear_density".into()),
        zero_initialize_workgroup_memory: false,
    });
    let splat_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_splat".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("splat_particles".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_x_water_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_x_water".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_x_water".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_y_water_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_y_water".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_y_water".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_x_stone_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_x_stone".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_x_stone".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_y_stone_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_y_stone".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_y_stone".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_x_soil_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_x_soil".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_x_soil".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_y_soil_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_y_soil".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_y_soil".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_x_sand_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_x_sand".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_x_sand".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_y_sand_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_y_sand".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_y_sand".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_x_grass_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_x_grass".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_x_grass".into()),
        zero_initialize_workgroup_memory: false,
    });
    let blur_y_grass_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("water_dot_blur_y_grass".into()),
        layout: vec![BindGroupLayoutDescriptor::new(
            "water_dot_preprocess_layout",
            &*entries,
        )],
        push_constant_ranges: vec![],
        shader: shader.clone(),
        shader_defs: vec![],
        entry_point: Some("blur_y_grass".into()),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(WaterDotPreprocessPipelines {
        layout,
        clear_pipeline,
        splat_pipeline,
        blur_x_water_pipeline,
        blur_y_water_pipeline,
        blur_x_stone_pipeline,
        blur_y_stone_pipeline,
        blur_x_soil_pipeline,
        blur_y_soil_pipeline,
        blur_x_sand_pipeline,
        blur_y_sand_pipeline,
        blur_x_grass_pipeline,
        blur_y_grass_pipeline,
    });
}

fn init_water_dot_gpu_render_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "water_dot_gpu_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer_sized(false, core::num::NonZeroU64::new(64)),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                uniform_buffer_sized(false, core::num::NonZeroU64::new(320)),
                uniform_buffer_sized(false, None),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Uint),
            ),
        ),
    );

    commands.insert_resource(WaterDotGpuPipeline {
        bind_group_layout,
        shader: asset_server.load(WATER_DOT_RENDER_SHADER_PATH),
    });
}

fn prepare_water_dot_gpu_pipeline(
    mut commands: Commands,
    pipeline: Res<WaterDotGpuPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<WaterDotGpuPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(Entity, &ViewTarget, Option<&Msaa>)>,
) {
    for (entity, view_target, msaa) in &views {
        let id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            WaterDotPipelineKey {
                target_format: view_target.main_texture_format(),
                sample_count: msaa.map_or(1, Msaa::samples),
            },
        );
        commands
            .entity(entity)
            .insert(ViewWaterDotGpuPipeline { id });
    }
}

fn warn_missing_compute_pipeline_once(
    warned: &std::sync::atomic::AtomicBool,
    name: &str,
    id: CachedComputePipelineId,
    cache: &PipelineCache,
) {
    if warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let state = cache.get_compute_pipeline_state(id);
    match state {
        CachedPipelineState::Queued => warn!("water_dot_gpu: {name} pipeline is queued"),
        CachedPipelineState::Creating(_) => warn!("water_dot_gpu: {name} pipeline is creating"),
        CachedPipelineState::Err(err) => warn!("water_dot_gpu: {name} pipeline failed: {err}"),
        CachedPipelineState::Ok(_) => warn!("water_dot_gpu: {name} pipeline missing unexpectedly"),
    }
}

impl Node for WaterDotPreprocessNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let _profile_span = cpu_profile_span("water", "preprocess_node").entered();
        let Some(resources) = world.get_resource::<WaterDotGpuResources>() else {
            return Ok(());
        };
        let Some(pipelines) = world.get_resource::<WaterDotPreprocessPipelines>() else {
            return Ok(());
        };
        let Some(mpm_buffers) = world.get_resource::<MpmGpuBuffers>() else {
            return Ok(());
        };

        let particle_count = mpm_buffers.particle_count;
        static HAD_PARTICLES_PREVIOUS_FRAME: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        let had_particles_previous_frame =
            HAD_PARTICLES_PREVIOUS_FRAME.load(std::sync::atomic::Ordering::Relaxed);
        let params =
            resources.params_for(particle_count, world.get_resource::<ActiveRenderParams>());
        world.resource::<RenderQueue>().write_buffer(
            &resources.params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        let palette_params =
            palette_uniform_from_active(world.get_resource::<ActivePaletteParams>());
        world.resource::<RenderQueue>().write_buffer(
            &resources.palette_buf,
            0,
            bytemuck::bytes_of(&palette_params),
        );

        let pipeline_cache = world.resource::<PipelineCache>();
        static CLEAR_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static SPLAT_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_X_WATER_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_Y_WATER_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_X_STONE_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_Y_STONE_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_X_SOIL_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_Y_SOIL_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_X_SAND_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_Y_SAND_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_X_GRASS_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_Y_GRASS_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);

        let Some(clear_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.clear_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &CLEAR_WARNED,
                "clear",
                pipelines.clear_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        if particle_count == 0 && !had_particles_previous_frame {
            return Ok(());
        }

        let particle_binding = if particle_count > 0 {
            mpm_buffers.particle_buf.as_entire_binding()
        } else {
            resources.fallback_particle_buf.as_entire_binding()
        };

        let bind_group: BindGroup = render_context.render_device().create_bind_group(
            "water_dot_preprocess_bind_group",
            &pipelines.layout,
            &BindGroupEntries::sequential((
                resources.params_buf.as_entire_binding(),
                particle_binding,
                resources.density_atomic_water_buf.as_entire_binding(),
                resources.density_atomic_stone_buf.as_entire_binding(),
                resources.density_atomic_soil_buf.as_entire_binding(),
                resources.density_atomic_sand_buf.as_entire_binding(),
                resources.density_atomic_grass_buf.as_entire_binding(),
                resources.blur_tmp_buf.as_entire_binding(),
                resources.blurred_density_water_buf.as_entire_binding(),
                resources.blurred_density_stone_buf.as_entire_binding(),
                resources.blurred_density_soil_buf.as_entire_binding(),
                resources.blurred_density_sand_buf.as_entire_binding(),
                resources.blurred_density_grass_buf.as_entire_binding(),
            )),
        );

        let dot_count = resources.layout.dot_count() as u32;
        let dot_workgroups = dot_count.div_ceil(WORKGROUP_SIZE);
        let particle_workgroups = particle_count.div_ceil(WORKGROUP_SIZE);

        let encoder = render_context.command_encoder();

        {
            let profile_query = begin_gpu_pass_query(world, "water", "clear", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_clear"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(clear_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        if particle_count == 0 {
            HAD_PARTICLES_PREVIOUS_FRAME.store(false, std::sync::atomic::Ordering::Relaxed);
            resolve_gpu_profiler_queries(world, encoder);
            return Ok(());
        }

        let Some(splat_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.splat_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &SPLAT_WARNED,
                "splat",
                pipelines.splat_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_x_water_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_x_water_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_X_WATER_WARNED,
                "blur_x_water",
                pipelines.blur_x_water_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_y_water_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_y_water_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_Y_WATER_WARNED,
                "blur_y_water",
                pipelines.blur_y_water_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_x_stone_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_x_stone_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_X_STONE_WARNED,
                "blur_x_stone",
                pipelines.blur_x_stone_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_y_stone_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_y_stone_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_Y_STONE_WARNED,
                "blur_y_stone",
                pipelines.blur_y_stone_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_x_soil_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_x_soil_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_X_SOIL_WARNED,
                "blur_x_soil",
                pipelines.blur_x_soil_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_y_soil_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_y_soil_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_Y_SOIL_WARNED,
                "blur_y_soil",
                pipelines.blur_y_soil_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_x_sand_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_x_sand_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_X_SAND_WARNED,
                "blur_x_sand",
                pipelines.blur_x_sand_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_y_sand_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_y_sand_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_Y_SAND_WARNED,
                "blur_y_sand",
                pipelines.blur_y_sand_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_x_grass_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_x_grass_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_X_GRASS_WARNED,
                "blur_x_grass",
                pipelines.blur_x_grass_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };
        let Some(blur_y_grass_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.blur_y_grass_pipeline)
        else {
            warn_missing_compute_pipeline_once(
                &BLUR_Y_GRASS_WARNED,
                "blur_y_grass",
                pipelines.blur_y_grass_pipeline,
                pipeline_cache,
            );
            return Ok(());
        };

        {
            let profile_query = begin_gpu_pass_query(world, "water", "splat", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_splat"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(splat_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_x_water", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_x_water"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_x_water_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_y_water", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_y_water"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_y_water_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_x_stone", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_x_stone"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_x_stone_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_y_stone", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_y_stone"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_y_stone_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_x_soil", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_x_soil"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_x_soil_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_y_soil", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_y_soil"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_y_soil_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_x_sand", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_x_sand"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_x_sand_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_y_sand", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_y_sand"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_y_sand_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_x_grass", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_x_grass"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_x_grass_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        {
            let profile_query = begin_gpu_pass_query(world, "water", "blur_y_grass", encoder);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_y_grass"),
                timestamp_writes: profile_query
                    .as_ref()
                    .and_then(|query| query.compute_pass_timestamp_writes()),
            });
            pass.set_pipeline(blur_y_grass_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
            drop(pass);
            end_gpu_pass_query(world, encoder, profile_query);
        }

        HAD_PARTICLES_PREVIOUS_FRAME.store(true, std::sync::atomic::Ordering::Relaxed);
        resolve_gpu_profiler_queries(world, encoder);

        Ok(())
    }
}

impl ViewNode for WaterDotGpuNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static ViewWaterDotGpuPipeline,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, view_pipeline): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let _profile_span = cpu_profile_span("water", "render_node").entered();
        let Some(resources) = world.get_resource::<WaterDotGpuResources>() else {
            return Ok(());
        };
        let Some(terrain_resources) = world.get_resource::<TerrainNearGpuResources>() else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(view_pipeline.id) else {
            static PIPELINE_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !PIPELINE_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                warn!("water_dot_gpu: render pipeline is not ready");
            }
            return Ok(());
        };

        let render_pipeline = world.resource::<WaterDotGpuPipeline>();
        let view_uniforms_resource = world.resource::<ViewUniforms>();
        let Some(view_binding) = view_uniforms_resource.uniforms.binding() else {
            static VIEW_UNIFORM_WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !VIEW_UNIFORM_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                warn!("water_dot_gpu: view uniform binding unavailable");
            }
            return Ok(());
        };

        let bind_group = render_context.render_device().create_bind_group(
            "water_dot_gpu_bind_group",
            &pipeline_cache.get_bind_group_layout(&render_pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                view_binding,
                resources.params_buf.as_entire_binding(),
                resources.blurred_density_water_buf.as_entire_binding(),
                resources.blurred_density_stone_buf.as_entire_binding(),
                resources.blurred_density_soil_buf.as_entire_binding(),
                resources.blurred_density_sand_buf.as_entire_binding(),
                resources.blurred_density_grass_buf.as_entire_binding(),
                resources.palette_buf.as_entire_binding(),
                terrain_resources.compose_params_buf.as_entire_binding(),
                BindingResource::TextureView(&terrain_resources.near_cache.near_texture_view),
                BindingResource::TextureView(&terrain_resources.override_cache.near_texture_view),
            )),
        );

        let profile_query = begin_gpu_pass_query(
            world,
            "water",
            "render",
            render_context.command_encoder(),
        );
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("water_dot_gpu_pass"),
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

pub struct WaterDotGpuPlugin;

impl Plugin for WaterDotGpuPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<ActivePaletteParams>::default());
        app.add_plugins(ExtractResourcePlugin::<ActiveRenderParams>::default());
        app.init_resource::<WaterDotLayoutRequest>()
            .add_plugins(ExtractResourcePlugin::<WaterDotLayoutRequest>::default())
            .add_systems(PostUpdate, update_water_dot_layout_request);
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SpecializedRenderPipelines<WaterDotGpuPipeline>>()
            .add_systems(
                RenderStartup,
                (
                    init_water_dot_gpu_compute_resources,
                    init_water_dot_gpu_render_pipeline,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_water_dot_layout.in_set(RenderSystems::PrepareResources),
                    prepare_water_dot_gpu_pipeline.in_set(RenderSystems::Prepare),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<WaterDotGpuNode>>(Core2d, WaterDotGpuLabel)
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::StartMainPass,
                    WaterDotGpuLabel,
                    Node2d::MainTransparentPass,
                ),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(WaterDotPreprocessLabel, WaterDotPreprocessNode);
        let _ = render_graph.try_add_node_edge(MpmComputeLabel, WaterDotPreprocessLabel);
        render_graph.add_node_edge(WaterDotPreprocessLabel, CameraDriverLabel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_locked_layout_covers_far_camera_with_padding() {
        let viewport_world_m = Vec2::new(24.0, 14.0);
        let camera_pos = Vec2::new(40.0, -6.0);
        let padding_world_m = DOT_GRID_PADDING_CELLS as f32 * CELL_SIZE_M;
        let layout = compute_world_locked_dot_grid_layout(camera_pos, viewport_world_m);
        let max_world_m = layout.origin
            + Vec2::new(layout.width as f32, layout.height as f32) * layout.dot_size_m;

        assert!(layout.origin.x <= camera_pos.x - viewport_world_m.x * 0.5 - padding_world_m);
        assert!(layout.origin.y <= camera_pos.y - viewport_world_m.y * 0.5 - padding_world_m);
        assert!(max_world_m.x >= camera_pos.x + viewport_world_m.x * 0.5 + padding_world_m);
        assert!(max_world_m.y >= camera_pos.y + viewport_world_m.y * 0.5 + padding_world_m);
    }

    #[test]
    fn world_locked_layout_origin_moves_in_chunk_steps() {
        let snap_world_m = DOT_GRID_ORIGIN_SNAP_CELLS as f32 * CELL_SIZE_M;
        let viewport_world_m = Vec2::new(24.0, 14.0);
        let base_camera_pos = Vec2::new(snap_world_m * 5.0, snap_world_m * 2.0);
        let layout_a = compute_world_locked_dot_grid_layout(base_camera_pos, viewport_world_m);
        let layout_b = compute_world_locked_dot_grid_layout(
            base_camera_pos + Vec2::splat(snap_world_m * 0.49),
            viewport_world_m,
        );
        let layout_c = compute_world_locked_dot_grid_layout(
            base_camera_pos + Vec2::new(snap_world_m * 1.01, 0.0),
            viewport_world_m,
        );

        assert_eq!(layout_a.origin, layout_b.origin);
        assert!((layout_c.origin.x - layout_a.origin.x - snap_world_m).abs() < 1.0e-6);
    }

    #[test]
    fn world_locked_layout_caps_dot_count_for_large_viewports() {
        let layout = compute_world_locked_dot_grid_layout(Vec2::ZERO, Vec2::new(1200.0, 1200.0));

        assert!(layout.dot_count() <= MAX_DOT_COUNT);
        assert!(layout.dot_size_m >= CELL_SIZE_M / DOTS_PER_CELL as f32);
    }
}
