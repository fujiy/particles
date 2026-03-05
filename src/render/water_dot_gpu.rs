use bevy::asset::AssetServer;
use bevy::core_pipeline::core_2d::graph::{Core2d, Node2d};
use bevy::ecs::query::QueryItem;
use bevy::log::warn;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode,
    ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntries, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer,
    BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor,
    ComputePipelineDescriptor, FragmentState, MultisampleState, PipelineCache, PrimitiveState,
    RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, SpecializedRenderPipeline,
    SpecializedRenderPipelines, TextureFormat, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::view::{Msaa, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bytemuck::{Pod, Zeroable};

use crate::params::palette::{MaterialPalette4, PaletteColor};
use crate::params::{ActivePaletteParams, ActiveRenderParams};
use crate::physics::gpu_mpm::{MpmComputeLabel, gpu_resources::MpmGpuBuffers};
use crate::physics::world::constants::{
    CELL_SIZE_M, CHUNK_SIZE_I32, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X,
    WORLD_MIN_CHUNK_Y,
};

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

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct WaterDotPreprocessLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct WaterDotGpuLabel;

#[derive(Default)]
struct WaterDotPreprocessNode;

#[derive(Default)]
struct WaterDotGpuNode;

#[derive(Clone, Copy, Debug)]
struct DotGridLayout {
    width: u32,
    height: u32,
    origin: Vec2,
    dot_size_m: f32,
}

impl DotGridLayout {
    fn dot_count(&self) -> u32 {
        self.width * self.height
    }
}

fn dot_grid_layout() -> DotGridLayout {
    let min_cell = IVec2::new(
        WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32 - DOT_GRID_PADDING_CELLS,
        WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32 - DOT_GRID_PADDING_CELLS,
    );
    let max_cell_exclusive = IVec2::new(
        (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 + DOT_GRID_PADDING_CELLS,
        (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 + DOT_GRID_PADDING_CELLS,
    );
    let span_cells = (max_cell_exclusive - min_cell).max(IVec2::ONE);

    DotGridLayout {
        width: span_cells.x as u32 * DOTS_PER_CELL,
        height: span_cells.y as u32 * DOTS_PER_CELL,
        origin: min_cell.as_vec2() * CELL_SIZE_M,
        dot_size_m: CELL_SIZE_M / DOTS_PER_CELL as f32,
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
    soil: [[f32; 4]; 4],
    sand: [[f32; 4]; 4],
}

const _: () = assert!(std::mem::size_of::<WaterDotPaletteParams>() == 192);

#[derive(Resource)]
struct WaterDotGpuResources {
    layout: DotGridLayout,
    params_buf: Buffer,
    palette_buf: Buffer,
    fallback_particle_buf: Buffer,
    density_atomic_water_buf: Buffer,
    density_atomic_soil_buf: Buffer,
    density_atomic_sand_buf: Buffer,
    blur_tmp_buf: Buffer,
    blurred_density_water_buf: Buffer,
    blurred_density_soil_buf: Buffer,
    blurred_density_sand_buf: Buffer,
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
        soil: palette4_to_linear_rows(&source.soil),
        sand: palette4_to_linear_rows(&source.sand),
    }
}

#[derive(Resource)]
struct WaterDotPreprocessPipelines {
    layout: BindGroupLayout,
    clear_pipeline: CachedComputePipelineId,
    splat_pipeline: CachedComputePipelineId,
    blur_x_water_pipeline: CachedComputePipelineId,
    blur_y_water_pipeline: CachedComputePipelineId,
    blur_x_soil_pipeline: CachedComputePipelineId,
    blur_y_soil_pipeline: CachedComputePipelineId,
    blur_x_sand_pipeline: CachedComputePipelineId,
    blur_y_sand_pipeline: CachedComputePipelineId,
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
    let dot_count = layout.dot_count() as u64;

    let params_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_params"),
        size: 64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let palette_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_palette_params"),
        size: 192,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let fallback_particle_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_fallback_particles"),
        size: 72,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let density_atomic_water_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_density_atomic_water"),
        size: dot_count * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let density_atomic_soil_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_density_atomic_soil"),
        size: dot_count * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let density_atomic_sand_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_density_atomic_sand"),
        size: dot_count * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let blur_tmp_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_blur_tmp"),
        size: dot_count * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let blurred_density_water_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_blurred_density_water"),
        size: dot_count * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let blurred_density_soil_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_blurred_density_soil"),
        size: dot_count * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let blurred_density_sand_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("water_dot_blurred_density_sand"),
        size: dot_count * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    commands.insert_resource(WaterDotGpuResources {
        layout,
        params_buf,
        palette_buf,
        fallback_particle_buf,
        density_atomic_water_buf,
        density_atomic_soil_buf,
        density_atomic_sand_buf,
        blur_tmp_buf,
        blurred_density_water_buf,
        blurred_density_soil_buf,
        blurred_density_sand_buf,
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
        shader,
        shader_defs: vec![],
        entry_point: Some("blur_y_sand".into()),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(WaterDotPreprocessPipelines {
        layout,
        clear_pipeline,
        splat_pipeline,
        blur_x_water_pipeline,
        blur_y_water_pipeline,
        blur_x_soil_pipeline,
        blur_y_soil_pipeline,
        blur_x_sand_pipeline,
        blur_y_sand_pipeline,
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
                uniform_buffer_sized(false, core::num::NonZeroU64::new(192)),
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
        static BLUR_X_SOIL_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_Y_SOIL_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_X_SAND_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        static BLUR_Y_SAND_WARNED: std::sync::atomic::AtomicBool =
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
                resources.density_atomic_soil_buf.as_entire_binding(),
                resources.density_atomic_sand_buf.as_entire_binding(),
                resources.blur_tmp_buf.as_entire_binding(),
                resources.blurred_density_water_buf.as_entire_binding(),
                resources.blurred_density_soil_buf.as_entire_binding(),
                resources.blurred_density_sand_buf.as_entire_binding(),
            )),
        );

        let dot_count = resources.layout.dot_count();
        let dot_workgroups = dot_count.div_ceil(WORKGROUP_SIZE);
        let particle_workgroups = particle_count.div_ceil(WORKGROUP_SIZE);

        let encoder = render_context.command_encoder();

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(clear_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
        }

        if particle_count == 0 {
            HAD_PARTICLES_PREVIOUS_FRAME.store(false, std::sync::atomic::Ordering::Relaxed);
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

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_splat"),
                timestamp_writes: None,
            });
            pass.set_pipeline(splat_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_x_water"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blur_x_water_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_y_water"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blur_y_water_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_x_soil"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blur_x_soil_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_y_soil"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blur_y_soil_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_x_sand"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blur_x_sand_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_dot_blur_y_sand"),
                timestamp_writes: None,
            });
            pass.set_pipeline(blur_y_sand_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dot_workgroups, 1, 1);
        }

        HAD_PARTICLES_PREVIOUS_FRAME.store(true, std::sync::atomic::Ordering::Relaxed);

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
        let Some(resources) = world.get_resource::<WaterDotGpuResources>() else {
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
                resources.blurred_density_soil_buf.as_entire_binding(),
                resources.blurred_density_sand_buf.as_entire_binding(),
                resources.palette_buf.as_entire_binding(),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("water_dot_gpu_pass"),
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

pub struct WaterDotGpuPlugin;

impl Plugin for WaterDotGpuPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<ActivePaletteParams>::default());
        app.add_plugins(ExtractResourcePlugin::<ActiveRenderParams>::default());
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
                prepare_water_dot_gpu_pipeline.in_set(RenderSystems::Prepare),
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
