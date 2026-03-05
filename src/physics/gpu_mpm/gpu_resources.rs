// GPU buffer resource for MLS-MPM.
//
// Holds GPU-resident particle and grid buffers.
// CPU uploads happen once at init and whenever particles are spawned/removed.
// Per-frame: params uniform is updated; particle/grid data stays GPU-resident.

use bevy::prelude::*;
use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferUsages};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bytemuck::cast_slice;
use std::mem::size_of;

use super::buffers::{
    GpuGridLayout, GpuGridNode, GpuMpmParams, GpuParticle, GpuStatisticsScalars,
};
use crate::physics::world::constants::{
    CHUNK_SIZE_I32, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y,
};

const GRID_PADDING_CELLS: i32 = 8;
const MAX_PARTICLES: u32 = 131_072; // 128k particles max

/// Computes the canonical grid layout covering the full world extent.
pub fn world_grid_layout() -> GpuGridLayout {
    let min_cell = IVec2::new(
        WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32 - GRID_PADDING_CELLS,
        WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32 - GRID_PADDING_CELLS,
    );
    let max_cell_exclusive = IVec2::new(
        (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 + GRID_PADDING_CELLS,
        (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 + GRID_PADDING_CELLS,
    );
    let dims = (max_cell_exclusive - min_cell + IVec2::ONE).max(IVec2::ONE);
    GpuGridLayout {
        origin: min_cell,
        dims: dims.as_uvec2(),
    }
}

/// Render-world resource owning GPU buffers for MPM.
#[derive(Resource)]
pub struct MpmGpuBuffers {
    /// Layout of the single-resolution grid.
    pub layout: GpuGridLayout,

    /// Params uniform buffer.
    pub params_buf: Buffer,

    /// Particle buffer.
    pub particle_buf: Buffer,

    /// Grid buffer.
    pub grid_buf: Buffer,

    /// Terrain SDF buffer: node_count * 4 bytes (f32 per node).
    pub terrain_sdf_buf: Buffer,

    /// Terrain normal buffer: node_count * 8 bytes (vec2<f32> per node).
    pub terrain_normal_buf: Buffer,
    /// Temporary per-cell occupancy flags for interaction statistics.
    pub stats_cell_flags_buf: Buffer,

    /// Staging buffer for GPU→CPU readback (particle data, MAP_READ).
    pub readback_buf: Buffer,

    /// GPU-side statistics scalar lanes (atomic<u32>).
    pub stats_scalar_buf: Buffer,
    /// Staging buffer for GPU→CPU readback (statistics scalar lanes, MAP_READ).
    pub stats_scalar_readback_buf: Buffer,

    /// Current active particle count on GPU.
    pub particle_count: u32,

    /// Whether GPU buffers contain valid data for this step.
    pub ready: bool,
}

impl MpmGpuBuffers {
    pub fn new(device: &RenderDevice) -> Self {
        let layout = world_grid_layout();
        let node_count = layout.node_count() as u64;

        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_params"),
            size: size_of::<GpuMpmParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_particles"),
            size: MAX_PARTICLES as u64 * size_of::<GpuParticle>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_grid"),
            size: node_count * size_of::<GpuGridNode>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_sdf_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_terrain_sdf"),
            size: node_count * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_normal_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_terrain_normal"),
            size: node_count * 8,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stats_cell_flags_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_stats_cell_flags"),
            size: node_count * 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Readback buffer: one GpuParticle per max particle slot
        let readback_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_readback"),
            size: MAX_PARTICLES as u64 * size_of::<GpuParticle>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let stats_scalar_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_stats_scalar"),
            size: size_of::<GpuStatisticsScalars>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let stats_scalar_readback_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_stats_scalar_readback"),
            size: size_of::<GpuStatisticsScalars>() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            layout,
            params_buf,
            particle_buf,
            grid_buf,
            terrain_sdf_buf,
            terrain_normal_buf,
            stats_cell_flags_buf,
            readback_buf,
            stats_scalar_buf,
            stats_scalar_readback_buf,
            particle_count: 0,
            ready: false,
        }
    }

    /// Upload simulation parameters to GPU.
    pub fn upload_params(&self, queue: &RenderQueue, params: &GpuMpmParams) {
        queue.write_buffer(
            &self.params_buf,
            0,
            cast_slice(std::slice::from_ref(params)),
        );
    }

    /// Upload particle data to GPU (replaces all particles).
    pub fn upload_particles(&mut self, queue: &RenderQueue, particles: &[GpuParticle]) {
        assert!(
            particles.len() <= MAX_PARTICLES as usize,
            "particle count {} exceeds MAX_PARTICLES {}",
            particles.len(),
            MAX_PARTICLES
        );
        self.particle_count = particles.len() as u32;
        if particles.is_empty() {
            return;
        }
        queue.write_buffer(&self.particle_buf, 0, cast_slice(particles));
    }

    /// Upload terrain SDF and normals to GPU.
    pub fn upload_terrain(&self, queue: &RenderQueue, sdf: &[f32], normals: &[[f32; 2]]) {
        let node_count = self.layout.node_count();
        assert_eq!(sdf.len(), node_count);
        assert_eq!(normals.len(), node_count);
        queue.write_buffer(&self.terrain_sdf_buf, 0, cast_slice(sdf));
        queue.write_buffer(&self.terrain_normal_buf, 0, cast_slice(normals));
    }
}

/// Bevy main-world resource: tracks what CPU needs to upload each frame.
#[derive(Resource, Default)]
pub struct MpmGpuUploadRequest {
    /// If true, upload full particle array this frame.
    pub upload_particles: bool,
    /// If true, upload terrain SDF/normals this frame.
    pub upload_terrain: bool,
    /// Particle data to upload.
    pub particles: Vec<GpuParticle>,
    /// Terrain SDF to upload (indexed by grid layout).
    pub terrain_sdf: Vec<f32>,
    /// Terrain normals to upload.
    pub terrain_normal: Vec<[f32; 2]>,
    /// Last uploaded terrain version to avoid full-grid rebuilds every frame.
    pub last_uploaded_terrain_version: Option<u64>,
}

/// Bevy main-world resource: GPU simulation params for current frame.
#[derive(Resource, Default)]
pub struct MpmGpuParamsRequest {
    pub params: GpuMpmParams,
}

/// Bevy main-world resource: controls whether GPU MPM compute runs this frame.
#[derive(Resource, Default)]
pub struct MpmGpuRunRequest {
    pub enabled: bool,
    /// Number of fixed MPM substeps to execute this render frame.
    pub substeps: u32,
}

/// Fixed-step accumulator for GPU MPM.
///
/// This decouples physics time from render FPS:
/// - if render is fast enough, multiple substeps can run per frame;
/// - if render is too slow, simulation falls behind (slow motion) after hitting max_substeps.
#[derive(Resource, Debug)]
pub struct MpmGpuStepClock {
    pub accumulator_secs: f32,
    pub max_substeps_per_frame: u32,
}

impl Default for MpmGpuStepClock {
    fn default() -> Self {
        Self {
            accumulator_secs: 0.0,
            max_substeps_per_frame: 8,
        }
    }
}

/// Runtime control for staged bring-up of GPU MPM.
///
/// `init_only=true` keeps GPU resources allocated but disables
/// upload/compute/readback work.
#[derive(Resource, Clone, Copy, Debug)]
pub struct MpmGpuControl {
    pub init_only: bool,
    pub readback_enabled: bool,
    /// Readback cadence in render frames (60 ~= once per second at 60 FPS).
    pub readback_interval_frames: u32,
}

impl Default for MpmGpuControl {
    fn default() -> Self {
        Self {
            init_only: false,
            readback_enabled: true,
            readback_interval_frames: 60,
        }
    }
}
