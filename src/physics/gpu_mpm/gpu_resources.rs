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
    GpuChunkMeta, GpuGridLayout, GpuGridNode, GpuMoverResult, GpuMpmParams, GpuParticle,
    GpuStatisticsScalars,
};
use crate::physics::world::constants::{
    CHUNK_SIZE_I32, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y,
};

const GRID_PADDING_CELLS: i32 = 8;
pub const MAX_PARTICLES: u32 = 131_072; // 128k particles max
pub const MAX_MOVER_RECORDS: u32 = MAX_PARTICLES;
pub const MPM_CHUNK_NODE_DIM: u32 = CHUNK_SIZE_I32 as u32;
pub const MPM_CHUNK_NODES_PER_SLOT: u32 = MPM_CHUNK_NODE_DIM * MPM_CHUNK_NODE_DIM;
pub const MAX_RESIDENT_CHUNK_SLOTS: u32 = 256;
pub const MAX_RESIDENT_NODE_CAPACITY: u64 =
    MAX_RESIDENT_CHUNK_SLOTS as u64 * MPM_CHUNK_NODES_PER_SLOT as u64;

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

    /// Chunk metadata buffer.
    pub chunk_meta_buf: Buffer,

    /// Terrain SDF buffer: node_count * 4 bytes (f32 per node).
    pub terrain_sdf_buf: Buffer,

    /// Terrain normal buffer: node_count * 8 bytes (vec2<f32> per node).
    pub terrain_normal_buf: Buffer,
    /// Temporary per-cell occupancy flags for interaction statistics.
    pub stats_cell_flags_buf: Buffer,

    /// Staging buffer for GPU→CPU readback (particle data, MAP_READ).
    pub readback_buf: Buffer,
    /// Atomic mover count generated on GPU each step.
    pub mover_count_buf: Buffer,
    /// GPU movers append buffer.
    pub mover_buf: Buffer,
    /// CPU-uploaded mover result count for GPU apply pass.
    pub mover_result_count_buf: Buffer,
    /// CPU-uploaded mover result records.
    pub mover_result_buf: Buffer,
    /// Staging buffer for GPU→CPU mover readback (count + records, MAP_READ).
    pub mover_readback_buf: Buffer,

    /// GPU-side statistics scalar lanes (atomic<u32>).
    pub stats_scalar_buf: Buffer,
    /// Staging buffer for GPU→CPU readback (statistics scalar lanes, MAP_READ).
    pub stats_scalar_readback_buf: Buffer,

    /// Current active particle count on GPU.
    pub particle_count: u32,
    /// Current active resident chunk count on GPU.
    pub active_chunk_count: u32,
    /// Current mover-result count to apply on GPU this frame.
    pub mover_result_count: u32,
    /// Max grid-node capacity allocated for sparse chunk layout.
    pub max_node_capacity: usize,

    /// Whether GPU buffers contain valid data for this step.
    pub ready: bool,
}

impl MpmGpuBuffers {
    pub fn new(device: &RenderDevice) -> Self {
        let layout = world_grid_layout();
        let max_node_capacity = MAX_RESIDENT_NODE_CAPACITY as usize;

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
            size: MAX_RESIDENT_NODE_CAPACITY * size_of::<GpuGridNode>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let chunk_meta_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_chunk_meta"),
            size: MAX_RESIDENT_CHUNK_SLOTS as u64 * size_of::<GpuChunkMeta>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_sdf_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_terrain_sdf"),
            size: MAX_RESIDENT_NODE_CAPACITY * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_normal_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_terrain_normal"),
            size: MAX_RESIDENT_NODE_CAPACITY * 8,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stats_cell_flags_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_stats_cell_flags"),
            size: MAX_RESIDENT_NODE_CAPACITY * 4,
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
        let mover_count_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_mover_count"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mover_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_movers"),
            size: MAX_MOVER_RECORDS as u64 * size_of::<super::buffers::GpuMoverRecord>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mover_result_count_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_mover_result_count"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mover_result_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_mover_results"),
            size: MAX_MOVER_RECORDS as u64 * size_of::<GpuMoverResult>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mover_readback_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_mover_readback"),
            size: size_of::<u32>() as u64
                + MAX_MOVER_RECORDS as u64 * size_of::<super::buffers::GpuMoverRecord>() as u64,
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
            chunk_meta_buf,
            terrain_sdf_buf,
            terrain_normal_buf,
            stats_cell_flags_buf,
            readback_buf,
            mover_count_buf,
            mover_buf,
            mover_result_count_buf,
            mover_result_buf,
            mover_readback_buf,
            stats_scalar_buf,
            stats_scalar_readback_buf,
            particle_count: 0,
            active_chunk_count: 0,
            mover_result_count: 0,
            max_node_capacity,
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
        assert_eq!(sdf.len(), normals.len());
        assert!(sdf.len() <= self.max_node_capacity);
        queue.write_buffer(&self.terrain_sdf_buf, 0, cast_slice(sdf));
        queue.write_buffer(&self.terrain_normal_buf, 0, cast_slice(normals));
    }

    pub fn upload_chunks(&mut self, queue: &RenderQueue, chunks: &[GpuChunkMeta]) {
        assert!(chunks.len() <= MAX_RESIDENT_CHUNK_SLOTS as usize);
        self.active_chunk_count = chunks.len() as u32;
        if chunks.is_empty() {
            return;
        }
        queue.write_buffer(&self.chunk_meta_buf, 0, cast_slice(chunks));
    }

    pub fn upload_chunk_diffs(&self, queue: &RenderQueue, diffs: &[(u32, GpuChunkMeta)]) {
        let chunk_meta_stride = size_of::<GpuChunkMeta>() as u64;
        for &(slot_id, meta) in diffs {
            assert!(slot_id < MAX_RESIDENT_CHUNK_SLOTS);
            let offset = slot_id as u64 * chunk_meta_stride;
            queue.write_buffer(
                &self.chunk_meta_buf,
                offset,
                cast_slice(std::slice::from_ref(&meta)),
            );
        }
    }

    pub fn upload_mover_results(&mut self, queue: &RenderQueue, results: &[GpuMoverResult]) {
        assert!(results.len() <= MAX_MOVER_RECORDS as usize);
        self.mover_result_count = results.len() as u32;
        queue.write_buffer(
            &self.mover_result_count_buf,
            0,
            cast_slice(std::slice::from_ref(&self.mover_result_count)),
        );
        if results.is_empty() {
            return;
        }
        queue.write_buffer(&self.mover_result_buf, 0, cast_slice(results));
    }
}

/// Bevy main-world resource: tracks what CPU needs to upload each frame.
#[derive(Resource, Default)]
pub struct MpmGpuUploadRequest {
    /// Main-world request bit: request full particle array upload on next extraction.
    pub upload_particles: bool,
    /// One-frame latched bit extracted into render world.
    pub upload_particles_frame: bool,
    /// Main-world request bit: request mover-result upload on next extraction.
    pub upload_mover_results: bool,
    /// One-frame latched bit extracted into render world.
    pub upload_mover_results_frame: bool,
    /// If true, upload chunk metadata this frame.
    pub upload_chunks: bool,
    /// If true, upload partial chunk metadata updates this frame.
    pub upload_chunk_diffs: bool,
    /// If true, upload terrain SDF/normals this frame.
    pub upload_terrain: bool,
    /// Chunk metadata (slot-indexed contiguous resident set).
    pub chunk_meta: Vec<GpuChunkMeta>,
    /// Particle data to upload.
    pub particles: Vec<GpuParticle>,
    /// CPU→GPU mover results (`particle_id -> new_home_slot_id`).
    pub mover_results: Vec<GpuMoverResult>,
    /// CPU→GPU chunk metadata diffs (`slot_id`, `meta`).
    pub chunk_meta_diffs: Vec<(u32, GpuChunkMeta)>,
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
