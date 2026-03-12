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
    ACTIVE_TILE_NODE_DIM, GpuActiveTileRecord, GpuChunkEventRecord, GpuChunkMeta, GpuGridLayout,
    GpuGridNode, GpuMoverResult, GpuMpmParams, GpuParticle, GpuStatisticsScalars,
    GpuWorldEditAddOp, GpuWorldEditAddParams, GpuWorldEditRemoveParams,
};
use crate::physics::world::constants::{
    CELL_SIZE_M, CHUNK_SIZE_I32, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X,
    WORLD_MIN_CHUNK_Y,
};

const GRID_PADDING_CELLS: i32 = 8;
pub const MPM_NODES_PER_CELL: u32 = 2;
pub const MPM_NODE_SPACING_M: f32 = CELL_SIZE_M / MPM_NODES_PER_CELL as f32;
const GRID_PADDING_NODES: i32 = GRID_PADDING_CELLS * MPM_NODES_PER_CELL as i32;
pub const MAX_PARTICLES: u32 = 131_072; // 128k particles max
pub const MAX_MOVER_RECORDS: u32 = MAX_PARTICLES;
pub const MAX_CHUNK_EVENT_RECORDS: u32 = MAX_RESIDENT_CHUNK_SLOTS * 4;
pub const MPM_CHUNK_NODE_DIM: u32 = CHUNK_SIZE_I32 as u32 * MPM_NODES_PER_CELL;
pub const MPM_CHUNK_NODES_PER_SLOT: u32 = MPM_CHUNK_NODE_DIM * MPM_CHUNK_NODE_DIM;
pub const MPM_ACTIVE_TILE_DIM_PER_AXIS: u32 =
    (MPM_CHUNK_NODE_DIM + ACTIVE_TILE_NODE_DIM - 1) / ACTIVE_TILE_NODE_DIM;
pub const MPM_ACTIVE_TILES_PER_SLOT: u32 =
    MPM_ACTIVE_TILE_DIM_PER_AXIS * MPM_ACTIVE_TILE_DIM_PER_AXIS;
pub const MAX_RESIDENT_CHUNK_SLOTS: u32 = 256;
pub const MAX_RESIDENT_NODE_CAPACITY: u64 =
    MAX_RESIDENT_CHUNK_SLOTS as u64 * MPM_CHUNK_NODES_PER_SLOT as u64;
pub const MAX_ACTIVE_TILE_RECORDS: u64 =
    MAX_RESIDENT_CHUNK_SLOTS as u64 * MPM_ACTIVE_TILES_PER_SLOT as u64;
const _: () = assert!(MPM_ACTIVE_TILES_PER_SLOT <= 32);

/// Computes the canonical grid layout covering the full world extent.
pub fn world_grid_layout() -> GpuGridLayout {
    let chunk_node_dim_i = MPM_CHUNK_NODE_DIM as i32;
    let min_cell = IVec2::new(
        WORLD_MIN_CHUNK_X * chunk_node_dim_i - GRID_PADDING_NODES,
        WORLD_MIN_CHUNK_Y * chunk_node_dim_i - GRID_PADDING_NODES,
    );
    let max_cell_exclusive = IVec2::new(
        (WORLD_MAX_CHUNK_X + 1) * chunk_node_dim_i + GRID_PADDING_NODES,
        (WORLD_MAX_CHUNK_Y + 1) * chunk_node_dim_i + GRID_PADDING_NODES,
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
    /// Scratch particle buffer used for GPU remove compaction.
    pub particle_scratch_buf: Buffer,

    /// Grid buffer.
    pub grid_buf: Buffer,

    /// Chunk metadata buffer.
    pub chunk_meta_buf: Buffer,
    /// Atomic active-tile count for sparse clear/grid-update dispatch.
    pub active_tile_count_buf: Buffer,
    /// Compacted active-tile records for sparse clear/grid-update dispatch.
    pub active_tile_list_buf: Buffer,
    /// Indirect dispatch arguments derived from the active-tile list.
    pub active_tile_dispatch_buf: Buffer,

    /// Terrain SDF buffer: node_count * 4 bytes (f32 per node).
    pub terrain_sdf_buf: Buffer,

    /// Terrain normal buffer: node_count * 8 bytes (vec2<f32> per node).
    pub terrain_normal_buf: Buffer,
    /// Terrain solid-node mask buffer: one u32(0/1) per resident node.
    pub terrain_node_solid_buf: Buffer,
    /// Terrain solid-cell occupancy buffer: one u32(0/1) per slot-local cell/node.
    pub terrain_cell_solid_buf: Buffer,
    /// Slot-update count for terrain SDF recompute pass.
    pub terrain_update_slot_count_buf: Buffer,
    /// Slot ids to recompute terrain SDF/normal.
    pub terrain_update_slot_buf: Buffer,
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
    /// Uniform params for GPU incremental add edits.
    pub world_edit_add_params_buf: Buffer,
    /// CPU-uploaded incremental add edit records.
    pub world_edit_add_op_buf: Buffer,
    /// Uniform params for GPU incremental remove edits.
    pub world_edit_remove_params_buf: Buffer,
    /// CPU-uploaded sorted particle ids to remove.
    pub world_edit_remove_id_buf: Buffer,
    /// Atomic keep-count scratch for GPU remove compaction.
    pub world_edit_remove_keep_count_buf: Buffer,
    /// Staging buffer for GPU→CPU mover readback (count + records, MAP_READ).
    pub mover_readback_buf: Buffer,
    /// Atomic chunk-event count generated on GPU each step.
    pub chunk_event_count_buf: Buffer,
    /// GPU chunk-event append buffer.
    pub chunk_event_buf: Buffer,
    /// Staging buffer for GPU→CPU chunk-event readback (count + records, MAP_READ).
    pub chunk_event_readback_buf: Buffer,

    /// GPU-side statistics scalar lanes (atomic<u32>).
    pub stats_scalar_buf: Buffer,
    /// Staging buffer for GPU→CPU readback (statistics scalar lanes, MAP_READ).
    pub stats_scalar_readback_buf: Buffer,

    /// Current active particle count on GPU.
    pub particle_count: u32,
    /// Revision of the particle set currently resident on GPU.
    pub particle_revision: u64,
    /// Current active resident chunk count on GPU.
    pub active_chunk_count: u32,
    /// Current mover-result count to apply on GPU this frame.
    pub mover_result_count: u32,
    /// Current incremental world-edit add op count to apply on GPU this frame.
    pub world_edit_add_count: u32,
    /// Current incremental world-edit remove particle-id count to apply on GPU this frame.
    pub world_edit_remove_count: u32,
    /// Source particle count before the current GPU remove compaction.
    pub world_edit_remove_source_count: u32,
    /// Current terrain slot-update count for GPU terrain SDF recompute.
    pub terrain_update_slot_count: u32,
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
        let particle_scratch_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_particles_scratch"),
            size: MAX_PARTICLES as u64 * size_of::<GpuParticle>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
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
        let active_tile_count_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_active_tile_count"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let active_tile_list_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_active_tile_list"),
            size: MAX_ACTIVE_TILE_RECORDS * size_of::<GpuActiveTileRecord>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let active_tile_dispatch_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_active_tile_dispatch"),
            size: (size_of::<u32>() * 3) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
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
        let terrain_node_solid_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_terrain_node_solid"),
            size: MAX_RESIDENT_NODE_CAPACITY * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let terrain_cell_solid_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_terrain_cell_solid"),
            size: MAX_RESIDENT_NODE_CAPACITY * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let terrain_update_slot_count_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_terrain_update_slot_count"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let terrain_update_slot_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_terrain_update_slot_ids"),
            size: MAX_RESIDENT_CHUNK_SLOTS as u64 * size_of::<u32>() as u64,
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
        let world_edit_add_params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_world_edit_add_params"),
            size: size_of::<GpuWorldEditAddParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let world_edit_add_op_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_world_edit_add_ops"),
            size: MAX_PARTICLES as u64 * size_of::<GpuWorldEditAddOp>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let world_edit_remove_params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_world_edit_remove_params"),
            size: size_of::<GpuWorldEditRemoveParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let world_edit_remove_id_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_world_edit_remove_ids"),
            size: MAX_PARTICLES as u64 * size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let world_edit_remove_keep_count_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_world_edit_remove_keep_count"),
            size: size_of::<u32>() as u64,
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
        let chunk_event_count_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_chunk_event_count"),
            size: size_of::<u32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_event_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_chunk_events"),
            size: MAX_CHUNK_EVENT_RECORDS as u64 * size_of::<GpuChunkEventRecord>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let chunk_event_readback_buf = device.create_buffer(&BufferDescriptor {
            label: Some("mpm_chunk_event_readback"),
            size: size_of::<u32>() as u64
                + MAX_CHUNK_EVENT_RECORDS as u64 * size_of::<GpuChunkEventRecord>() as u64,
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
            particle_scratch_buf,
            grid_buf,
            chunk_meta_buf,
            active_tile_count_buf,
            active_tile_list_buf,
            active_tile_dispatch_buf,
            terrain_sdf_buf,
            terrain_normal_buf,
            terrain_node_solid_buf,
            terrain_cell_solid_buf,
            terrain_update_slot_count_buf,
            terrain_update_slot_buf,
            stats_cell_flags_buf,
            readback_buf,
            mover_count_buf,
            mover_buf,
            mover_result_count_buf,
            mover_result_buf,
            world_edit_add_params_buf,
            world_edit_add_op_buf,
            world_edit_remove_params_buf,
            world_edit_remove_id_buf,
            world_edit_remove_keep_count_buf,
            mover_readback_buf,
            chunk_event_count_buf,
            chunk_event_buf,
            chunk_event_readback_buf,
            stats_scalar_buf,
            stats_scalar_readback_buf,
            particle_count: 0,
            particle_revision: 0,
            active_chunk_count: 0,
            mover_result_count: 0,
            world_edit_add_count: 0,
            world_edit_remove_count: 0,
            world_edit_remove_source_count: 0,
            terrain_update_slot_count: 0,
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

    /// Upload terrain solid-cell diffs for selected chunk slots.
    pub fn upload_terrain_cell_slot_diffs(
        &self,
        queue: &RenderQueue,
        slot_ids: &[u32],
        cell_solid: &[u32],
    ) {
        let nodes_per_slot = MPM_CHUNK_NODES_PER_SLOT as usize;
        assert_eq!(cell_solid.len(), slot_ids.len() * nodes_per_slot);
        let slot_bytes = (MPM_CHUNK_NODES_PER_SLOT as u64) * size_of::<u32>() as u64;
        for (i, &slot_id) in slot_ids.iter().enumerate() {
            assert!(slot_id < MAX_RESIDENT_CHUNK_SLOTS);
            let base = i * nodes_per_slot;
            let end = base + nodes_per_slot;
            let offset = slot_id as u64 * slot_bytes;
            queue.write_buffer(
                &self.terrain_cell_solid_buf,
                offset,
                cast_slice(&cell_solid[base..end]),
            );
        }
    }

    /// Upload slot update list for GPU terrain SDF recompute pass.
    pub fn upload_terrain_update_slots(&mut self, queue: &RenderQueue, slot_ids: &[u32]) {
        assert!(slot_ids.len() <= MAX_RESIDENT_CHUNK_SLOTS as usize);
        self.terrain_update_slot_count = slot_ids.len() as u32;
        queue.write_buffer(
            &self.terrain_update_slot_count_buf,
            0,
            cast_slice(std::slice::from_ref(&self.terrain_update_slot_count)),
        );
        if slot_ids.is_empty() {
            return;
        }
        queue.write_buffer(&self.terrain_update_slot_buf, 0, cast_slice(slot_ids));
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

    pub fn upload_world_edit_adds(
        &mut self,
        queue: &RenderQueue,
        base_particle_count: u32,
        ops: &[GpuWorldEditAddOp],
    ) {
        assert!(ops.len() <= MAX_PARTICLES as usize);
        self.world_edit_add_count = ops.len() as u32;
        let params = GpuWorldEditAddParams {
            base_particle_count,
            op_count: self.world_edit_add_count,
            _pad0: [0; 2],
        };
        queue.write_buffer(
            &self.world_edit_add_params_buf,
            0,
            cast_slice(std::slice::from_ref(&params)),
        );
        if ops.is_empty() {
            return;
        }
        queue.write_buffer(&self.world_edit_add_op_buf, 0, cast_slice(ops));
    }

    pub fn upload_world_edit_removes(
        &mut self,
        queue: &RenderQueue,
        particle_count: u32,
        remove_particle_ids: &[u32],
    ) {
        assert!(remove_particle_ids.len() <= MAX_PARTICLES as usize);
        self.world_edit_remove_count = remove_particle_ids.len() as u32;
        self.world_edit_remove_source_count = particle_count;
        let params = GpuWorldEditRemoveParams {
            particle_count,
            remove_count: self.world_edit_remove_count,
            _pad0: [0; 2],
        };
        queue.write_buffer(
            &self.world_edit_remove_params_buf,
            0,
            cast_slice(std::slice::from_ref(&params)),
        );
        queue.write_buffer(
            &self.world_edit_remove_keep_count_buf,
            0,
            cast_slice(std::slice::from_ref(&0u32)),
        );
        if remove_particle_ids.is_empty() {
            return;
        }
        queue.write_buffer(
            &self.world_edit_remove_id_buf,
            0,
            cast_slice(remove_particle_ids),
        );
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
    /// One-frame latched bit extracted into render world.
    pub upload_chunks_frame: bool,
    /// If true, upload partial chunk metadata updates this frame.
    pub upload_chunk_diffs: bool,
    /// One-frame latched bit extracted into render world.
    pub upload_chunk_diffs_frame: bool,
    /// If true, upload terrain SDF/normals this frame.
    pub upload_terrain: bool,
    /// One-frame latched bit extracted into render world.
    pub upload_terrain_frame: bool,
    /// If true, upload terrain solid-cell diffs for selected slots this frame.
    pub upload_terrain_cell_slot_diffs: bool,
    /// One-frame latched bit extracted into render world.
    pub upload_terrain_cell_slot_diffs_frame: bool,
    /// Chunk metadata (slot-indexed contiguous resident set).
    pub chunk_meta: Vec<GpuChunkMeta>,
    /// Particle data to upload.
    pub particles: Vec<GpuParticle>,
    /// Monotonic revision for `particles`.
    pub particle_revision: u64,
    /// CPU→GPU mover results (`particle_id -> new_home_slot_id`).
    pub mover_results: Vec<GpuMoverResult>,
    /// CPU→GPU chunk metadata diffs (`slot_id`, `meta`).
    pub chunk_meta_diffs: Vec<(u32, GpuChunkMeta)>,
    /// Terrain SDF to upload (indexed by grid layout).
    pub terrain_sdf: Vec<f32>,
    /// Terrain normals to upload.
    pub terrain_normal: Vec<[f32; 2]>,
    /// Slot ids for terrain diff upload.
    pub terrain_slot_ids: Vec<u32>,
    /// Terrain solid-cell diffs for slot updates (`terrain_slot_ids.len() * NODES_PER_SLOT`).
    pub terrain_cell_solid_slot_diffs: Vec<u32>,
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
            readback_interval_frames: 1,
        }
    }
}
