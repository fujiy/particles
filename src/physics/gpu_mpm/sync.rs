// CPU → GPU synchronization systems for MLS-MPM.
//
// Runs in main-world (not render-world): prepares upload requests,
// which are then executed in the render-world extract/prepare phase.

use bevy::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

use super::buffers::{
    self, CHUNK_EVENT_KIND_FRONTIER_REQUEST, CHUNK_EVENT_KIND_NEWLY_EMPTY,
    CHUNK_EVENT_KIND_NEWLY_OCCUPIED, CHUNK_EVENT_KIND_SLOT_SNAPSHOT, GpuChunkMeta, GpuGridLayout,
    GpuMoverResult, GpuMpmParams, GpuParticle, GpuStatisticsScalars, GpuWorldEditAddOp,
    INVALID_CHUNK_SLOT_ID,
};
use super::gpu_resources::{
    MAX_RESIDENT_CHUNK_SLOTS, MPM_ACTIVE_TILES_PER_SLOT, MPM_CHUNK_NODE_DIM,
    MPM_CHUNK_NODES_PER_SLOT, MPM_NODE_SPACING_M, MPM_NODES_PER_CELL, MpmGpuControl,
    MpmGpuParamsRequest, MpmGpuRunRequest, MpmGpuStepClock, MpmGpuUploadRequest, world_grid_layout,
};
use super::phase::mpm_phase_id_for_particle;
use super::readback::{
    GpuChunkEventReadbackResult, GpuMoverApplyAck, GpuMoverReadbackResult, GpuParticleReadbackPayload,
    GpuReadbackResult, GpuStatisticsReadbackResult,
};
use crate::params::ActivePhysicsParams;
use crate::physics::material::{ParticleMaterial, particle_properties};
use crate::physics::state::{ReplayState, SimulationState};
use crate::physics::world::constants::{CELL_SIZE_M, CHUNK_SIZE_I32};
use crate::physics::world::terrain::{
    TerrainCell, TerrainWorld, cell_to_world_center, world_to_cell,
};
use crate::render::TerrainGeneratedChunkCache;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const CHUNK_HALO_RADIUS: i32 = 1;
const ACTIVE_TILE_NODE_DIM_I32: i32 = 8;
const MOORE_OFFSETS: [IVec2; 8] = [
    IVec2::new(-1, -1),
    IVec2::new(0, -1),
    IVec2::new(1, -1),
    IVec2::new(-1, 0),
    IVec2::new(1, 0),
    IVec2::new(-1, 1),
    IVec2::new(0, 1),
    IVec2::new(1, 1),
];

#[derive(Clone, Copy, Debug, Default)]
struct ChunkSlotState {
    occupied_particle_count: u32,
    halo_ref_count: u32,
    resident: bool,
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum MpmSyncSet {
    ApplyReadback,
    PrepareUpload,
}

#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct MpmParticleReadbackStatus {
    pub particle_count: u32,
    pub particle_revision: u64,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct MpmFullParticleReadbackRequest {
    pub requested: bool,
    pub min_particle_revision: u64,
}

impl MpmFullParticleReadbackRequest {
    pub fn request(&mut self, min_particle_revision: u64) {
        self.requested = true;
        self.min_particle_revision = self.min_particle_revision.max(min_particle_revision);
    }

    fn clear_if_satisfied(&mut self, particle_revision: u64) {
        if self.requested && particle_revision >= self.min_particle_revision {
            self.requested = false;
            self.min_particle_revision = 0;
        }
    }
}

#[derive(Resource, Debug, Default, Clone)]
pub struct MpmFullParticleReadbackCache {
    pub payload: Option<GpuParticleReadbackPayload>,
}

impl MpmFullParticleReadbackCache {
    fn payload_for_revision(&self, min_particle_revision: u64) -> Option<&GpuParticleReadbackPayload> {
        self.payload
            .as_ref()
            .filter(|payload| payload.particle_revision >= min_particle_revision)
    }

    fn invalidate_before(&mut self, particle_revision: u64) {
        if self
            .payload
            .as_ref()
            .map(|payload| payload.particle_revision < particle_revision)
            .unwrap_or(false)
        {
            self.payload = None;
        }
    }
}

#[derive(Resource, Debug, Default)]
pub struct DeferredGpuWorldEditRequests {
    pub requests: Vec<GpuWorldEditRequest>,
}

#[derive(Resource, Clone, Default)]
pub struct GpuWorldEditAddApplyAck {
    inner: Arc<AtomicU64>,
}

impl GpuWorldEditAddApplyAck {
    pub fn signal_revision(&self, particle_revision: u64) {
        let mut current = self.inner.load(Ordering::Acquire);
        while particle_revision > current {
            match self.inner.compare_exchange(
                current,
                particle_revision,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(next) => current = next,
            }
        }
    }

    pub fn value(&self) -> u64 {
        self.inner.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PendingGpuParticleAddOp {
    pub cell: IVec2,
    pub material: ParticleMaterial,
    pub count_per_cell: u32,
}

#[derive(Clone, Debug)]
pub struct PendingGpuParticleAddBatch {
    pub particle_revision: u64,
    pub added_particle_count: u32,
    pub expected_total_particle_count: u32,
    pub ops: Vec<PendingGpuParticleAddOp>,
}

#[derive(Resource, Debug, Default)]
pub struct PendingGpuParticleAdds {
    batches: VecDeque<PendingGpuParticleAddBatch>,
}

impl PendingGpuParticleAdds {
    pub fn clear(&mut self) {
        self.batches.clear();
    }

    pub fn latest_revision(&self) -> u64 {
        self.batches
            .back()
            .map(|batch| batch.particle_revision)
            .unwrap_or(0)
    }

    pub fn total_added_particles(&self) -> u32 {
        self.batches
            .iter()
            .fold(0u32, |sum, batch| sum.saturating_add(batch.added_particle_count))
    }

    fn push_batch(&mut self, batch: PendingGpuParticleAddBatch) {
        self.batches.push_back(batch);
    }

    fn drop_acked_batches(&mut self, particle_revision: u64) {
        while self
            .batches
            .front()
            .map(|batch| batch.particle_revision <= particle_revision)
            .unwrap_or(false)
        {
            self.batches.pop_front();
        }
    }

    fn oldest_batch(&self) -> Option<&PendingGpuParticleAddBatch> {
        self.batches.front()
    }
}

#[derive(Resource, Debug, Default, Clone)]
pub struct MpmGpuWorldEditAddQueueRequest {
    pub particle_revision: u64,
    pub base_particle_count: u32,
    pub added_particle_count: u32,
    pub ops: Vec<GpuWorldEditAddOp>,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct MpmGpuWorldEditRemoveQueueRequest {
    pub particle_revision: u64,
    pub removed_particle_count: u32,
    pub remove_particle_ids: Vec<u32>,
}

#[derive(Resource, Debug, Clone)]
pub struct MpmChunkResidencyState {
    pub initialized: bool,
    pub chunk_origin: IVec2,
    pub chunk_dims: UVec2,
    pub grid_layout: GpuGridLayout,
    /// Slot-capacity count used by GPU kernels (`chunk_dims.x * chunk_dims.y`).
    pub resident_chunk_count: u32,
    /// CPU-tracked number of currently resident slots.
    pub active_resident_chunk_count: u32,
    pub chunk_sdf_samples: u32,
    pub active_tile_count: u32,
    pub active_tile_capacity: u32,
    pub inactive_skip_rate: f32,
    pub invalid_slot_access_count: u64,
    pub runtime_rebuild_count: u64,
    pub runtime_rebuild_reason_empty_state: u64,
    pub runtime_rebuild_reason_invalid_old_slot: u64,
    pub runtime_rebuild_reason_new_chunk_oob: u64,
    pub runtime_rebuild_reason_old_slot_empty: u64,
    pub runtime_rebuild_reason_halo_update_fail: u64,
    pub runtime_rebuild_waiting_readback_count: u64,
    pub pending_mover_apply: bool,
    pub pending_mover_readback: bool,
    pub observed_mover_apply_ack: u64,
    pub mover_readback_flush_budget: u32,
    rebuild_requested: bool,
    chunk_to_slot: HashMap<IVec2, u32>,
    slot_to_chunk: Vec<IVec2>,
    slot_allocated: Vec<bool>,
    free_slots: Vec<u32>,
    slot_state: Vec<ChunkSlotState>,
    chunk_meta_cache: Vec<GpuChunkMeta>,
}

impl Default for MpmChunkResidencyState {
    fn default() -> Self {
        Self {
            initialized: false,
            chunk_origin: IVec2::ZERO,
            chunk_dims: UVec2::ZERO,
            grid_layout: world_grid_layout(),
            resident_chunk_count: 0,
            active_resident_chunk_count: 0,
            chunk_sdf_samples: 0,
            active_tile_count: 0,
            active_tile_capacity: 0,
            inactive_skip_rate: 0.0,
            invalid_slot_access_count: 0,
            runtime_rebuild_count: 0,
            runtime_rebuild_reason_empty_state: 0,
            runtime_rebuild_reason_invalid_old_slot: 0,
            runtime_rebuild_reason_new_chunk_oob: 0,
            runtime_rebuild_reason_old_slot_empty: 0,
            runtime_rebuild_reason_halo_update_fail: 0,
            runtime_rebuild_waiting_readback_count: 0,
            pending_mover_apply: false,
            pending_mover_readback: false,
            observed_mover_apply_ack: 0,
            mover_readback_flush_budget: 0,
            rebuild_requested: false,
            chunk_to_slot: HashMap::default(),
            slot_to_chunk: Vec::new(),
            slot_allocated: Vec::new(),
            free_slots: Vec::new(),
            slot_state: Vec::new(),
            chunk_meta_cache: Vec::new(),
        }
    }
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct MpmStatisticsStatus {
    pub total_particles: bool,
    pub phase_counts: bool,
    pub max_speed: bool,
    pub penetration: bool,
    pub tracked_summary: bool,
    pub water_surface_p95: bool,
    pub granular_repose: bool,
    pub material_interaction: bool,
    pub grid_density: bool,
    pub tracked_phase_id: u32,
    pub tracked_fallback_to_all: bool,
    pub repose_phase_id: u32,
    pub interaction_primary_phase_id: u32,
    pub interaction_secondary_phase_id: u32,
}

impl MpmStatisticsStatus {
    pub fn any_enabled(self) -> bool {
        self.total_particles
            || self.phase_counts
            || self.max_speed
            || self.penetration
            || self.tracked_summary
            || self.water_surface_p95
            || self.granular_repose
            || self.material_interaction
            || self.grid_density
    }
}

impl Default for MpmStatisticsStatus {
    fn default() -> Self {
        Self {
            total_particles: true,
            phase_counts: true,
            max_speed: false,
            penetration: false,
            tracked_summary: false,
            water_surface_p95: false,
            granular_repose: false,
            material_interaction: false,
            grid_density: false,
            tracked_phase_id: 0,
            tracked_fallback_to_all: true,
            repose_phase_id: 1,
            interaction_primary_phase_id: 2,
            interaction_secondary_phase_id: 0,
        }
    }
}

#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct MpmStatisticsSnapshot {
    pub total_particles: u32,
    pub water_liquid: u32,
    pub soil_granular: u32,
    pub sand_granular: u32,
    pub unknown: u32,
    pub max_speed_mps: f32,
    pub all_count: u32,
    pub all_penetration_count: u32,
    pub all_mean_y_m: f32,
    pub tracked_count: u32,
    pub tracked_penetration_count: u32,
    pub tracked_mean_y_m: f32,
    pub tracked_min_x_m: Option<f32>,
    pub tracked_max_x_m: Option<f32>,
    pub tracked_min_y_m: Option<f32>,
    pub tracked_max_y_m: Option<f32>,
    pub water_surface_p95_cell: Option<i32>,
    pub granular_repose_angle_deg: Option<f32>,
    pub granular_repose_base_span_cells: Option<i32>,
    pub material_interaction_contact_ratio: f32,
    pub material_interaction_primary_centroid_y_m: Option<f32>,
    pub material_interaction_secondary_centroid_y_m: Option<f32>,
    pub grid_water_phi_max: f32,
    pub grid_water_phi_p99: f32,
    pub grid_water_phi_mean_nonzero: f32,
    pub grid_water_nonzero_nodes: u32,
    pub grid_granular_phi_max: f32,
    pub grid_granular_phi_p99: f32,
    pub grid_granular_phi_mean_nonzero: f32,
    pub grid_granular_nonzero_nodes: u32,
}

impl MpmStatisticsSnapshot {
    pub fn total(self) -> u32 {
        self.total_particles
    }

    pub fn tracked_penetration_ratio(self) -> f32 {
        if self.tracked_count == 0 {
            0.0
        } else {
            self.tracked_penetration_count as f32 / self.tracked_count as f32
        }
    }

    pub fn all_penetration_ratio(self) -> f32 {
        if self.all_count == 0 {
            0.0
        } else {
            self.all_penetration_count as f32 / self.all_count as f32
        }
    }

    fn from_gpu_scalars(scalars: GpuStatisticsScalars) -> Self {
        fn ordered_u32_to_f32(ordered: u32) -> f32 {
            let bits = if (ordered & 0x8000_0000) != 0 {
                ordered ^ 0x8000_0000
            } else {
                !ordered
            };
            f32::from_bits(bits)
        }

        fn decode_optional_ordered_f32(raw: u32) -> Option<f32> {
            let value = ordered_u32_to_f32(raw);
            value.is_finite().then_some(value)
        }

        fn lane_as_i32(raw: u32) -> i32 {
            i32::from_ne_bytes(raw.to_ne_bytes())
        }

        let all_count = scalars.lanes[buffers::GPU_STATS_LANE_ALL_COUNT];
        let tracked_count = scalars.lanes[buffers::GPU_STATS_LANE_TRACKED_COUNT];
        let all_sum_y_fp = lane_as_i32(scalars.lanes[buffers::GPU_STATS_LANE_ALL_SUM_Y_FP]);
        let tracked_sum_y_fp = lane_as_i32(scalars.lanes[buffers::GPU_STATS_LANE_TRACKED_SUM_Y_FP]);
        let fp_scale = 100.0_f32;
        let all_mean_y_m = if all_count > 0 {
            all_sum_y_fp as f32 / (fp_scale * all_count as f32)
        } else {
            0.0
        };
        let tracked_mean_y_m = if tracked_count > 0 {
            tracked_sum_y_fp as f32 / (fp_scale * tracked_count as f32)
        } else {
            0.0
        };
        let water_surface_p95_cell_bits =
            scalars.lanes[buffers::GPU_STATS_LANE_WATER_SURFACE_P95_CELL_BITS];
        let water_surface_p95_cell = if water_surface_p95_cell_bits == 0x8000_0000 {
            None
        } else {
            Some(lane_as_i32(water_surface_p95_cell_bits))
        };

        let repose_angle =
            f32::from_bits(scalars.lanes[buffers::GPU_STATS_LANE_GRANULAR_REPOSE_ANGLE_BITS]);
        let granular_repose_angle_deg = repose_angle.is_finite().then_some(repose_angle);
        let repose_base_bits =
            scalars.lanes[buffers::GPU_STATS_LANE_GRANULAR_REPOSE_BASE_SPAN_BITS];
        let granular_repose_base_span_cells = if repose_base_bits == 0x8000_0000 {
            None
        } else {
            Some(lane_as_i32(repose_base_bits))
        };

        let primary_centroid = f32::from_bits(
            scalars.lanes[buffers::GPU_STATS_LANE_INTERACTION_PRIMARY_CENTROID_Y_BITS],
        );
        let secondary_centroid = f32::from_bits(
            scalars.lanes[buffers::GPU_STATS_LANE_INTERACTION_SECONDARY_CENTROID_Y_BITS],
        );

        Self {
            total_particles: scalars.lanes[buffers::GPU_STATS_LANE_TOTAL_PARTICLES],
            water_liquid: scalars.lanes[buffers::GPU_STATS_LANE_PHASE_WATER],
            soil_granular: scalars.lanes[buffers::GPU_STATS_LANE_PHASE_GRANULAR_SOIL],
            sand_granular: scalars.lanes[buffers::GPU_STATS_LANE_PHASE_GRANULAR_SAND],
            unknown: scalars.lanes[buffers::GPU_STATS_LANE_PHASE_UNKNOWN],
            max_speed_mps: f32::from_bits(scalars.lanes[buffers::GPU_STATS_LANE_MAX_SPEED_BITS]),
            all_count,
            all_penetration_count: scalars.lanes[buffers::GPU_STATS_LANE_ALL_PENETRATION_COUNT],
            all_mean_y_m,
            tracked_count,
            tracked_penetration_count: scalars.lanes
                [buffers::GPU_STATS_LANE_TRACKED_PENETRATION_COUNT],
            tracked_mean_y_m,
            tracked_min_x_m: decode_optional_ordered_f32(
                scalars.lanes[buffers::GPU_STATS_LANE_TRACKED_X_MIN_ORDERED_BITS],
            ),
            tracked_max_x_m: decode_optional_ordered_f32(
                scalars.lanes[buffers::GPU_STATS_LANE_TRACKED_X_MAX_ORDERED_BITS],
            ),
            tracked_min_y_m: decode_optional_ordered_f32(
                scalars.lanes[buffers::GPU_STATS_LANE_TRACKED_Y_MIN_ORDERED_BITS],
            ),
            tracked_max_y_m: decode_optional_ordered_f32(
                scalars.lanes[buffers::GPU_STATS_LANE_TRACKED_Y_MAX_ORDERED_BITS],
            ),
            water_surface_p95_cell,
            granular_repose_angle_deg,
            granular_repose_base_span_cells,
            material_interaction_contact_ratio: f32::from_bits(
                scalars.lanes[buffers::GPU_STATS_LANE_INTERACTION_CONTACT_RATIO_BITS],
            ),
            material_interaction_primary_centroid_y_m: primary_centroid
                .is_finite()
                .then_some(primary_centroid),
            material_interaction_secondary_centroid_y_m: secondary_centroid
                .is_finite()
                .then_some(secondary_centroid),
            grid_water_phi_max: f32::from_bits(
                scalars.lanes[buffers::GPU_STATS_LANE_GRID_WATER_PHI_MAX_BITS],
            ),
            grid_water_phi_p99: f32::from_bits(
                scalars.lanes[buffers::GPU_STATS_LANE_GRID_WATER_PHI_P99_BITS],
            ),
            grid_water_phi_mean_nonzero: f32::from_bits(
                scalars.lanes[buffers::GPU_STATS_LANE_GRID_WATER_PHI_MEAN_NONZERO_BITS],
            ),
            grid_water_nonzero_nodes: scalars.lanes
                [buffers::GPU_STATS_LANE_GRID_WATER_NONZERO_NODES],
            grid_granular_phi_max: f32::from_bits(
                scalars.lanes[buffers::GPU_STATS_LANE_GRID_GRANULAR_PHI_MAX_BITS],
            ),
            grid_granular_phi_p99: f32::from_bits(
                scalars.lanes[buffers::GPU_STATS_LANE_GRID_GRANULAR_PHI_P99_BITS],
            ),
            grid_granular_phi_mean_nonzero: f32::from_bits(
                scalars.lanes[buffers::GPU_STATS_LANE_GRID_GRANULAR_PHI_MEAN_NONZERO_BITS],
            ),
            grid_granular_nonzero_nodes: scalars.lanes
                [buffers::GPU_STATS_LANE_GRID_GRANULAR_NONZERO_NODES],
        }
    }
}

#[derive(Clone, Debug)]
pub enum GpuWorldEditCommand {
    AddParticles {
        material: ParticleMaterial,
        count_per_cell: u32,
    },
    RemoveParticles {
        max_count: u32,
    },
}

#[derive(Clone, Debug, Message)]
pub struct GpuWorldEditRequest {
    pub cells: Vec<IVec2>,
    pub command: GpuWorldEditCommand,
}

/// System: apply queued world-edit requests onto GPU upload particles.
pub fn apply_world_edit_requests(
    control: Res<MpmGpuControl>,
    active_params: Res<ActivePhysicsParams>,
    mut terrain: ResMut<TerrainWorld>,
    mut generated_chunk_cache: Option<ResMut<TerrainGeneratedChunkCache>>,
    mut edit_requests: MessageReader<GpuWorldEditRequest>,
    mut deferred_requests: ResMut<DeferredGpuWorldEditRequests>,
    mut upload: ResMut<MpmGpuUploadRequest>,
    mut readback_status: ResMut<MpmParticleReadbackStatus>,
    mut full_readback_request: ResMut<MpmFullParticleReadbackRequest>,
    mut full_readback_cache: ResMut<MpmFullParticleReadbackCache>,
    mut residency: ResMut<MpmChunkResidencyState>,
    mut pending_gpu_adds: ResMut<PendingGpuParticleAdds>,
    mut add_queue: ResMut<MpmGpuWorldEditAddQueueRequest>,
    mut remove_queue: ResMut<MpmGpuWorldEditRemoveQueueRequest>,
) {
    for request in edit_requests.read() {
        deferred_requests.requests.push(request.clone());
    }

    add_queue.ops.clear();
    add_queue.base_particle_count = 0;
    add_queue.added_particle_count = 0;
    add_queue.particle_revision = 0;
    remove_queue.remove_particle_ids.clear();
    remove_queue.removed_particle_count = 0;
    remove_queue.particle_revision = 0;

    if control.init_only {
        deferred_requests.requests.clear();
        return;
    }
    let rho0 = active_params.0.water.rho0.max(1.0e-6);
    if deferred_requests.requests.is_empty() {
        queue_oldest_pending_gpu_add(&residency, &pending_gpu_adds, rho0, &mut add_queue);
        return;
    }
    let has_adds = deferred_requests
        .requests
        .iter()
        .any(|request| matches!(request.command, GpuWorldEditCommand::AddParticles { .. }));
    let has_removes = deferred_requests
        .requests
        .iter()
        .any(|request| matches!(request.command, GpuWorldEditCommand::RemoveParticles { .. }));

    if has_removes {
        let min_particle_revision = readback_status
            .particle_revision
            .max(pending_gpu_adds.latest_revision());
        let Some(payload) = full_readback_cache
            .payload_for_revision(min_particle_revision)
            .cloned()
        else {
            full_readback_request.request(min_particle_revision);
            return;
        };

        let remove_particle_ids =
            collect_remove_particle_ids(&payload.particles, &deferred_requests.requests);
        if remove_particle_ids.is_empty() {
            if !has_adds {
                deferred_requests.requests.clear();
                queue_oldest_pending_gpu_add(&residency, &pending_gpu_adds, rho0, &mut add_queue);
                return;
            }
        } else {
            let removed_particle_count = remove_particle_ids.len() as u32;
            let particle_revision =
                next_particle_revision(&upload, &readback_status, &pending_gpu_adds);
            upload.particle_revision = particle_revision;
            readback_status.particle_count = readback_status
                .particle_count
                .saturating_sub(removed_particle_count);
            readback_status.particle_revision = particle_revision;
            full_readback_cache.invalidate_before(particle_revision);
            remove_queue.particle_revision = particle_revision;
            remove_queue.removed_particle_count = removed_particle_count;
            remove_queue.remove_particle_ids = remove_particle_ids;
            if !has_adds {
                deferred_requests.requests.clear();
                queue_oldest_pending_gpu_add(&residency, &pending_gpu_adds, rho0, &mut add_queue);
                return;
            }
        }
    }

    if !has_adds {
        deferred_requests.requests.clear();
        queue_oldest_pending_gpu_add(&residency, &pending_gpu_adds, rho0, &mut add_queue);
        return;
    }

    let mut queued_gpu_add_ops: Vec<PendingGpuParticleAddOp> = Vec::new();
    let mut queued_gpu_add_particles = 0u32;
    let mut requested_add_cells = Vec::new();
    for request in &deferred_requests.requests {
        let GpuWorldEditCommand::AddParticles {
            material,
            count_per_cell,
        } = &request.command
        else {
            continue;
        };
        if *count_per_cell == 0 || mpm_phase_id_for_particle(*material).is_none() {
            continue;
        }
        for &cell in &request.cells {
            requested_add_cells.push(cell);
            queued_gpu_add_ops.push(PendingGpuParticleAddOp {
                cell,
                material: *material,
                count_per_cell: *count_per_cell,
            });
            queued_gpu_add_particles = queued_gpu_add_particles.saturating_add(*count_per_cell);
        }
    }
    if queued_gpu_add_ops.is_empty() {
        deferred_requests.requests.clear();
        if remove_queue.removed_particle_count > 0 {
            queue_oldest_pending_gpu_add(&residency, &pending_gpu_adds, rho0, &mut add_queue);
            return;
        }
        queue_oldest_pending_gpu_add(&residency, &pending_gpu_adds, rho0, &mut add_queue);
        return;
    }
    if !ensure_gpu_add_slots_for_cells(
        &mut terrain,
        generated_chunk_cache.as_mut().map(|cache| &mut **cache),
        &mut upload,
        &mut full_readback_request,
        &mut full_readback_cache,
        &mut readback_status,
        &mut pending_gpu_adds,
        &mut deferred_requests.requests,
        &mut residency,
        &requested_add_cells,
        rho0,
    ) {
        return;
    }
    deferred_requests.requests.clear();

    let particle_revision = next_particle_revision(&upload, &readback_status, &pending_gpu_adds);
    let mut gpu_ops = Vec::with_capacity(queued_gpu_add_ops.len());
    let mut particle_offset = 0u32;
    for op in &queued_gpu_add_ops {
        let Some((slot_id, local_cell)) = gpu_add_slot_for_cell(&residency, op.cell) else {
            continue;
        };
        let Some(phase_id) = mpm_phase_id_for_particle(op.material) else {
            continue;
        };
        let props = particle_properties(op.material);
        gpu_ops.push(GpuWorldEditAddOp {
            slot_id,
            local_cell_x: local_cell.x as u32,
            local_cell_y: local_cell.y as u32,
            count_per_cell: op.count_per_cell,
            particle_offset,
            phase_id: phase_id as u32,
            mass: props.mass,
            v0: props.mass.max(0.0) / rho0,
        });
        particle_offset = particle_offset.saturating_add(op.count_per_cell);
    }
    if gpu_ops.is_empty() {
        queue_oldest_pending_gpu_add(&residency, &pending_gpu_adds, rho0, &mut add_queue);
        return;
    }

    let expected_total_particle_count = readback_status
        .particle_count
        .saturating_add(queued_gpu_add_particles);
    pending_gpu_adds.push_batch(PendingGpuParticleAddBatch {
        particle_revision,
        added_particle_count: queued_gpu_add_particles,
        expected_total_particle_count,
        ops: queued_gpu_add_ops,
    });
    upload.particle_revision = particle_revision;
    readback_status.particle_count = readback_status
        .particle_count
        .saturating_add(queued_gpu_add_particles);
    readback_status.particle_revision = particle_revision;
    full_readback_cache.invalidate_before(particle_revision);
    queue_oldest_pending_gpu_add(&residency, &pending_gpu_adds, rho0, &mut add_queue);
}

/// System: keep GPU MPM run-state aligned to global simulation mode.
pub fn prepare_particle_upload(
    control: Res<MpmGpuControl>,
    readback_status: Res<MpmParticleReadbackStatus>,
    mut upload: ResMut<MpmGpuUploadRequest>,
    mut sim_state: ResMut<SimulationState>,
) {
    if control.init_only {
        upload.upload_particles = false;
        upload.upload_particles_frame = false;
        upload.upload_mover_results = false;
        upload.upload_mover_results_frame = false;
        upload.upload_chunks = false;
        upload.upload_chunk_diffs = false;
        sim_state.gpu_mpm_active = false;
        upload.particles.clear();
        upload.mover_results.clear();
        upload.chunk_meta_diffs.clear();
        upload.terrain_slot_ids.clear();
        upload.terrain_cell_solid_slot_diffs.clear();
        return;
    }
    let upload_requested = upload.upload_particles;
    upload.upload_particles_frame = upload_requested;
    // Consume main-world request flag; explicit requests set this again when needed.
    upload.upload_particles = false;
    let mover_upload_requested = upload.upload_mover_results;
    upload.upload_mover_results_frame = mover_upload_requested;
    upload.upload_mover_results = false;

    // Keep GPU run-state stable regardless of one-shot upload flag consumption.
    // `upload.particles` is the authoritative CPU-side snapshot for the current set.
    let has_particles = if upload.upload_particles_frame {
        !upload.particles.is_empty()
    } else {
        readback_status.particle_count > 0
    };
    // Pure GPU mode: no CPU fallback; active when MPM is enabled and particles exist.
    sim_state.gpu_mpm_active = sim_state.mpm_enabled && has_particles;
}

/// System: build terrain SDF/normal upload request.
///
/// Only re-uploads when TerrainWorld changes.
pub fn prepare_terrain_upload(
    control: Res<MpmGpuControl>,
    sim_state: Res<SimulationState>,
    mut terrain: ResMut<TerrainWorld>,
    mut generated_chunk_cache: Option<ResMut<TerrainGeneratedChunkCache>>,
    readback_status: Res<MpmParticleReadbackStatus>,
    mut full_readback_request: ResMut<MpmFullParticleReadbackRequest>,
    full_readback_cache: Res<MpmFullParticleReadbackCache>,
    mut residency: ResMut<MpmChunkResidencyState>,
    mut upload: ResMut<MpmGpuUploadRequest>,
) {
    if !upload.upload_chunk_diffs {
        upload.chunk_meta_diffs.clear();
    }
    if !upload.upload_terrain_cell_slot_diffs {
        upload.terrain_slot_ids.clear();
        upload.terrain_cell_solid_slot_diffs.clear();
    }
    upload.upload_terrain = false;

    if control.init_only {
        upload.chunk_meta.clear();
        upload.terrain_sdf.clear();
        upload.terrain_normal.clear();
        upload.terrain_slot_ids.clear();
        upload.terrain_cell_solid_slot_diffs.clear();
        upload.last_uploaded_terrain_version = None;
        upload.chunk_meta_diffs.clear();
        residency.initialized = false;
        residency.resident_chunk_count = 0;
        residency.active_resident_chunk_count = 0;
        residency.chunk_sdf_samples = 0;
        residency.active_tile_count = 0;
        residency.active_tile_capacity = 0;
        residency.inactive_skip_rate = 0.0;
        residency.invalid_slot_access_count = 0;
        residency.runtime_rebuild_count = 0;
        residency.runtime_rebuild_reason_empty_state = 0;
        residency.runtime_rebuild_reason_invalid_old_slot = 0;
        residency.runtime_rebuild_reason_new_chunk_oob = 0;
        residency.runtime_rebuild_reason_old_slot_empty = 0;
        residency.runtime_rebuild_reason_halo_update_fail = 0;
        residency.runtime_rebuild_waiting_readback_count = 0;
        residency.pending_mover_apply = false;
        residency.pending_mover_readback = false;
        residency.observed_mover_apply_ack = 0;
        residency.mover_readback_flush_budget = 0;
        residency.rebuild_requested = false;
        residency.chunk_to_slot.clear();
        residency.slot_to_chunk.clear();
        residency.slot_allocated.clear();
        residency.free_slots.clear();
        residency.slot_state.clear();
        residency.chunk_meta_cache.clear();
        return;
    }
    // When MLS-MPM stepping is disabled, terrain upload is unnecessary for overlay debug.
    if !sim_state.mpm_enabled {
        return;
    }

    // MPM-CHUNK-01: static residency.
    // Rebuild only on initial bring-up or explicit particle upload (scenario/reset/edit).
    let should_rebuild =
        !residency.initialized || upload.upload_particles_frame || residency.rebuild_requested;
    if !should_rebuild {
        let terrain_version = terrain.terrain_version();
        if upload.last_uploaded_terrain_version != Some(terrain_version) && residency.initialized {
            let dirty_chunks = terrain.dirty_chunk_coords();
            let dirty_slots = collect_slots_for_dirty_chunks(&residency, &dirty_chunks);
            enqueue_terrain_slot_updates(&residency, &terrain, &dirty_slots, &mut upload);
            upload.last_uploaded_terrain_version = Some(terrain_version);
            if !dirty_slots.is_empty() {
                residency.chunk_sdf_samples = residency
                    .resident_chunk_count
                    .saturating_mul(MPM_CHUNK_NODES_PER_SLOT);
            }
        }
        return;
    }
    if residency.rebuild_requested {
        bevy::log::warn!(
            "[gpu_mpm] runtime rebuild start: particles_upload={} particles_readback={} reason_new_chunk_oob={} reason_invalid_old_slot={} wait_frames={}",
            upload.particles.len(),
            readback_status.particle_count,
            residency.runtime_rebuild_reason_new_chunk_oob,
            residency.runtime_rebuild_reason_invalid_old_slot,
            residency.runtime_rebuild_waiting_readback_count
        );
    }

    let rebuild_particles: Vec<GpuParticle>;
    let particle_source: &[GpuParticle] = if upload.upload_particles_frame && !upload.particles.is_empty() {
        upload.particles.as_slice()
    } else if residency.rebuild_requested {
        let min_particle_revision = readback_status.particle_revision;
        let Some(payload) = full_readback_cache.payload_for_revision(min_particle_revision) else {
            full_readback_request.request(min_particle_revision);
            residency.runtime_rebuild_waiting_readback_count = residency
                .runtime_rebuild_waiting_readback_count
                .saturating_add(1);
            if residency.runtime_rebuild_waiting_readback_count == 1 {
                bevy::log::warn!(
                    "[gpu_mpm] runtime rebuild pending: waiting for on-demand GPU particle readback"
                );
            }
            return;
        };
        rebuild_particles = payload.particles.clone();
        rebuild_particles.as_slice()
    } else {
        return;
    };
    let Some(build) = build_static_chunk_upload(&terrain, particle_source) else {
        upload.chunk_meta.clear();
        upload.terrain_sdf.clear();
        upload.terrain_normal.clear();
        upload.terrain_slot_ids.clear();
        upload.terrain_cell_solid_slot_diffs.clear();
        residency.initialized = false;
        residency.resident_chunk_count = 0;
        residency.active_resident_chunk_count = 0;
        residency.chunk_sdf_samples = 0;
        residency.active_tile_count = 0;
        residency.active_tile_capacity = 0;
        residency.inactive_skip_rate = 0.0;
        residency.runtime_rebuild_count = 0;
        residency.runtime_rebuild_reason_empty_state = 0;
        residency.runtime_rebuild_reason_invalid_old_slot = 0;
        residency.runtime_rebuild_reason_new_chunk_oob = 0;
        residency.runtime_rebuild_reason_old_slot_empty = 0;
        residency.runtime_rebuild_reason_halo_update_fail = 0;
        residency.runtime_rebuild_waiting_readback_count = 0;
        residency.pending_mover_apply = false;
        residency.pending_mover_readback = false;
        residency.observed_mover_apply_ack = 0;
        residency.mover_readback_flush_budget = 0;
        residency.rebuild_requested = false;
        residency.chunk_to_slot.clear();
        residency.slot_to_chunk.clear();
        residency.slot_allocated.clear();
        residency.free_slots.clear();
        residency.slot_state.clear();
        residency.chunk_meta_cache.clear();
        return;
    };

    residency.initialized = true;
    residency.chunk_origin = build.chunk_origin;
    residency.chunk_dims = build.chunk_dims;
    residency.grid_layout = build.grid_layout;
    if !initialize_slot_pool(&mut residency, &build.resident_chunks) {
        bevy::log::warn!(
            "[gpu_mpm] failed to initialize chunk slot pool: requested={} limit={}",
            build.resident_chunks.len(),
            MAX_RESIDENT_CHUNK_SLOTS
        );
        residency.initialized = false;
        return;
    }

    if upload.upload_particles_frame && !upload.particles.is_empty() {
        let unresolved = assign_home_slot_ids(&mut upload.particles, &residency.chunk_to_slot);
        if unresolved > 0 {
            bevy::log::warn!(
                "[gpu_mpm] unresolved home slots during residency rebuild (upload): {}",
                unresolved
            );
        }
        upload.mover_results.clear();
        upload.upload_mover_results = false;
        upload.upload_mover_results_frame = false;
        residency.pending_mover_apply = false;
        residency.pending_mover_readback = false;
    } else {
        let mut rebuilt_particles = particle_source.to_vec();
        let unresolved = assign_home_slot_ids(&mut rebuilt_particles, &residency.chunk_to_slot);
        if unresolved > 0 {
            bevy::log::warn!(
                "[gpu_mpm] unresolved home slots during residency rebuild (readback): {}",
                unresolved
            );
        }
        upload.mover_results = rebuilt_particles
            .iter()
            .enumerate()
            .map(|(pid, particle)| GpuMoverResult {
                particle_id: pid as u32,
                new_home_slot_id: particle.home_chunk_slot_id,
                _pad_a: 0,
                _pad_b: 0,
            })
            .collect();
        upload.upload_mover_results = !upload.mover_results.is_empty();
        upload.upload_mover_results_frame = false;
        residency.pending_mover_apply = upload.upload_mover_results;
        residency.pending_mover_readback = false;
        // Keep CPU snapshot in sync with latest readback-derived positions.
        upload.particles = rebuilt_particles;
        upload.upload_particles = false;
        upload.upload_particles_frame = false;
    }

    initialize_chunk_slot_state(&mut residency, upload.particles.as_slice());
    update_active_tile_stats(&mut residency, upload.particles.as_slice());
    let slot_span = residency.resident_chunk_count as usize;
    upload.chunk_meta = residency.chunk_meta_cache[..slot_span].to_vec();
    upload.upload_chunks = true;
    upload.terrain_sdf.clear();
    upload.terrain_normal.clear();
    upload.upload_terrain = false;
    let full_slots = collect_allocated_slots(&residency);
    let full_slots_ready = ensure_chunks_available_for_slots(
        &mut terrain,
        generated_chunk_cache.as_mut().map(|cache| &mut **cache),
        &residency,
        &full_slots,
    );
    if full_slots_ready.len() < full_slots.len() {
        bevy::log::debug!(
            "[gpu_mpm] terrain slot updates pending generated chunks: ready={} requested={}",
            full_slots_ready.len(),
            full_slots.len()
        );
    }
    enqueue_terrain_slot_updates(&residency, &terrain, &full_slots_ready, &mut upload);
    upload.last_uploaded_terrain_version = Some(terrain.terrain_version());
    residency.active_resident_chunk_count = residency
        .slot_state
        .iter()
        .enumerate()
        .filter(|(idx, slot)| {
            residency.slot_allocated.get(*idx).copied().unwrap_or(false) && slot.resident
        })
        .count() as u32;
    residency.active_tile_capacity = residency
        .resident_chunk_count
        .saturating_mul(MPM_ACTIVE_TILES_PER_SLOT);
    if residency.active_tile_capacity == 0 {
        residency.active_tile_count = 0;
        residency.inactive_skip_rate = 0.0;
    }
    residency.chunk_sdf_samples = residency
        .resident_chunk_count
        .saturating_mul(MPM_CHUNK_NODES_PER_SLOT);
    residency.invalid_slot_access_count = 0;
    residency.mover_readback_flush_budget = 2;
    residency.rebuild_requested = false;
}

/// Latch one-shot non-particle upload flags for render extraction, then clear the main-world bits.
pub fn prepare_aux_upload_frames(mut upload: ResMut<MpmGpuUploadRequest>) {
    upload.upload_chunks_frame = std::mem::take(&mut upload.upload_chunks);
    upload.upload_chunk_diffs_frame = std::mem::take(&mut upload.upload_chunk_diffs);
    upload.upload_terrain_frame = std::mem::take(&mut upload.upload_terrain);
    upload.upload_terrain_cell_slot_diffs_frame =
        std::mem::take(&mut upload.upload_terrain_cell_slot_diffs);
}

/// System: update MpmGpuParamsRequest from simulation state and PhysicsParams asset.
pub fn prepare_gpu_params(
    control: Res<MpmGpuControl>,
    active_params: Res<ActivePhysicsParams>,
    upload: Res<MpmGpuUploadRequest>,
    readback_status: Res<MpmParticleReadbackStatus>,
    residency: Res<MpmChunkResidencyState>,
    stats_status: Res<MpmStatisticsStatus>,
    mut params_req: ResMut<MpmGpuParamsRequest>,
) {
    if control.init_only {
        return;
    }
    let layout = if residency.initialized {
        residency.grid_layout
    } else {
        world_grid_layout()
    };
    let h = MPM_NODE_SPACING_M;
    let p = &active_params.0;
    let soil = drucker_prager_params(
        p.soil.youngs_modulus_pa,
        p.soil.poisson_ratio,
        p.soil.friction_deg,
        p.soil.cohesion_pa,
        p.soil.hardening,
    );
    let sand = drucker_prager_params(
        p.sand.youngs_modulus_pa,
        p.sand.poisson_ratio,
        p.sand.friction_deg,
        p.sand.cohesion_pa,
        p.sand.hardening,
    );
    let boundary_threshold_m = h * p.runtime.boundary_velocity_sdf_threshold_h;
    let particle_count = if upload.upload_particles_frame {
        upload.particles.len()
    } else {
        readback_status.particle_count as usize
    };

    params_req.params = GpuMpmParams {
        dt: p.fixed_dt,
        gx: 0.0,
        gy: -9.81,
        rho0: p.water.rho0,
        bulk_modulus: p.water.rho0 * p.water.sound_speed_mps * p.water.sound_speed_mps,
        h,
        grid_origin_x: layout.origin.x,
        grid_origin_y: layout.origin.y,
        grid_width: layout.dims.x,
        grid_height: layout.dims.y,
        particle_count: particle_count as u32,
        j_min: p.deformation.j_min,
        j_max: p.deformation.j_max,
        c_max_norm: p.deformation.c_max_norm,
        sdf_velocity_threshold_m: boundary_threshold_m,
        boundary_normal_projection_scale: p.runtime.boundary_velocity_normal_projection_scale,
        boundary_friction_water: p.boundary.water,
        boundary_friction_granular: p.boundary.granular,
        dp_lambda_soil: soil.lambda,
        dp_mu_soil: soil.mu,
        dp_alpha_soil: soil.alpha,
        dp_k_soil: soil.k,
        dp_hardening_soil: soil.hardening,
        dp_lambda_sand: sand.lambda,
        dp_mu_sand: sand.mu,
        dp_alpha_sand: sand.alpha,
        dp_k_sand: sand.k,
        dp_hardening_sand: sand.hardening,
        granular_tensile_clamp: p.granular_tensile_clamp,
        coupling_drag_gamma: p.coupling.drag_gamma,
        coupling_friction: p.coupling.friction,
        coupling_interface_min_grad: p.coupling.interface_min_grad,
        coupling_interface_normal_eps: p.coupling.interface_normal_eps,
        alpha_apic_water: p.apic.water,
        alpha_apic_granular: p.apic.granular,
        stats_tracked_phase_id: stats_status.tracked_phase_id,
        stats_repose_phase_id: stats_status.repose_phase_id,
        stats_interaction_primary_phase_id: stats_status.interaction_primary_phase_id,
        stats_interaction_secondary_phase_id: stats_status.interaction_secondary_phase_id,
        stats_penetration_epsilon_m: p.runtime.stats_penetration_epsilon_m,
        stats_position_fp_scale: 100.0,
        chunk_origin_x: residency.chunk_origin.x,
        chunk_origin_y: residency.chunk_origin.y,
        chunk_dims_x: residency.chunk_dims.x,
        chunk_dims_y: residency.chunk_dims.y,
        resident_chunk_count: residency.resident_chunk_count,
        chunk_node_dim: MPM_CHUNK_NODE_DIM,
        _pad_tail: [0; 1],
    };
}

/// System: update GPU compute run state from SimulationState.
pub fn prepare_gpu_run_state(
    control: Res<MpmGpuControl>,
    mut sim_state: ResMut<SimulationState>,
    mut replay_state: ResMut<ReplayState>,
    active_params: Res<ActivePhysicsParams>,
    time: Res<Time>,
    mut step_clock: ResMut<MpmGpuStepClock>,
    mut run_req: ResMut<MpmGpuRunRequest>,
) {
    if control.init_only {
        run_req.enabled = false;
        run_req.substeps = 0;
        step_clock.accumulator_secs = 0.0;
        sim_state.step_once = false;
        return;
    }
    let active = sim_state.mpm_enabled && sim_state.gpu_mpm_active;
    if !active {
        run_req.enabled = false;
        run_req.substeps = 0;
        step_clock.accumulator_secs = 0.0;
        sim_state.step_once = false;
        return;
    }

    // Single-step command always executes exactly one substep.
    if sim_state.step_once {
        run_req.enabled = true;
        run_req.substeps = 1;
        step_clock.accumulator_secs = 0.0;
        if replay_state.enabled {
            replay_state.current_step = replay_state.current_step.saturating_add(1);
        }
        sim_state.step_once = false;
        return;
    }

    if !sim_state.running {
        run_req.enabled = false;
        run_req.substeps = 0;
        return;
    }

    let fixed_dt = active_params.0.fixed_dt.max(1.0e-5);
    let max_substeps_per_frame = active_params.0.runtime.max_substeps_per_frame.max(1);
    step_clock.max_substeps_per_frame = max_substeps_per_frame;
    // Avoid runaway catch-up (spiral of death). If we can't keep up, simulation slows down.
    let max_catchup = fixed_dt * max_substeps_per_frame as f32;
    step_clock.accumulator_secs =
        (step_clock.accumulator_secs + time.delta_secs()).min(max_catchup);

    let mut substeps = (step_clock.accumulator_secs / fixed_dt).floor() as u32;
    if substeps > max_substeps_per_frame {
        substeps = max_substeps_per_frame;
    }
    if substeps == 0 {
        run_req.enabled = false;
        run_req.substeps = 0;
        return;
    }

    step_clock.accumulator_secs =
        (step_clock.accumulator_secs - fixed_dt * substeps as f32).max(0.0);
    run_req.enabled = true;
    run_req.substeps = substeps;
    if replay_state.enabled {
        replay_state.current_step = replay_state.current_step.saturating_add(substeps as usize);
    }
}

/// System: apply on-demand GPU readback results to cache.
pub fn apply_gpu_readback(
    control: Res<MpmGpuControl>,
    readback_result: Res<GpuReadbackResult>,
    mut request: ResMut<MpmFullParticleReadbackRequest>,
    mut cache: ResMut<MpmFullParticleReadbackCache>,
) {
    if !control.readback_enabled {
        return;
    }
    let Some(payload) = readback_result.take() else {
        return;
    };
    if !request.requested {
        return;
    }
    if payload.particle_revision < request.min_particle_revision {
        return;
    }
    cache.payload = Some(payload.clone());
    request.clear_if_satisfied(payload.particle_revision);
}

pub fn consume_world_edit_add_ack(
    control: Res<MpmGpuControl>,
    add_apply_ack: Res<GpuWorldEditAddApplyAck>,
    mut pending_gpu_adds: ResMut<PendingGpuParticleAdds>,
) {
    if control.init_only {
        return;
    }
    let acked_revision = add_apply_ack.value();
    if acked_revision == 0 {
        return;
    }
    pending_gpu_adds.drop_acked_batches(acked_revision);
}

fn gpu_add_slot_for_cell(
    residency: &MpmChunkResidencyState,
    cell: IVec2,
) -> Option<(u32, IVec2)> {
    if !residency.initialized {
        return None;
    }
    let chunk = IVec2::new(
        cell.x.div_euclid(CHUNK_SIZE_I32),
        cell.y.div_euclid(CHUNK_SIZE_I32),
    );
    let slot_id = residency.chunk_to_slot.get(&chunk).copied()?;
    let slot_idx = slot_id as usize;
    if !residency
        .slot_allocated
        .get(slot_idx)
        .copied()
        .unwrap_or(false)
    {
        return None;
    }
    let local_cell = IVec2::new(
        cell.x.rem_euclid(CHUNK_SIZE_I32),
        cell.y.rem_euclid(CHUNK_SIZE_I32),
    );
    Some((slot_id, local_cell))
}

fn build_gpu_add_queue_from_batch(
    residency: &MpmChunkResidencyState,
    batch: &PendingGpuParticleAddBatch,
    rho0: f32,
) -> Option<MpmGpuWorldEditAddQueueRequest> {
    let mut ops = Vec::with_capacity(batch.ops.len());
    let mut particle_offset = 0u32;
    for op in &batch.ops {
        let Some((slot_id, local_cell)) = gpu_add_slot_for_cell(residency, op.cell) else {
            continue;
        };
        let Some(phase_id) = mpm_phase_id_for_particle(op.material) else {
            continue;
        };
        let props = particle_properties(op.material);
        ops.push(GpuWorldEditAddOp {
            slot_id,
            local_cell_x: local_cell.x as u32,
            local_cell_y: local_cell.y as u32,
            count_per_cell: op.count_per_cell,
            particle_offset,
            phase_id: phase_id as u32,
            mass: props.mass,
            v0: props.mass.max(0.0) / rho0.max(1.0e-6),
        });
        particle_offset = particle_offset.saturating_add(op.count_per_cell);
    }
    if ops.is_empty() {
        return None;
    }
    Some(MpmGpuWorldEditAddQueueRequest {
        particle_revision: batch.particle_revision,
        base_particle_count: batch
            .expected_total_particle_count
            .saturating_sub(batch.added_particle_count),
        added_particle_count: batch.added_particle_count,
        ops,
    })
}

fn queue_oldest_pending_gpu_add(
    residency: &MpmChunkResidencyState,
    pending_gpu_adds: &PendingGpuParticleAdds,
    rho0: f32,
    add_queue: &mut MpmGpuWorldEditAddQueueRequest,
) {
    if let Some(batch) = pending_gpu_adds.oldest_batch() {
        if let Some(request) = build_gpu_add_queue_from_batch(residency, batch, rho0) {
            *add_queue = request;
        }
    }
}

fn collect_remove_particle_ids(
    particles: &[GpuParticle],
    requests: &[GpuWorldEditRequest],
) -> Vec<u32> {
    let mut removed = Vec::new();
    let mut keep = vec![true; particles.len()];
    for request in requests {
        let GpuWorldEditCommand::RemoveParticles { max_count } = request.command else {
            continue;
        };
        if max_count == 0 || request.cells.is_empty() {
            continue;
        }
        let target_cells: HashSet<IVec2> = request.cells.iter().copied().collect();
        let max_count = max_count as usize;
        let mut removed_this_request = 0usize;
        for (particle_id, particle) in particles.iter().enumerate() {
            if removed_this_request >= max_count {
                break;
            }
            if !keep[particle_id] {
                continue;
            }
            let cell = world_to_cell(Vec2::from_array(particle.x));
            if !target_cells.contains(&cell) {
                continue;
            }
            keep[particle_id] = false;
            removed.push(particle_id as u32);
            removed_this_request += 1;
        }
    }
    removed.sort_unstable();
    removed
}

fn ensure_gpu_add_slots_for_cells(
    terrain: &mut TerrainWorld,
    mut generated_chunk_cache: Option<&mut TerrainGeneratedChunkCache>,
    upload: &mut MpmGpuUploadRequest,
    full_readback_request: &mut MpmFullParticleReadbackRequest,
    full_readback_cache: &mut MpmFullParticleReadbackCache,
    readback_status: &mut MpmParticleReadbackStatus,
    pending_gpu_adds: &mut PendingGpuParticleAdds,
    deferred_requests: &mut Vec<GpuWorldEditRequest>,
    residency: &mut MpmChunkResidencyState,
    cells: &[IVec2],
    rho0: f32,
) -> bool {
    if cells.is_empty() {
        return true;
    }

    let residency_was_uninitialized = !residency.initialized;
    if !residency.initialized {
        if upload.upload_particles && !upload.particles.is_empty() {
            let mut changed = false;
            for request in deferred_requests.iter() {
                let GpuWorldEditCommand::AddParticles {
                    material,
                    count_per_cell,
                } = request.command
                else {
                    continue;
                };
                if count_per_cell == 0 || mpm_phase_id_for_particle(material).is_none() {
                    continue;
                }
                let before = upload.particles.len();
                for &cell in &request.cells {
                    append_particles_in_cell(
                        &mut upload.particles,
                        cell,
                        material,
                        count_per_cell,
                        rho0,
                    );
                }
                changed |= upload.particles.len() != before;
            }
            if changed {
                upload.particle_revision =
                    next_particle_revision(upload, readback_status, pending_gpu_adds);
                readback_status.particle_count = upload.particles.len() as u32;
                readback_status.particle_revision = upload.particle_revision;
                full_readback_cache.payload = Some(GpuParticleReadbackPayload {
                    particles: upload.particles.clone(),
                    particle_revision: upload.particle_revision,
                });
                pending_gpu_adds.clear();
                full_readback_request.requested = false;
                full_readback_request.min_particle_revision = 0;
            }
            return false;
        }

        if readback_status.particle_count > 0 {
            let min_particle_revision = readback_status
                .particle_revision
                .max(pending_gpu_adds.latest_revision());
            let Some(payload) = full_readback_cache
                .payload_for_revision(min_particle_revision)
                .cloned()
            else {
                full_readback_request.request(min_particle_revision);
                return false;
            };
            if !bootstrap_residency_from_particles(
                terrain,
                generated_chunk_cache.as_deref_mut(),
                residency,
                upload,
                &payload.particles,
            ) {
                return false;
            }
        }
    }

    let mut needed_chunks = HashSet::<IVec2>::default();
    let mut target_chunks = Vec::new();
    for &cell in cells {
        let chunk = IVec2::new(
            cell.x.div_euclid(CHUNK_SIZE_I32),
            cell.y.div_euclid(CHUNK_SIZE_I32),
        );
        target_chunks.push(chunk);
        let chunk_already_allocated = residency.initialized && residency.chunk_to_slot.contains_key(&chunk);
        if chunk_already_allocated {
            needed_chunks.insert(chunk);
        } else {
            for oy in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
                for ox in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
                    needed_chunks.insert(chunk + IVec2::new(ox, oy));
                }
            }
        }
    }
    let mut needed_chunks = needed_chunks.into_iter().collect::<Vec<_>>();
    needed_chunks.sort_by_key(|chunk| (chunk.y, chunk.x));

    let mut allocated_new_slots = residency_was_uninitialized;
    let seed_slots = if !residency.initialized {
        if !initialize_slot_pool(residency, &needed_chunks) {
            return false;
        }
        residency.initialized = true;
        initialize_chunk_slot_state(residency, &[]);
        let _ = stage_pending_add_target_residency(residency, &target_chunks);
        let slot_count = residency.slot_to_chunk.len();
        residency.chunk_meta_cache = (0..slot_count as u32)
            .map(|slot_id| {
                build_chunk_meta_for_slot(
                    slot_id,
                    &residency.slot_to_chunk,
                    &residency.chunk_to_slot,
                    &residency.slot_state,
                    &residency.slot_allocated,
                )
            })
            .collect();
        let slot_span = residency.resident_chunk_count as usize;
        upload.chunk_meta = residency.chunk_meta_cache[..slot_span].to_vec();
        upload.upload_chunks = !upload.chunk_meta.is_empty();
        upload.chunk_meta_diffs.clear();
        upload.upload_chunk_diffs = false;
        target_chunks
            .iter()
            .filter_map(|&chunk| slot_id_from_chunk_coord(chunk, &residency.chunk_to_slot))
            .collect::<Vec<_>>()
    } else {
        let mut dirty_slots = HashSet::<u32>::default();
        for chunk in needed_chunks {
            let existed = residency.chunk_to_slot.contains_key(&chunk);
            let Some(slot_id) = allocate_chunk_slot(residency, chunk) else {
                return false;
            };
            allocated_new_slots |= !existed;
            dirty_slots.insert(slot_id);
            for offset in MOORE_OFFSETS {
                if let Some(neighbor_slot) = slot_id_from_chunk_coord(chunk + offset, &residency.chunk_to_slot)
                {
                    dirty_slots.insert(neighbor_slot);
                }
            }
        }
        let seed_slots = target_chunks
            .iter()
            .filter_map(|&chunk| slot_id_from_chunk_coord(chunk, &residency.chunk_to_slot))
            .collect::<Vec<_>>();
        let mut refresh_slots = dirty_slots.into_iter().collect::<Vec<_>>();
        refresh_slots.extend(stage_pending_add_target_residency(residency, &target_chunks));
        refresh_slots.sort_unstable();
        refresh_slots.dedup();
        enqueue_chunk_meta_refresh_for_slots(residency, upload, &refresh_slots);
        seed_slots
    };

    let update_slots = collect_slots_with_neighbors(residency, &seed_slots);
    let ready_slots = ensure_chunks_available_for_slots(
        terrain,
        generated_chunk_cache,
        residency,
        &update_slots,
    );
    enqueue_terrain_slot_updates(residency, terrain, &ready_slots, upload);
    residency.chunk_sdf_samples = residency
        .resident_chunk_count
        .saturating_mul(MPM_CHUNK_NODES_PER_SLOT);
    residency.active_tile_capacity = residency
        .resident_chunk_count
        .saturating_mul(MPM_ACTIVE_TILES_PER_SLOT);
    !allocated_new_slots
}

fn enqueue_chunk_meta_refresh_for_slots(
    residency: &mut MpmChunkResidencyState,
    upload: &mut MpmGpuUploadRequest,
    slot_ids: &[u32],
) {
    if slot_ids.is_empty() {
        return;
    }
    let mut dirty_slots = slot_ids.to_vec();
    dirty_slots.sort_unstable();
    dirty_slots.dedup();
    let mut new_diffs = Vec::new();
    for slot_id in dirty_slots {
        let slot_idx = slot_id as usize;
        if !residency
            .slot_allocated
            .get(slot_idx)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        let meta = build_chunk_meta_for_slot(
            slot_id,
            &residency.slot_to_chunk,
            &residency.chunk_to_slot,
            &residency.slot_state,
            &residency.slot_allocated,
        );
        if residency.chunk_meta_cache.get(slot_idx).copied() != Some(meta) {
            if slot_idx < residency.chunk_meta_cache.len() {
                residency.chunk_meta_cache[slot_idx] = meta;
            }
            new_diffs.push((slot_id, meta));
        }
    }
    stage_chunk_meta_uploads(upload, residency, new_diffs);
}

fn stage_chunk_meta_uploads(
    upload: &mut MpmGpuUploadRequest,
    residency: &MpmChunkResidencyState,
    new_diffs: Vec<(u32, GpuChunkMeta)>,
) {
    let slot_span = residency.resident_chunk_count as usize;
    if upload.upload_chunks {
        upload.chunk_meta = residency.chunk_meta_cache[..slot_span].to_vec();
        upload.upload_chunk_diffs = false;
        upload.chunk_meta_diffs.clear();
        return;
    }

    if !new_diffs.is_empty() {
        upload.chunk_meta_diffs.extend(new_diffs);
        upload.chunk_meta_diffs.sort_by_key(|(slot_id, _)| *slot_id);
        upload.chunk_meta_diffs.dedup_by(|a, b| {
            if a.0 == b.0 {
                *a = *b;
                true
            } else {
                false
            }
        });
    }

    if !upload.chunk_meta_diffs.is_empty()
        && upload.chunk_meta_diffs.len() >= slot_span.saturating_div(2).max(1)
    {
        upload.chunk_meta = residency.chunk_meta_cache[..slot_span].to_vec();
        upload.upload_chunks = true;
        upload.upload_chunk_diffs = false;
        upload.chunk_meta_diffs.clear();
    } else {
        upload.upload_chunk_diffs = !upload.chunk_meta_diffs.is_empty();
    }
}

fn next_particle_revision(
    upload: &MpmGpuUploadRequest,
    readback_status: &MpmParticleReadbackStatus,
    pending_gpu_adds: &PendingGpuParticleAdds,
) -> u64 {
    upload
        .particle_revision
        .max(readback_status.particle_revision)
        .max(pending_gpu_adds.latest_revision())
        .saturating_add(1)
}

fn bootstrap_residency_from_particles(
    terrain: &mut TerrainWorld,
    mut generated_chunk_cache: Option<&mut TerrainGeneratedChunkCache>,
    residency: &mut MpmChunkResidencyState,
    upload: &mut MpmGpuUploadRequest,
    particles: &[GpuParticle],
) -> bool {
    let Some(build) = build_static_chunk_upload(terrain, particles) else {
        return false;
    };

    residency.initialized = true;
    residency.chunk_origin = build.chunk_origin;
    residency.chunk_dims = build.chunk_dims;
    residency.grid_layout = build.grid_layout;
    if !initialize_slot_pool(residency, &build.resident_chunks) {
        bevy::log::warn!(
            "[gpu_mpm] failed to initialize chunk slot pool for world edit bootstrap: requested={} limit={}",
            build.resident_chunks.len(),
            MAX_RESIDENT_CHUNK_SLOTS
        );
        residency.initialized = false;
        return false;
    }

    let mut rebuilt_particles = particles.to_vec();
    let unresolved = assign_home_slot_ids(&mut rebuilt_particles, &residency.chunk_to_slot);
    if unresolved > 0 {
        bevy::log::warn!(
            "[gpu_mpm] unresolved home slots during world edit residency bootstrap: {}",
            unresolved
        );
    }
    upload.mover_results = rebuilt_particles
        .iter()
        .enumerate()
        .map(|(pid, particle)| GpuMoverResult {
            particle_id: pid as u32,
            new_home_slot_id: particle.home_chunk_slot_id,
            _pad_a: 0,
            _pad_b: 0,
        })
        .collect();
    upload.upload_mover_results = !upload.mover_results.is_empty();
    upload.upload_mover_results_frame = false;
    residency.pending_mover_apply = upload.upload_mover_results;
    residency.pending_mover_readback = false;

    initialize_chunk_slot_state(residency, &rebuilt_particles);
    update_active_tile_stats(residency, &rebuilt_particles);
    let slot_span = residency.resident_chunk_count as usize;
    upload.chunk_meta = residency.chunk_meta_cache[..slot_span].to_vec();
    upload.upload_chunks = !upload.chunk_meta.is_empty();
    upload.upload_chunk_diffs = false;
    upload.chunk_meta_diffs.clear();

    let full_slots = collect_allocated_slots(residency);
    let full_slots_ready = ensure_chunks_available_for_slots(
        terrain,
        generated_chunk_cache.as_mut().map(|cache| &mut **cache),
        residency,
        &full_slots,
    );
    if full_slots_ready.len() < full_slots.len() {
        bevy::log::debug!(
            "[gpu_mpm] world edit bootstrap pending generated chunks: ready={} requested={}",
            full_slots_ready.len(),
            full_slots.len()
        );
    }
    upload.terrain_slot_ids.clear();
    upload.terrain_cell_solid_slot_diffs.clear();
    upload.upload_terrain_cell_slot_diffs = false;
    enqueue_terrain_slot_updates(residency, terrain, &full_slots_ready, upload);
    upload.last_uploaded_terrain_version = Some(terrain.terrain_version());
    residency.active_resident_chunk_count = residency
        .slot_state
        .iter()
        .enumerate()
        .filter(|(idx, slot)| {
            residency.slot_allocated.get(*idx).copied().unwrap_or(false) && slot.resident
        })
        .count() as u32;
    residency.active_tile_capacity = residency
        .resident_chunk_count
        .saturating_mul(MPM_ACTIVE_TILES_PER_SLOT);
    residency.chunk_sdf_samples = residency
        .resident_chunk_count
        .saturating_mul(MPM_CHUNK_NODES_PER_SLOT);
    residency.invalid_slot_access_count = 0;
    true
}

pub fn apply_statistics_readback(
    control: Res<MpmGpuControl>,
    stats_status: Res<MpmStatisticsStatus>,
    stats_readback: Res<GpuStatisticsReadbackResult>,
    mut stats_snapshot: ResMut<MpmStatisticsSnapshot>,
) {
    if control.init_only || !control.readback_enabled {
        return;
    }
    if !stats_status.any_enabled() {
        return;
    }
    let Some(scalars) = stats_readback.take() else {
        return;
    };
    *stats_snapshot = MpmStatisticsSnapshot::from_gpu_scalars(scalars);
}

pub fn consume_mover_apply_ack(
    control: Res<MpmGpuControl>,
    mover_apply_ack: Res<GpuMoverApplyAck>,
    mut residency: ResMut<MpmChunkResidencyState>,
    mut upload: ResMut<MpmGpuUploadRequest>,
) {
    if control.init_only || !control.readback_enabled {
        return;
    }
    let ack = mover_apply_ack.value();
    if ack == residency.observed_mover_apply_ack {
        return;
    }
    residency.observed_mover_apply_ack = ack;
    residency.pending_mover_apply = false;
    upload.mover_results.clear();
    upload.upload_mover_results = true;
}

pub fn apply_chunk_event_readback(
    control: Res<MpmGpuControl>,
    chunk_event_readback: Res<GpuChunkEventReadbackResult>,
    mut terrain: ResMut<TerrainWorld>,
    mut generated_chunk_cache: Option<ResMut<TerrainGeneratedChunkCache>>,
    mut residency: ResMut<MpmChunkResidencyState>,
    mut upload: ResMut<MpmGpuUploadRequest>,
) {
    if control.init_only || !control.readback_enabled {
        return;
    }
    let Some(events) = chunk_event_readback.take() else {
        return;
    };
    if !residency.initialized
        || residency.slot_state.is_empty()
        || residency.chunk_meta_cache.is_empty()
    {
        return;
    }
    if events.is_empty() {
        return;
    }

    let slot_count = residency.slot_state.len();
    let mut dirty = vec![false; slot_count];
    let mut snapshot_seen = vec![false; slot_count];
    let mut occupancy_changed = false;
    let mut frontier_requests = 0u32;
    let mut newly_allocated_slots: Vec<u32> = Vec::new();

    for event in events {
        let slot_id = event.slot_id as usize;
        if slot_id >= slot_count {
            continue;
        }
        dirty[slot_id] = true;
        let slot = &mut residency.slot_state[slot_id];
        match event.event_kind {
            CHUNK_EVENT_KIND_SLOT_SNAPSHOT => {
                snapshot_seen[slot_id] = true;
                let next_occupied = event._pad_a != 0;
                let prev_occupied = slot.occupied_particle_count > 0;
                if prev_occupied != next_occupied {
                    occupancy_changed = true;
                }
                slot.occupied_particle_count = u32::from(next_occupied);
                if event._pad_b != 0 {
                    frontier_requests = frontier_requests.saturating_add(1);
                }
            }
            CHUNK_EVENT_KIND_NEWLY_OCCUPIED => {
                if !snapshot_seen[slot_id] && slot.occupied_particle_count == 0 {
                    slot.occupied_particle_count = 1;
                    occupancy_changed = true;
                }
            }
            CHUNK_EVENT_KIND_NEWLY_EMPTY => {
                if !snapshot_seen[slot_id] && slot.occupied_particle_count > 0 {
                    slot.occupied_particle_count = 0;
                    occupancy_changed = true;
                }
            }
            CHUNK_EVENT_KIND_FRONTIER_REQUEST => {
                if !snapshot_seen[slot_id] {
                    frontier_requests = frontier_requests.saturating_add(1);
                }
            }
            _ => {}
        }
    }

    let mut missing_halo_slots = 0u32;
    if occupancy_changed || frontier_requests > 0 {
        match ensure_halo_slots_for_occupied(&mut residency, &mut dirty) {
            Ok(allocated) => {
                if !allocated.is_empty() {
                    newly_allocated_slots.extend(allocated);
                }
            }
            Err(_) => {
                missing_halo_slots = missing_halo_slots.saturating_add(1);
            }
        }

        let slot_to_chunk = residency.slot_to_chunk.clone();
        let chunk_to_slot = residency.chunk_to_slot.clone();
        let slot_allocated = residency.slot_allocated.clone();
        missing_halo_slots = missing_halo_slots.saturating_add(rebuild_halo_ref_counts(
            &mut residency.slot_state,
            &slot_to_chunk,
            &slot_allocated,
            &chunk_to_slot,
        ));
    }
    if missing_halo_slots > 0 {
        residency.runtime_rebuild_reason_halo_update_fail = residency
            .runtime_rebuild_reason_halo_update_fail
            .saturating_add(missing_halo_slots as u64);
        bevy::log::warn!(
            "[gpu_mpm] frontier halo reached slot-window edge: missing_neighbors={}",
            missing_halo_slots
        );
    }

    let changed_resident_slots = recompute_resident_flags(&mut residency.slot_state);
    let newly_resident_slots: Vec<u32> = changed_resident_slots
        .iter()
        .copied()
        .filter(|&slot_id| {
            residency
                .slot_state
                .get(slot_id as usize)
                .map(|slot| slot.resident)
                .unwrap_or(false)
        })
        .collect();
    if !changed_resident_slots.is_empty() {
        let neighbor_dirty = collect_neighbor_dirty_slots(
            &changed_resident_slots,
            &residency.slot_to_chunk,
            &residency.chunk_to_slot,
        );
        for slot_id in neighbor_dirty {
            if (slot_id as usize) < dirty.len() {
                dirty[slot_id as usize] = true;
            }
        }
    }
    residency.active_resident_chunk_count = residency
        .slot_state
        .iter()
        .enumerate()
        .filter(|(idx, slot)| {
            residency.slot_allocated.get(*idx).copied().unwrap_or(false) && slot.resident
        })
        .count() as u32;

    let mut new_chunk_meta_diffs = Vec::new();
    for (slot_id, is_dirty) in dirty.into_iter().enumerate() {
        if !is_dirty {
            continue;
        }
        let meta = build_chunk_meta_for_slot(
            slot_id as u32,
            &residency.slot_to_chunk,
            &residency.chunk_to_slot,
            &residency.slot_state,
            &residency.slot_allocated,
        );
        if residency.chunk_meta_cache[slot_id] != meta {
            residency.chunk_meta_cache[slot_id] = meta;
            new_chunk_meta_diffs.push((slot_id as u32, meta));
        }
    }
    stage_chunk_meta_uploads(&mut upload, &residency, new_chunk_meta_diffs);

    if frontier_requests > 0 {
        bevy::log::debug!(
            "[gpu_mpm] frontier requests handled via halo refresh: {}",
            frontier_requests
        );
    }

    let mut terrain_refresh_seed_slots = newly_allocated_slots;
    terrain_refresh_seed_slots.extend_from_slice(&newly_resident_slots);
    terrain_refresh_seed_slots.sort_unstable();
    terrain_refresh_seed_slots.dedup();
    if !terrain_refresh_seed_slots.is_empty() {
        let update_slots = collect_slots_with_neighbors(&residency, &terrain_refresh_seed_slots);
        let ready_slots = ensure_chunks_available_for_slots(
            &mut terrain,
            generated_chunk_cache.as_mut().map(|cache| &mut **cache),
            &residency,
            &update_slots,
        );
        if ready_slots.len() < update_slots.len() {
            bevy::log::debug!(
                "[gpu_mpm] halo terrain updates pending generated chunks: ready={} requested={}",
                ready_slots.len(),
                update_slots.len()
            );
        }
        enqueue_terrain_slot_updates(&residency, &terrain, &ready_slots, &mut upload);
        residency.chunk_sdf_samples = residency
            .resident_chunk_count
            .saturating_mul(MPM_CHUNK_NODES_PER_SLOT);
    }
}

pub fn apply_mover_readback(
    control: Res<MpmGpuControl>,
    mover_readback: Res<GpuMoverReadbackResult>,
    mut terrain: ResMut<TerrainWorld>,
    mut generated_chunk_cache: Option<ResMut<TerrainGeneratedChunkCache>>,
    mut residency: ResMut<MpmChunkResidencyState>,
    mut upload: ResMut<MpmGpuUploadRequest>,
) {
    if control.init_only || !control.readback_enabled {
        return;
    }
    let Some(movers) = mover_readback.take() else {
        return;
    };
    if !residency.initialized {
        return;
    }

    upload.mover_results.clear();

    let mut rebuild_required = false;
    let mut reason_invalid_old_slot = 0u64;
    let mut reason_new_chunk_oob = 0u32;
    let slot_span_before = residency.resident_chunk_count;
    let mut newly_allocated_slots: Vec<u32> = Vec::new();

    let mover_count = movers.len();

    // Mover readback is now exceptional-path only:
    // unresolved (frontier / long-jump / invalid-home) particles are CPU-resolved here.
    for mover in movers {
        if mover.old_home_slot_id == INVALID_CHUNK_SLOT_ID {
            reason_invalid_old_slot = reason_invalid_old_slot.saturating_add(1);
        }
        let new_chunk = IVec2::new(mover.new_chunk_coord_x, mover.new_chunk_coord_y);
        let new_slot = if let Some(&slot) = residency.chunk_to_slot.get(&new_chunk) {
            slot
        } else if let Some(slot) = allocate_chunk_slot(&mut residency, new_chunk) {
            newly_allocated_slots.push(slot);
            slot
        } else {
            rebuild_required = true;
            reason_new_chunk_oob = reason_new_chunk_oob.saturating_add(1);
            continue;
        };

        upload.mover_results.push(GpuMoverResult {
            particle_id: mover.particle_id,
            new_home_slot_id: new_slot,
            _pad_a: 0,
            _pad_b: 0,
        });
    }

    if rebuild_required {
        residency.runtime_rebuild_count = residency.runtime_rebuild_count.saturating_add(1);
        residency.runtime_rebuild_reason_invalid_old_slot = residency
            .runtime_rebuild_reason_invalid_old_slot
            .saturating_add(reason_invalid_old_slot);
        residency.runtime_rebuild_reason_new_chunk_oob = residency
            .runtime_rebuild_reason_new_chunk_oob
            .saturating_add(reason_new_chunk_oob as u64);
        upload.mover_results.clear();
        upload.upload_mover_results = false;
        bevy::log::warn!(
            "[gpu_mpm] runtime rebuild requested: reason_new_chunk_oob={} reason_invalid_old_slot={} movers={}",
            reason_new_chunk_oob,
            reason_invalid_old_slot,
            mover_count
        );
        residency.rebuild_requested = true;
        return;
    }

    // Exceptional resolution is best-effort and non-blocking.
    upload.upload_mover_results = !upload.mover_results.is_empty();

    if residency.resident_chunk_count != slot_span_before {
        let mut dirty = vec![true; residency.slot_state.len()];
        let mut new_chunk_meta_diffs = Vec::new();
        if let Ok(allocated) = ensure_halo_slots_for_occupied(&mut residency, &mut dirty) {
            newly_allocated_slots.extend(allocated);
        }
        let slot_to_chunk = residency.slot_to_chunk.clone();
        let chunk_to_slot = residency.chunk_to_slot.clone();
        let slot_allocated = residency.slot_allocated.clone();
        let _ = rebuild_halo_ref_counts(
            &mut residency.slot_state,
            &slot_to_chunk,
            &slot_allocated,
            &chunk_to_slot,
        );
        let changed_resident_slots = recompute_resident_flags(&mut residency.slot_state);
        let newly_resident_slots: Vec<u32> = changed_resident_slots
            .iter()
            .copied()
            .filter(|&slot_id| {
                residency
                    .slot_state
                    .get(slot_id as usize)
                    .map(|slot| slot.resident)
                    .unwrap_or(false)
            })
            .collect();
        for (slot_id, is_dirty) in dirty.into_iter().enumerate() {
            if !is_dirty {
                continue;
            }
            let meta = build_chunk_meta_for_slot(
                slot_id as u32,
                &residency.slot_to_chunk,
                &residency.chunk_to_slot,
                &residency.slot_state,
                &residency.slot_allocated,
            );
            if residency.chunk_meta_cache[slot_id] != meta {
                residency.chunk_meta_cache[slot_id] = meta;
                new_chunk_meta_diffs.push((slot_id as u32, meta));
            }
        }
        stage_chunk_meta_uploads(&mut upload, &residency, new_chunk_meta_diffs);
        let mut terrain_refresh_seed_slots = newly_allocated_slots;
        terrain_refresh_seed_slots.extend_from_slice(&newly_resident_slots);
        terrain_refresh_seed_slots.sort_unstable();
        terrain_refresh_seed_slots.dedup();
        if !terrain_refresh_seed_slots.is_empty() {
            let update_slots =
                collect_slots_with_neighbors(&residency, &terrain_refresh_seed_slots);
            let ready_slots = ensure_chunks_available_for_slots(
                &mut terrain,
                generated_chunk_cache.as_mut().map(|cache| &mut **cache),
                &residency,
                &update_slots,
            );
            if ready_slots.len() < update_slots.len() {
                bevy::log::debug!(
                    "[gpu_mpm] exceptional terrain updates pending generated chunks: ready={} requested={}",
                    ready_slots.len(),
                    update_slots.len()
                );
            }
            enqueue_terrain_slot_updates(&residency, &terrain, &ready_slots, &mut upload);
            residency.chunk_sdf_samples = residency
                .resident_chunk_count
                .saturating_mul(MPM_CHUNK_NODES_PER_SLOT);
        }
        residency.active_tile_capacity = residency
            .resident_chunk_count
            .saturating_mul(MPM_ACTIVE_TILES_PER_SLOT);
        if residency.active_tile_capacity == 0 {
            residency.active_tile_count = 0;
            residency.inactive_skip_rate = 0.0;
        }
    }
}

fn append_particles_in_cell(
    upload_particles: &mut Vec<GpuParticle>,
    cell: IVec2,
    material: ParticleMaterial,
    count_per_cell: u32,
    rho0: f32,
) {
    let Some(phase_id) = mpm_phase_id_for_particle(material) else {
        return;
    };
    let props = particle_properties(material);
    let axis = particle_grid_axis(count_per_cell);
    let axis_f = axis as f32;
    let spacing = CELL_SIZE_M / axis_f.max(1.0);
    let cell_min = cell_to_world_center(cell) - Vec2::splat(CELL_SIZE_M * 0.5);

    let mut spawned = 0u32;
    'grid: for y in 0..axis {
        for x in 0..axis {
            if spawned >= count_per_cell {
                break 'grid;
            }
            let offset = Vec2::new((x as f32 + 0.5) * spacing, (y as f32 + 0.5) * spacing);
            let p = cell_min + offset;
            upload_particles.push(GpuParticle::from_cpu(
                p,
                Vec2::ZERO,
                props.mass,
                props.mass.max(0.0) / rho0,
                Mat2::IDENTITY,
                Mat2::ZERO,
                0.0,
                phase_id,
            ));
            spawned += 1;
        }
    }
}

fn particle_grid_axis(count: u32) -> u32 {
    (count as f32).sqrt().ceil() as u32
}

#[derive(Debug)]
struct StaticChunkUploadBuild {
    chunk_origin: IVec2,
    chunk_dims: UVec2,
    grid_layout: GpuGridLayout,
    resident_chunks: Vec<IVec2>,
}

fn chunk_coord_from_world_pos(world_pos: Vec2) -> IVec2 {
    let cell = world_to_cell(world_pos);
    IVec2::new(
        cell.x.div_euclid(CHUNK_SIZE_I32),
        cell.y.div_euclid(CHUNK_SIZE_I32),
    )
}

fn slot_id_from_chunk_coord(chunk: IVec2, chunk_to_slot: &HashMap<IVec2, u32>) -> Option<u32> {
    chunk_to_slot.get(&chunk).copied()
}

fn assign_home_slot_ids(particles: &mut [GpuParticle], chunk_to_slot: &HashMap<IVec2, u32>) -> u32 {
    let mut unresolved = 0u32;
    for particle in particles.iter_mut() {
        let chunk = chunk_coord_from_world_pos(Vec2::from_array(particle.x));
        if let Some(slot_id) = slot_id_from_chunk_coord(chunk, chunk_to_slot) {
            particle.home_chunk_slot_id = slot_id;
        } else {
            unresolved = unresolved.saturating_add(1);
            particle.home_chunk_slot_id = INVALID_CHUNK_SLOT_ID;
        }
    }
    unresolved
}

fn slot_coord_from_slot_id(slot_id: u32, slot_to_chunk: &[IVec2]) -> Option<IVec2> {
    slot_to_chunk.get(slot_id as usize).copied()
}

fn active_tile_mask_for_particles(
    particles: &[GpuParticle],
    residency: &MpmChunkResidencyState,
) -> Vec<u32> {
    let mut masks = vec![0u32; residency.slot_to_chunk.len()];
    let cdim_i = MPM_CHUNK_NODE_DIM as i32;
    let tiles_per_axis = ((MPM_CHUNK_NODE_DIM + ACTIVE_TILE_NODE_DIM_I32 as u32 - 1)
        / ACTIVE_TILE_NODE_DIM_I32 as u32) as i32;
    if cdim_i <= 0 || tiles_per_axis <= 0 {
        return masks;
    }

    for particle in particles {
        let home_slot = particle.home_chunk_slot_id as usize;
        if particle.home_chunk_slot_id == INVALID_CHUNK_SLOT_ID
            || home_slot >= residency.slot_to_chunk.len()
            || !residency
                .slot_allocated
                .get(home_slot)
                .copied()
                .unwrap_or(false)
            || !residency
                .slot_state
                .get(home_slot)
                .map(|slot| slot.resident)
                .unwrap_or(false)
        {
            continue;
        }

        let home_chunk = residency.slot_to_chunk[home_slot];
        let home_node_origin = home_chunk * cdim_i;
        let grid_pos = Vec2::from_array(particle.x) / MPM_NODE_SPACING_M.max(1.0e-6);
        let base = IVec2::new(
            (grid_pos.x - 0.5).floor() as i32,
            (grid_pos.y - 0.5).floor() as i32,
        );

        for oy in 0..3 {
            for ox in 0..3 {
                let node = base + IVec2::new(ox, oy);
                let local = node - home_node_origin;
                let mut delta_chunk = IVec2::ZERO;
                if local.x < 0 {
                    delta_chunk.x = -1;
                } else if local.x >= cdim_i {
                    delta_chunk.x = 1;
                }
                if local.y < 0 {
                    delta_chunk.y = -1;
                } else if local.y >= cdim_i {
                    delta_chunk.y = 1;
                }
                if delta_chunk.x.abs() > 1 || delta_chunk.y.abs() > 1 {
                    continue;
                }

                let slot_id = if delta_chunk == IVec2::ZERO {
                    home_slot as u32
                } else {
                    let neighbor_chunk = home_chunk + delta_chunk;
                    let Some(slot_id) =
                        slot_id_from_chunk_coord(neighbor_chunk, &residency.chunk_to_slot)
                    else {
                        continue;
                    };
                    if !residency
                        .slot_allocated
                        .get(slot_id as usize)
                        .copied()
                        .unwrap_or(false)
                        || !residency
                            .slot_state
                            .get(slot_id as usize)
                            .map(|slot| slot.resident)
                            .unwrap_or(false)
                    {
                        continue;
                    }
                    slot_id
                };

                let local_x = local.x - delta_chunk.x * cdim_i;
                let local_y = local.y - delta_chunk.y * cdim_i;
                if local_x < 0 || local_y < 0 || local_x >= cdim_i || local_y >= cdim_i {
                    continue;
                }

                let tile_x = local_x / ACTIVE_TILE_NODE_DIM_I32;
                let tile_y = local_y / ACTIVE_TILE_NODE_DIM_I32;
                let bit_idx = tile_y * tiles_per_axis + tile_x;
                if !(0..32).contains(&bit_idx) {
                    continue;
                }
                masks[slot_id as usize] |= 1u32 << bit_idx;
            }
        }
    }

    masks
}

fn update_active_tile_stats(residency: &mut MpmChunkResidencyState, particles: &[GpuParticle]) {
    let masks = active_tile_mask_for_particles(particles, residency);
    let slot_span = residency.resident_chunk_count as usize;
    residency.active_tile_capacity = residency
        .resident_chunk_count
        .saturating_mul(MPM_ACTIVE_TILES_PER_SLOT);
    residency.active_tile_count = masks
        .iter()
        .take(slot_span)
        .map(|mask| mask.count_ones())
        .sum();
    residency.inactive_skip_rate = if residency.active_tile_capacity == 0 {
        0.0
    } else {
        1.0 - residency.active_tile_count as f32 / residency.active_tile_capacity as f32
    };
}

fn rebuild_halo_ref_counts(
    slot_state: &mut [ChunkSlotState],
    slot_to_chunk: &[IVec2],
    slot_allocated: &[bool],
    chunk_to_slot: &HashMap<IVec2, u32>,
) -> u32 {
    for slot in slot_state.iter_mut() {
        slot.halo_ref_count = 0;
    }
    let mut missing_neighbors = 0u32;
    for slot_id in 0..slot_state.len() {
        if !slot_allocated.get(slot_id).copied().unwrap_or(false) {
            continue;
        }
        if slot_state[slot_id].occupied_particle_count == 0 {
            continue;
        }
        let Some(center_chunk) = slot_to_chunk.get(slot_id).copied() else {
            continue;
        };
        for oy in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
            for ox in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
                let neighbor = center_chunk + IVec2::new(ox, oy);
                if let Some(neighbor_slot) = chunk_to_slot.get(&neighbor).copied() {
                    if let Some(neighbor_state) = slot_state.get_mut(neighbor_slot as usize) {
                        neighbor_state.halo_ref_count =
                            neighbor_state.halo_ref_count.saturating_add(1);
                    }
                } else {
                    missing_neighbors = missing_neighbors.saturating_add(1);
                }
            }
        }
    }
    missing_neighbors
}

fn stage_pending_add_target_residency(
    residency: &mut MpmChunkResidencyState,
    target_chunks: &[IVec2],
) -> Vec<u32> {
    if target_chunks.is_empty() || residency.slot_state.is_empty() {
        return Vec::new();
    }

    let mut dirty_slots = HashSet::<u32>::default();
    for &chunk in target_chunks {
        let Some(slot_id) = slot_id_from_chunk_coord(chunk, &residency.chunk_to_slot) else {
            continue;
        };
        let slot_idx = slot_id as usize;
        let Some(slot) = residency.slot_state.get_mut(slot_idx) else {
            continue;
        };
        if slot.occupied_particle_count == 0 {
            // Pending GPU add means this chunk will become occupied this frame.
            slot.occupied_particle_count = 1;
        }
        dirty_slots.insert(slot_id);
    }

    let slot_to_chunk = residency.slot_to_chunk.clone();
    let chunk_to_slot = residency.chunk_to_slot.clone();
    let slot_allocated = residency.slot_allocated.clone();
    let _ = rebuild_halo_ref_counts(
        &mut residency.slot_state,
        &slot_to_chunk,
        &slot_allocated,
        &chunk_to_slot,
    );
    let changed_resident_slots = recompute_resident_flags(&mut residency.slot_state);
    dirty_slots.extend(changed_resident_slots.iter().copied());
    dirty_slots.extend(collect_neighbor_dirty_slots(
        &changed_resident_slots,
        &residency.slot_to_chunk,
        &residency.chunk_to_slot,
    ));

    residency.active_resident_chunk_count = residency
        .slot_state
        .iter()
        .enumerate()
        .filter(|(idx, slot)| {
            residency.slot_allocated.get(*idx).copied().unwrap_or(false) && slot.resident
        })
        .count() as u32;

    let mut ordered = dirty_slots.into_iter().collect::<Vec<_>>();
    ordered.sort_unstable();
    ordered
}

fn recompute_resident_flags(slot_state: &mut [ChunkSlotState]) -> Vec<u32> {
    let mut changed = Vec::new();
    for (slot_id, slot) in slot_state.iter_mut().enumerate() {
        let resident = slot.occupied_particle_count > 0 || slot.halo_ref_count > 0;
        if resident != slot.resident {
            slot.resident = resident;
            changed.push(slot_id as u32);
        }
    }
    changed
}

fn collect_neighbor_dirty_slots(
    changed_slots: &[u32],
    slot_to_chunk: &[IVec2],
    chunk_to_slot: &HashMap<IVec2, u32>,
) -> Vec<u32> {
    let mut dirty = vec![false; slot_to_chunk.len()];
    for &slot_id in changed_slots {
        if slot_id as usize >= dirty.len() {
            continue;
        }
        dirty[slot_id as usize] = true;
        let Some(chunk) = slot_coord_from_slot_id(slot_id, slot_to_chunk) else {
            continue;
        };
        for offset in MOORE_OFFSETS {
            let neighbor = chunk + offset;
            let Some(neighbor_slot) = slot_id_from_chunk_coord(neighbor, chunk_to_slot) else {
                continue;
            };
            if let Some(slot_dirty) = dirty.get_mut(neighbor_slot as usize) {
                *slot_dirty = true;
            }
        }
    }
    dirty
        .iter()
        .enumerate()
        .filter_map(|(slot_id, &is_dirty)| is_dirty.then_some(slot_id as u32))
        .collect()
}

fn build_chunk_meta_for_slot(
    slot_id: u32,
    slot_to_chunk: &[IVec2],
    chunk_to_slot: &HashMap<IVec2, u32>,
    slot_state: &[ChunkSlotState],
    slot_allocated: &[bool],
) -> GpuChunkMeta {
    let mut meta = GpuChunkMeta::default();
    let Some(chunk) = slot_coord_from_slot_id(slot_id, slot_to_chunk) else {
        return meta;
    };
    let occupied_count = slot_state
        .get(slot_id as usize)
        .map(|slot| slot.occupied_particle_count)
        .unwrap_or(0);
    meta.chunk_coord_x = chunk.x;
    meta.chunk_coord_y = chunk.y;
    meta.active_tile_mask = 0;
    meta.particle_count_curr = occupied_count;
    meta.particle_count_next = occupied_count;
    meta.occupied_bit_curr = u32::from(occupied_count > 0);
    let is_allocated = slot_allocated
        .get(slot_id as usize)
        .copied()
        .unwrap_or(false);
    meta.occupied_bit_next = u32::from(
        is_allocated
            && slot_state
                .get(slot_id as usize)
                .map(|slot| slot.resident)
                .unwrap_or(false),
    );

    if !is_allocated {
        return meta;
    }

    for (i, offset) in MOORE_OFFSETS.into_iter().enumerate() {
        let neighbor_coord = chunk + offset;
        let Some(neighbor_slot) = slot_id_from_chunk_coord(neighbor_coord, chunk_to_slot) else {
            meta.neighbor_slot_id[i] = INVALID_CHUNK_SLOT_ID;
            continue;
        };
        meta.neighbor_slot_id[i] = if slot_state
            .get(neighbor_slot as usize)
            .map(|slot| slot.resident)
            .unwrap_or(false)
        {
            neighbor_slot
        } else {
            INVALID_CHUNK_SLOT_ID
        };
    }

    meta
}

fn update_resident_slot_span(residency: &mut MpmChunkResidencyState) {
    let mut max_slot_plus_one = 0u32;
    for (slot_id, allocated) in residency.slot_allocated.iter().enumerate() {
        if *allocated {
            max_slot_plus_one = (slot_id as u32).saturating_add(1);
        }
    }
    residency.resident_chunk_count = max_slot_plus_one;
}

fn refresh_chunk_bounds_from_pool(residency: &mut MpmChunkResidencyState) {
    let mut min_chunk = IVec2::new(i32::MAX, i32::MAX);
    let mut max_chunk = IVec2::new(i32::MIN, i32::MIN);
    let mut found = false;
    for (slot_id, allocated) in residency.slot_allocated.iter().enumerate() {
        if !*allocated {
            continue;
        }
        found = true;
        let chunk = residency.slot_to_chunk[slot_id];
        min_chunk.x = min_chunk.x.min(chunk.x);
        min_chunk.y = min_chunk.y.min(chunk.y);
        max_chunk.x = max_chunk.x.max(chunk.x);
        max_chunk.y = max_chunk.y.max(chunk.y);
    }
    if !found {
        residency.chunk_origin = IVec2::ZERO;
        residency.chunk_dims = UVec2::ZERO;
        residency.grid_layout = GpuGridLayout {
            origin: IVec2::ZERO,
            dims: UVec2::ZERO,
        };
        return;
    }
    residency.chunk_origin = min_chunk;
    residency.chunk_dims = (max_chunk - min_chunk + IVec2::ONE).as_uvec2();
    let cdim_i = MPM_CHUNK_NODE_DIM as i32;
    residency.grid_layout = GpuGridLayout {
        origin: IVec2::new(min_chunk.x * cdim_i, min_chunk.y * cdim_i),
        dims: UVec2::new(
            residency.chunk_dims.x * MPM_CHUNK_NODE_DIM,
            residency.chunk_dims.y * MPM_CHUNK_NODE_DIM,
        ),
    };
}

fn allocate_chunk_slot(residency: &mut MpmChunkResidencyState, chunk: IVec2) -> Option<u32> {
    if let Some(&slot_id) = residency.chunk_to_slot.get(&chunk) {
        return Some(slot_id);
    }
    let slot_id = residency.free_slots.pop()?;
    let slot_idx = slot_id as usize;
    if slot_idx >= residency.slot_allocated.len() || slot_idx >= residency.slot_to_chunk.len() {
        return None;
    }
    residency.slot_allocated[slot_idx] = true;
    residency.slot_to_chunk[slot_idx] = chunk;
    residency.chunk_to_slot.insert(chunk, slot_id);
    if slot_idx < residency.slot_state.len() {
        residency.slot_state[slot_idx] = ChunkSlotState::default();
    }
    update_resident_slot_span(residency);
    refresh_chunk_bounds_from_pool(residency);
    Some(slot_id)
}

#[allow(dead_code)]
fn free_chunk_slot(residency: &mut MpmChunkResidencyState, slot_id: u32) -> bool {
    let slot_idx = slot_id as usize;
    if slot_idx >= residency.slot_allocated.len() || !residency.slot_allocated[slot_idx] {
        return false;
    }
    let chunk = residency.slot_to_chunk[slot_idx];
    if residency.chunk_to_slot.get(&chunk).copied() == Some(slot_id) {
        residency.chunk_to_slot.remove(&chunk);
    }
    residency.slot_allocated[slot_idx] = false;
    if slot_idx < residency.slot_state.len() {
        residency.slot_state[slot_idx] = ChunkSlotState::default();
    }
    if slot_idx < residency.chunk_meta_cache.len() {
        residency.chunk_meta_cache[slot_idx] = GpuChunkMeta::default();
    }
    residency.free_slots.push(slot_id);
    update_resident_slot_span(residency);
    refresh_chunk_bounds_from_pool(residency);
    true
}

fn initialize_slot_pool(residency: &mut MpmChunkResidencyState, resident_chunks: &[IVec2]) -> bool {
    let slot_capacity = MAX_RESIDENT_CHUNK_SLOTS as usize;
    if resident_chunks.len() > slot_capacity {
        return false;
    }

    residency.chunk_to_slot.clear();
    residency.slot_to_chunk = vec![IVec2::ZERO; slot_capacity];
    residency.slot_allocated = vec![false; slot_capacity];
    residency.free_slots.clear();
    residency.slot_state = vec![ChunkSlotState::default(); slot_capacity];
    residency.chunk_meta_cache = vec![GpuChunkMeta::default(); slot_capacity];

    // pop() allocates lower slot ids first.
    for slot_id in (0..MAX_RESIDENT_CHUNK_SLOTS).rev() {
        residency.free_slots.push(slot_id);
    }

    for &chunk in resident_chunks {
        if allocate_chunk_slot(residency, chunk).is_none() {
            return false;
        }
    }
    refresh_chunk_bounds_from_pool(residency);
    true
}

fn ensure_halo_slots_for_occupied(
    residency: &mut MpmChunkResidencyState,
    dirty: &mut [bool],
) -> Result<Vec<u32>, ()> {
    let mut occupied_chunks = Vec::new();
    for slot_id in 0..residency.slot_state.len() {
        if !residency
            .slot_allocated
            .get(slot_id)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        if residency.slot_state[slot_id].occupied_particle_count == 0 {
            continue;
        }
        occupied_chunks.push(residency.slot_to_chunk[slot_id]);
    }

    let mut needed = Vec::new();
    for center in occupied_chunks {
        for oy in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
            for ox in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
                let chunk = center + IVec2::new(ox, oy);
                if !residency.chunk_to_slot.contains_key(&chunk) {
                    needed.push(chunk);
                }
            }
        }
    }
    needed.sort_by_key(|c| (c.y, c.x));
    needed.dedup();

    let mut allocated_slots = Vec::new();
    for chunk in needed {
        let Some(slot_id) = allocate_chunk_slot(residency, chunk) else {
            return Err(());
        };
        allocated_slots.push(slot_id);
        let idx = slot_id as usize;
        if idx < dirty.len() {
            dirty[idx] = true;
        }
    }
    Ok(allocated_slots)
}

#[allow(dead_code)]
fn release_free_slots(residency: &mut MpmChunkResidencyState, dirty: &mut [bool]) -> bool {
    let mut to_free = Vec::new();
    for slot_id in 0..residency.slot_state.len() {
        if !residency
            .slot_allocated
            .get(slot_id)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        let slot = &residency.slot_state[slot_id];
        if slot.occupied_particle_count == 0 && slot.halo_ref_count == 0 {
            to_free.push(slot_id as u32);
        }
    }

    let mut changed = false;
    for slot_id in to_free {
        let slot_idx = slot_id as usize;
        if slot_idx >= residency.slot_to_chunk.len() {
            continue;
        }
        if slot_idx < dirty.len() {
            dirty[slot_idx] = true;
        }
        let chunk = residency.slot_to_chunk[slot_idx];
        for offset in MOORE_OFFSETS {
            let neighbor = chunk + offset;
            if let Some(neighbor_slot) = residency.chunk_to_slot.get(&neighbor).copied() {
                let nidx = neighbor_slot as usize;
                if nidx < dirty.len() {
                    dirty[nidx] = true;
                }
            }
        }
        changed |= free_chunk_slot(residency, slot_id);
    }
    changed
}

fn collect_allocated_slots(residency: &MpmChunkResidencyState) -> Vec<u32> {
    let mut slots = Vec::new();
    for (slot_id, allocated) in residency.slot_allocated.iter().enumerate() {
        if *allocated {
            slots.push(slot_id as u32);
        }
    }
    slots
}

fn collect_slots_with_neighbors(
    residency: &MpmChunkResidencyState,
    seed_slots: &[u32],
) -> Vec<u32> {
    let mut slots = HashSet::<u32>::default();
    for &slot_id in seed_slots {
        let idx = slot_id as usize;
        if !residency.slot_allocated.get(idx).copied().unwrap_or(false) {
            continue;
        }
        slots.insert(slot_id);
        let chunk = residency.slot_to_chunk[idx];
        for oy in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
            for ox in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
                let neighbor = chunk + IVec2::new(ox, oy);
                if let Some(&neighbor_slot) = residency.chunk_to_slot.get(&neighbor) {
                    if residency
                        .slot_allocated
                        .get(neighbor_slot as usize)
                        .copied()
                        .unwrap_or(false)
                    {
                        slots.insert(neighbor_slot);
                    }
                }
            }
        }
    }
    let mut ordered = slots.into_iter().collect::<Vec<_>>();
    ordered.sort_unstable();
    ordered
}

fn collect_slots_for_dirty_chunks(
    residency: &MpmChunkResidencyState,
    dirty_chunks: &[IVec2],
) -> Vec<u32> {
    if dirty_chunks.is_empty() {
        return Vec::new();
    }
    let mut seed_slots = Vec::new();
    for &chunk in dirty_chunks {
        if let Some(&slot_id) = residency.chunk_to_slot.get(&chunk) {
            seed_slots.push(slot_id);
        }
    }
    collect_slots_with_neighbors(residency, &seed_slots)
}

fn ensure_chunks_available_for_slots(
    terrain: &mut TerrainWorld,
    mut generated_chunk_cache: Option<&mut TerrainGeneratedChunkCache>,
    residency: &MpmChunkResidencyState,
    slot_ids: &[u32],
) -> Vec<u32> {
    if slot_ids.is_empty() {
        return Vec::new();
    }
    let mut ready_slots = Vec::with_capacity(slot_ids.len());
    let mut requested_chunks = HashSet::<IVec2>::default();
    for &slot_id in slot_ids {
        let idx = slot_id as usize;
        if !residency.slot_allocated.get(idx).copied().unwrap_or(false) {
            continue;
        }
        let chunk = residency.slot_to_chunk[idx];
        let mut chunk_ready = terrain.chunk(chunk).is_some();
        if !chunk_ready {
            if !terrain.generation_enabled() {
                terrain.ensure_chunk_loaded(chunk);
                chunk_ready = true;
            } else if let Some(cache) = generated_chunk_cache.as_mut() {
                if let Some(material_ids) = cache.material_ids_for_chunk(chunk) {
                    terrain.load_generated_chunk_from_material_ids(chunk, material_ids);
                    chunk_ready = true;
                } else if requested_chunks.insert(chunk) {
                    cache.enqueue_prefetch_square(chunk, CHUNK_HALO_RADIUS);
                }
            }
        }
        if chunk_ready {
            ready_slots.push(slot_id);
        }
    }
    ready_slots.sort_unstable();
    ready_slots.dedup();
    ready_slots
}

fn build_terrain_cells_for_slots(
    residency: &MpmChunkResidencyState,
    terrain: &TerrainWorld,
    slot_ids: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let chunk_node_dim_u = MPM_CHUNK_NODE_DIM as usize;
    let nodes_per_chunk = chunk_node_dim_u * chunk_node_dim_u;
    let mut ordered_slots = Vec::new();
    let mut terrain_cell_solid = Vec::with_capacity(slot_ids.len() * nodes_per_chunk);

    for &slot_id in slot_ids {
        let idx = slot_id as usize;
        if !residency.slot_allocated.get(idx).copied().unwrap_or(false) {
            continue;
        }
        ordered_slots.push(slot_id);
        let chunk = residency.slot_to_chunk[idx];
        let chunk_cell_origin = IVec2::new(chunk.x * CHUNK_SIZE_I32, chunk.y * CHUNK_SIZE_I32);
        for local_y in 0..chunk_node_dim_u {
            for local_x in 0..chunk_node_dim_u {
                let global_cell = chunk_cell_origin
                    + IVec2::new(
                        (local_x / MPM_NODES_PER_CELL as usize) as i32,
                        (local_y / MPM_NODES_PER_CELL as usize) as i32,
                    );
                let solid = matches!(
                    terrain.get_cell_or_generated(global_cell),
                    TerrainCell::Solid { .. }
                );
                terrain_cell_solid.push(u32::from(solid));
            }
        }
    }

    (ordered_slots, terrain_cell_solid)
}

fn enqueue_terrain_slot_updates(
    residency: &MpmChunkResidencyState,
    terrain: &TerrainWorld,
    slot_ids: &[u32],
    upload: &mut MpmGpuUploadRequest,
) {
    let mut merged_slots = upload.terrain_slot_ids.clone();
    merged_slots.extend_from_slice(slot_ids);
    merged_slots.sort_unstable();
    merged_slots.dedup();
    let (ordered_slots, cell_solid) =
        build_terrain_cells_for_slots(residency, terrain, &merged_slots);
    if ordered_slots.is_empty() {
        return;
    }
    upload.upload_terrain_cell_slot_diffs = true;
    upload.terrain_slot_ids = ordered_slots;
    upload.terrain_cell_solid_slot_diffs = cell_solid;
}

fn initialize_chunk_slot_state(residency: &mut MpmChunkResidencyState, particles: &[GpuParticle]) {
    let slot_count = residency.slot_to_chunk.len();
    residency.slot_state = vec![ChunkSlotState::default(); slot_count];
    for particle in particles {
        let chunk = chunk_coord_from_world_pos(Vec2::from_array(particle.x));
        let Some(slot_id) = slot_id_from_chunk_coord(chunk, &residency.chunk_to_slot) else {
            continue;
        };
        let slot = &mut residency.slot_state[slot_id as usize];
        slot.occupied_particle_count = slot.occupied_particle_count.saturating_add(1);
    }
    let _ = rebuild_halo_ref_counts(
        &mut residency.slot_state,
        &residency.slot_to_chunk,
        &residency.slot_allocated,
        &residency.chunk_to_slot,
    );

    let _ = recompute_resident_flags(&mut residency.slot_state);

    residency.chunk_meta_cache = (0..slot_count as u32)
        .map(|slot_id| {
            build_chunk_meta_for_slot(
                slot_id,
                &residency.slot_to_chunk,
                &residency.chunk_to_slot,
                &residency.slot_state,
                &residency.slot_allocated,
            )
        })
        .collect();
}

fn build_static_chunk_upload(
    terrain: &TerrainWorld,
    particles: &[GpuParticle],
) -> Option<StaticChunkUploadBuild> {
    let mut occupied = HashSet::<IVec2>::default();
    if particles.is_empty() {
        for chunk in terrain.loaded_chunk_coords() {
            occupied.insert(chunk);
        }
    } else {
        for particle in particles {
            occupied.insert(chunk_coord_from_world_pos(Vec2::from_array(particle.x)));
        }
    }
    if occupied.is_empty() {
        return None;
    }

    let mut resident = HashSet::<IVec2>::default();
    for &chunk in &occupied {
        for oy in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
            for ox in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
                resident.insert(chunk + IVec2::new(ox, oy));
            }
        }
    }
    if resident.is_empty() {
        return None;
    }

    let mut min_chunk = IVec2::new(i32::MAX, i32::MAX);
    let mut max_chunk = IVec2::new(i32::MIN, i32::MIN);
    for &chunk in &resident {
        min_chunk.x = min_chunk.x.min(chunk.x);
        min_chunk.y = min_chunk.y.min(chunk.y);
        max_chunk.x = max_chunk.x.max(chunk.x);
        max_chunk.y = max_chunk.y.max(chunk.y);
    }
    if min_chunk.x > max_chunk.x || min_chunk.y > max_chunk.y {
        return None;
    }

    let chunk_dims_i = max_chunk - min_chunk + IVec2::ONE;
    if chunk_dims_i.x <= 0 || chunk_dims_i.y <= 0 {
        return None;
    }
    let resident_chunk_count = resident.len() as u32;
    if resident_chunk_count > MAX_RESIDENT_CHUNK_SLOTS {
        bevy::log::warn!(
            "[gpu_mpm] static chunk residency exceeds slot capacity: requested={} limit={}",
            resident_chunk_count,
            MAX_RESIDENT_CHUNK_SLOTS
        );
        return None;
    }
    let chunk_origin = min_chunk;
    let chunk_dims = chunk_dims_i.as_uvec2();

    let chunk_node_dim_i = MPM_CHUNK_NODE_DIM as i32;
    let grid_layout = GpuGridLayout {
        origin: IVec2::new(
            chunk_origin.x * chunk_node_dim_i,
            chunk_origin.y * chunk_node_dim_i,
        ),
        dims: UVec2::new(
            chunk_dims.x * MPM_CHUNK_NODE_DIM,
            chunk_dims.y * MPM_CHUNK_NODE_DIM,
        ),
    };
    let mut resident_chunks = resident.into_iter().collect::<Vec<_>>();
    resident_chunks.sort_by_key(|c| (c.y, c.x));

    Some(StaticChunkUploadBuild {
        chunk_origin,
        chunk_dims,
        grid_layout,
        resident_chunks,
    })
}

#[derive(Clone, Copy, Debug)]
struct GpuDruckerPragerParams {
    lambda: f32,
    mu: f32,
    alpha: f32,
    k: f32,
    hardening: f32,
}

fn drucker_prager_params(
    youngs_modulus: f32,
    poisson_ratio: f32,
    friction_deg: f32,
    cohesion_pa: f32,
    hardening: f32,
) -> GpuDruckerPragerParams {
    let e = youngs_modulus.max(1.0);
    let nu = poisson_ratio.clamp(0.0, 0.45);
    let mu = e / (2.0 * (1.0 + nu).max(1e-6));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)).max(1e-6);

    let phi = friction_deg.to_radians().clamp(0.0, 1.3);
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    let denom = (3.0 - sin_phi).max(1e-4);
    let alpha = (2.0_f32 / 3.0_f32).sqrt() * (2.0 * sin_phi) / denom;
    let k = (6.0 * cohesion_pa.max(0.0) * cos_phi) / (3.0_f32.sqrt() * denom);

    GpuDruckerPragerParams {
        lambda,
        mu,
        alpha,
        k,
        hardening: hardening.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::gpu_mpm::buffers::GpuParticle;
    use crate::physics::gpu_mpm::gpu_resources::MpmGpuUploadRequest;
    use crate::physics::gpu_mpm::phase::MPM_PHASE_ID_WATER;
    use crate::physics::gpu_mpm::readback::{
        GpuParticleReadbackPayload, GpuReadbackResult, GpuStatisticsReadbackResult,
    };
    use crate::physics::material::ParticleMaterial;
    use crate::physics::state::SimulationState;
    use bevy::ecs::message::Messages;

    fn setup_edit_app() -> App {
        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(ActivePhysicsParams::default())
            .insert_resource(TerrainWorld::default())
            .insert_resource(MpmGpuUploadRequest::default())
            .insert_resource(MpmParticleReadbackStatus::default())
            .insert_resource(MpmFullParticleReadbackRequest::default())
            .insert_resource(MpmFullParticleReadbackCache {
                payload: Some(GpuParticleReadbackPayload {
                    particles: Vec::new(),
                    particle_revision: 0,
                }),
            })
            .insert_resource(DeferredGpuWorldEditRequests::default())
            .insert_resource(MpmChunkResidencyState::default())
            .insert_resource(PendingGpuParticleAdds::default())
            .insert_resource(MpmGpuWorldEditAddQueueRequest::default())
            .insert_resource(MpmGpuWorldEditRemoveQueueRequest::default())
            .add_message::<GpuWorldEditRequest>()
            .add_systems(Update, apply_world_edit_requests);
        app
    }

    #[test]
    fn world_edit_add_particles_bootstraps_residency_from_full_readback_without_full_upload() {
        let mut app = setup_edit_app();
        let existing_particle = GpuParticle::from_cpu(
            cell_to_world_center(IVec2::new(-1, 0)),
            Vec2::ZERO,
            1.0,
            1.0,
            Mat2::IDENTITY,
            Mat2::ZERO,
            0.0,
            MPM_PHASE_ID_WATER,
        );
        {
            let world = app.world_mut();
            world.resource_mut::<MpmParticleReadbackStatus>().particle_count = 1;
            world.resource_mut::<MpmParticleReadbackStatus>().particle_revision = 1;
            world.resource_mut::<MpmFullParticleReadbackCache>().payload = Some(
                GpuParticleReadbackPayload {
                    particles: vec![existing_particle],
                    particle_revision: 1,
                },
            );
        }
        let cells = vec![IVec2::new(0, 0), IVec2::new(1, 0)];
        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: cells.clone(),
                command: GpuWorldEditCommand::AddParticles {
                    material: ParticleMaterial::WaterLiquid,
                    count_per_cell: 4,
                },
            });

        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();
        assert!(upload.particles.is_empty());
        assert!(!upload.upload_particles);
        assert!(upload.upload_mover_results);
        assert!(upload.upload_chunks);
        assert!(app
            .world()
            .resource::<MpmGpuWorldEditAddQueueRequest>()
            .ops
            .is_empty());

        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();
        let status = app.world().resource::<MpmParticleReadbackStatus>();
        let cache = app.world().resource::<MpmFullParticleReadbackCache>();
        let residency = app.world().resource::<MpmChunkResidencyState>();
        let add_queue = app.world().resource::<MpmGpuWorldEditAddQueueRequest>();

        assert_eq!(upload.mover_results.len(), 1);
        assert!(cache.payload.is_none());
        assert_eq!(upload.particle_revision, 2);
        assert_eq!(status.particle_count, 9);
        assert_eq!(status.particle_revision, 2);
        assert!(residency.initialized);
        assert_eq!(add_queue.added_particle_count, 8);
        assert_eq!(add_queue.ops.len(), 2);
    }

    #[test]
    fn world_edit_add_particles_queues_gpu_ops_when_chunk_slots_are_resident() {
        let mut app = setup_edit_app();
        let cells = vec![IVec2::new(0, 0), IVec2::new(CHUNK_SIZE_I32, 0)];
        app.world_mut()
            .insert_resource(MpmChunkResidencyState {
                initialized: true,
                chunk_to_slot: HashMap::from([(IVec2::ZERO, 0), (IVec2::new(1, 0), 1)]),
                slot_to_chunk: vec![IVec2::ZERO, IVec2::new(1, 0)],
                slot_allocated: vec![true, true],
                slot_state: vec![ChunkSlotState::default(), ChunkSlotState::default()],
                ..Default::default()
            });
        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: cells.clone(),
                command: GpuWorldEditCommand::AddParticles {
                    material: ParticleMaterial::WaterLiquid,
                    count_per_cell: 4,
                },
            });

        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();
        let status = app.world().resource::<MpmParticleReadbackStatus>();
        let cache = app.world().resource::<MpmFullParticleReadbackCache>();
        let pending = app.world().resource::<PendingGpuParticleAdds>();
        let add_queue = app.world().resource::<MpmGpuWorldEditAddQueueRequest>();

        assert!(upload.particles.is_empty());
        assert!(!upload.upload_particles);
        assert!(cache.payload.is_none());
        assert_eq!(upload.particle_revision, 1);
        assert_eq!(status.particle_count, 8);
        assert_eq!(status.particle_revision, 1);
        assert_eq!(pending.latest_revision(), 1);
        assert_eq!(pending.total_added_particles(), 8);
        assert_eq!(add_queue.particle_revision, 1);
        assert_eq!(add_queue.added_particle_count, 8);
        assert_eq!(add_queue.ops.len(), 2);
        assert_eq!(add_queue.ops[0].slot_id, 0);
        assert_eq!(add_queue.ops[1].slot_id, 1);
        assert_eq!(add_queue.ops[0].count_per_cell, 4);
        assert_eq!(add_queue.ops[1].count_per_cell, 4);
    }

    #[test]
    fn world_edit_add_particles_allocates_missing_chunk_slots_without_full_upload() {
        let mut app = setup_edit_app();
        app.world_mut()
            .resource_mut::<TerrainWorld>()
            .set_generation_enabled(false);
        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: vec![IVec2::new(CHUNK_SIZE_I32 * 2, 0)],
                command: GpuWorldEditCommand::AddParticles {
                    material: ParticleMaterial::WaterLiquid,
                    count_per_cell: 4,
                },
            });

        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();
        let residency = app.world().resource::<MpmChunkResidencyState>();

        assert!(!upload.upload_particles);
        assert!(upload.upload_chunks || upload.upload_chunk_diffs);
        assert!(residency.initialized);
        assert!(app
            .world()
            .resource::<MpmGpuWorldEditAddQueueRequest>()
            .ops
            .is_empty());

        app.update();

        let add_queue = app.world().resource::<MpmGpuWorldEditAddQueueRequest>();
        assert_eq!(add_queue.added_particle_count, 4);
        assert_eq!(add_queue.ops.len(), 1);
    }

    #[test]
    fn world_edit_add_particles_marks_target_halo_slots_resident_before_gpu_apply() {
        let mut app = setup_edit_app();
        app.world_mut()
            .resource_mut::<TerrainWorld>()
            .set_generation_enabled(false);
        let target_cell = IVec2::new(CHUNK_SIZE_I32 * 4, CHUNK_SIZE_I32 * 3);
        let target_chunk = IVec2::new(
            target_cell.x.div_euclid(CHUNK_SIZE_I32),
            target_cell.y.div_euclid(CHUNK_SIZE_I32),
        );
        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: vec![target_cell],
                command: GpuWorldEditCommand::AddParticles {
                    material: ParticleMaterial::WaterLiquid,
                    count_per_cell: 4,
                },
            });

        app.update();

        let residency = app.world().resource::<MpmChunkResidencyState>();
        for oy in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
            for ox in -CHUNK_HALO_RADIUS..=CHUNK_HALO_RADIUS {
                let chunk = target_chunk + IVec2::new(ox, oy);
                let slot_id = residency
                    .chunk_to_slot
                    .get(&chunk)
                    .copied()
                    .expect("halo chunk should be allocated");
                assert!(
                    residency.slot_state[slot_id as usize].resident,
                    "chunk {chunk:?} should be resident before GPU add apply"
                );
            }
        }
    }

    #[test]
    fn world_edit_remove_particles_respects_max_count() {
        let mut app = setup_edit_app();
        let target = IVec2::new(2, 3);
        let mut particles = Vec::new();
        for i in 0..9 {
            particles.push(GpuParticle::from_cpu(
                cell_to_world_center(target) + Vec2::new(i as f32 * 0.001, 0.0),
                Vec2::ZERO,
                1.0,
                1.0,
                Mat2::IDENTITY,
                Mat2::ZERO,
                0.0,
                MPM_PHASE_ID_WATER,
            ));
        }
        {
            let world = app.world_mut();
            world.resource_mut::<MpmFullParticleReadbackCache>().payload = Some(
                GpuParticleReadbackPayload {
                    particles,
                    particle_revision: 1,
                },
            );
            world.resource_mut::<MpmParticleReadbackStatus>().particle_count = 9;
            world.resource_mut::<MpmParticleReadbackStatus>().particle_revision = 1;
        }

        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: vec![target],
                command: GpuWorldEditCommand::RemoveParticles { max_count: 4 },
            });
        app.update();

        let remove_queue = app.world().resource::<MpmGpuWorldEditRemoveQueueRequest>();
        let status = app.world().resource::<MpmParticleReadbackStatus>();

        assert_eq!(remove_queue.removed_particle_count, 4);
        assert_eq!(remove_queue.remove_particle_ids.len(), 4);
        assert_eq!(status.particle_count, 5);
    }

    #[test]
    fn world_edit_add_particles_ignores_stale_cpu_upload_when_bootstrapping_from_readback() {
        let mut app = setup_edit_app();
        let stale_particle = GpuParticle::from_cpu(
            cell_to_world_center(IVec2::new(-CHUNK_SIZE_I32 * 2, 0)),
            Vec2::ZERO,
            1.0,
            1.0,
            Mat2::IDENTITY,
            Mat2::ZERO,
            0.0,
            MPM_PHASE_ID_WATER,
        );
        let latest_particle = GpuParticle::from_cpu(
            cell_to_world_center(IVec2::new(CHUNK_SIZE_I32 * 4, 0)),
            Vec2::ZERO,
            1.0,
            1.0,
            Mat2::IDENTITY,
            Mat2::ZERO,
            0.0,
            MPM_PHASE_ID_WATER,
        );
        {
            let world = app.world_mut();
            world.resource_mut::<MpmGpuUploadRequest>().particles = vec![stale_particle];
            world.resource_mut::<MpmGpuUploadRequest>().particle_revision = 1;
            world.resource_mut::<MpmParticleReadbackStatus>().particle_count = 1;
            world.resource_mut::<MpmParticleReadbackStatus>().particle_revision = 2;
            world.resource_mut::<MpmFullParticleReadbackCache>().payload = Some(
                GpuParticleReadbackPayload {
                    particles: vec![latest_particle],
                    particle_revision: 2,
                },
            );
        }

        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: vec![IVec2::new(7, 5)],
                command: GpuWorldEditCommand::AddParticles {
                    material: ParticleMaterial::WaterLiquid,
                    count_per_cell: 4,
                },
            });

        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();
        assert_eq!(upload.particle_revision, 1);
        assert!(!upload.upload_particles);
        assert!(upload.upload_mover_results);
        assert_eq!(upload.mover_results.len(), 1);
        assert!(app
            .world()
            .resource::<MpmGpuWorldEditAddQueueRequest>()
            .ops
            .is_empty());

        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();
        let residency = app.world().resource::<MpmChunkResidencyState>();
        let add_queue = app.world().resource::<MpmGpuWorldEditAddQueueRequest>();

        assert_eq!(upload.particle_revision, 3);
        assert_eq!(add_queue.added_particle_count, 4);
        assert!(residency.chunk_to_slot.contains_key(&IVec2::new(4, 0)));
        assert!(!residency.chunk_to_slot.contains_key(&IVec2::new(-2, 0)));
    }

    #[test]
    fn world_edit_add_particles_updates_pending_full_upload_over_stale_full_readback_cache() {
        let mut app = setup_edit_app();
        let newer_particle = GpuParticle::from_cpu(
            cell_to_world_center(IVec2::new(9, 1)),
            Vec2::ZERO,
            1.0,
            1.0,
            Mat2::IDENTITY,
            Mat2::ZERO,
            0.0,
            MPM_PHASE_ID_WATER,
        );
        let stale_particle = GpuParticle::from_cpu(
            cell_to_world_center(IVec2::new(1, 1)),
            Vec2::ZERO,
            1.0,
            1.0,
            Mat2::IDENTITY,
            Mat2::ZERO,
            0.0,
            MPM_PHASE_ID_WATER,
        );
        {
            let world = app.world_mut();
            world.resource_mut::<MpmGpuUploadRequest>().particles = vec![newer_particle];
            world.resource_mut::<MpmGpuUploadRequest>().particle_revision = 5;
            world.resource_mut::<MpmGpuUploadRequest>().upload_particles = true;
            world.resource_mut::<MpmParticleReadbackStatus>().particle_count = 1;
            world.resource_mut::<MpmParticleReadbackStatus>().particle_revision = 4;
            world.resource_mut::<MpmFullParticleReadbackCache>().payload = Some(
                GpuParticleReadbackPayload {
                    particles: vec![stale_particle],
                    particle_revision: 4,
                },
            );
        }

        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: vec![IVec2::new(10, 1)],
                command: GpuWorldEditCommand::AddParticles {
                    material: ParticleMaterial::WaterLiquid,
                    count_per_cell: 4,
                },
            });

        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();
        let add_queue = app.world().resource::<MpmGpuWorldEditAddQueueRequest>();
        let cache = app.world().resource::<MpmFullParticleReadbackCache>();
        let payload = cache.payload.as_ref().expect("pending full upload cache");
        assert_eq!(upload.particle_revision, 6);
        assert!(upload.upload_particles);
        assert_eq!(add_queue.added_particle_count, 0);
        assert_eq!(upload.particles.len(), 5);
        assert_eq!(payload.particle_revision, 6);
        assert_eq!(payload.particles.len(), 5);
        assert!(
            upload
                .particles
                .iter()
                .any(|particle| world_to_cell(Vec2::from_array(particle.x)) == IVec2::new(9, 1))
        );
        assert!(
            upload
                .particles
                .iter()
                .all(|particle| world_to_cell(Vec2::from_array(particle.x)) != IVec2::new(1, 1))
        );
    }

    #[test]
    fn apply_statistics_readback_updates_snapshot() {
        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(MpmStatisticsStatus {
                total_particles: true,
                phase_counts: true,
                max_speed: true,
                penetration: false,
                tracked_summary: false,
                water_surface_p95: false,
                granular_repose: false,
                material_interaction: false,
                grid_density: false,
                tracked_phase_id: 0,
                tracked_fallback_to_all: true,
                repose_phase_id: 1,
                interaction_primary_phase_id: 2,
                interaction_secondary_phase_id: 0,
            })
            .insert_resource(GpuStatisticsReadbackResult::default())
            .insert_resource(MpmStatisticsSnapshot::default())
            .add_systems(Update, apply_statistics_readback);

        let mut lanes = [0u32; buffers::GPU_STATS_SCALAR_LANES];
        lanes[buffers::GPU_STATS_LANE_TOTAL_PARTICLES] = 99;
        lanes[buffers::GPU_STATS_LANE_PHASE_WATER] = 11;
        lanes[buffers::GPU_STATS_LANE_PHASE_GRANULAR_SOIL] = 22;
        lanes[buffers::GPU_STATS_LANE_PHASE_GRANULAR_SAND] = 33;
        lanes[buffers::GPU_STATS_LANE_PHASE_UNKNOWN] = 44;
        lanes[buffers::GPU_STATS_LANE_MAX_SPEED_BITS] = 1.5f32.to_bits();
        app.world()
            .resource::<GpuStatisticsReadbackResult>()
            .store(GpuStatisticsScalars { lanes });

        app.update();

        let snapshot = app.world().resource::<MpmStatisticsSnapshot>();
        assert_eq!(snapshot.total_particles, 99);
        assert_eq!(snapshot.water_liquid, 11);
        assert_eq!(snapshot.soil_granular, 22);
        assert_eq!(snapshot.sand_granular, 33);
        assert_eq!(snapshot.unknown, 44);
        assert_eq!(snapshot.total(), 99);
        assert!((snapshot.max_speed_mps - 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn prepare_particle_upload_prefers_explicit_clear_over_stale_snapshot() {
        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(MpmGpuUploadRequest {
                upload_particles: true,
                particle_revision: 8,
                ..Default::default()
            })
            .insert_resource(MpmParticleReadbackStatus {
                particle_count: 1,
                particle_revision: 7,
            })
            .insert_resource(PendingGpuParticleAdds::default())
            .insert_resource(SimulationState::default())
            .add_systems(Update, prepare_particle_upload);

        app.update();

        let sim_state = app.world().resource::<SimulationState>();
        let upload = app.world().resource::<MpmGpuUploadRequest>();
        assert!(!sim_state.gpu_mpm_active);
        assert!(upload.upload_particles_frame);
        assert!(!upload.upload_particles);
    }

    #[test]
    fn prepare_gpu_params_uses_effective_particle_count_without_double_counting_pending_adds() {
        let mut pending_gpu_adds = PendingGpuParticleAdds::default();
        pending_gpu_adds.push_batch(PendingGpuParticleAddBatch {
            particle_revision: 9,
            added_particle_count: 6,
            expected_total_particle_count: 16,
            ops: vec![PendingGpuParticleAddOp {
                cell: IVec2::new(0, 0),
                material: ParticleMaterial::WaterLiquid,
                count_per_cell: 6,
            }],
        });

        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(ActivePhysicsParams::default())
            .insert_resource(MpmGpuUploadRequest {
                particles: vec![
                    GpuParticle::from_cpu(
                        Vec2::new(1.0, 2.0),
                        Vec2::ZERO,
                        1.0,
                        1.0,
                        Mat2::IDENTITY,
                        Mat2::ZERO,
                        0.0,
                        MPM_PHASE_ID_WATER,
                    );
                    3
                ],
                upload_particles_frame: false,
                ..Default::default()
            })
            .insert_resource(MpmParticleReadbackStatus {
                particle_count: 16,
                particle_revision: 8,
            })
            .insert_resource(pending_gpu_adds)
            .insert_resource(MpmChunkResidencyState::default())
            .insert_resource(MpmStatisticsStatus::default())
            .insert_resource(MpmGpuParamsRequest::default())
            .add_systems(Update, prepare_gpu_params);

        app.update();

        let params = app.world().resource::<MpmGpuParamsRequest>();
        assert_eq!(params.params.particle_count, 16);
    }

    #[test]
    fn prepare_terrain_upload_preserves_staged_world_edit_uploads() {
        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(SimulationState {
                mpm_enabled: true,
                ..Default::default()
            })
            .insert_resource(TerrainWorld::default())
            .insert_resource(MpmParticleReadbackStatus::default())
            .insert_resource(MpmFullParticleReadbackRequest::default())
            .insert_resource(MpmFullParticleReadbackCache::default())
            .insert_resource(MpmChunkResidencyState {
                initialized: true,
                resident_chunk_count: 1,
                ..Default::default()
            })
            .insert_resource(MpmGpuUploadRequest {
                upload_chunks: true,
                chunk_meta: vec![GpuChunkMeta::default()],
                upload_terrain_cell_slot_diffs: true,
                terrain_slot_ids: vec![0],
                terrain_cell_solid_slot_diffs: vec![0; MPM_CHUNK_NODES_PER_SLOT as usize],
                last_uploaded_terrain_version: Some(0),
                ..Default::default()
            })
            .add_systems(Update, prepare_terrain_upload);

        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();
        assert!(upload.upload_chunks);
        assert!(upload.upload_terrain_cell_slot_diffs);
        assert_eq!(upload.chunk_meta.len(), 1);
        assert_eq!(upload.terrain_slot_ids, vec![0]);
        assert_eq!(
            upload.terrain_cell_solid_slot_diffs.len(),
            MPM_CHUNK_NODES_PER_SLOT as usize
        );
    }

    #[test]
    fn apply_gpu_readback_ignores_stale_when_gpu_path_inactive() {
        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(GpuReadbackResult::default())
            .insert_resource(MpmFullParticleReadbackRequest::default())
            .insert_resource(MpmFullParticleReadbackCache::default())
            .add_systems(Update, apply_gpu_readback);

        app.world()
            .resource::<GpuReadbackResult>()
            .store(GpuParticleReadbackPayload {
                particles: vec![GpuParticle::from_cpu(
                    Vec2::new(3.0, 4.0),
                    Vec2::ZERO,
                    1.0,
                    1.0,
                    Mat2::IDENTITY,
                    Mat2::ZERO,
                    0.0,
                    MPM_PHASE_ID_WATER,
                )],
                particle_revision: 1,
            });

        app.update();

        let cache = app.world().resource::<MpmFullParticleReadbackCache>();
        assert!(cache.payload.is_none());
    }

    #[test]
    fn apply_gpu_readback_ignores_older_particle_revision() {
        let stale_particle = GpuParticle::from_cpu(
            cell_to_world_center(IVec2::new(2, 2)),
            Vec2::ZERO,
            1.0,
            1.0,
            Mat2::IDENTITY,
            Mat2::ZERO,
            0.0,
            MPM_PHASE_ID_WATER,
        );

        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(GpuReadbackResult::default())
            .insert_resource(MpmFullParticleReadbackRequest {
                requested: true,
                min_particle_revision: 5,
            })
            .insert_resource(MpmFullParticleReadbackCache::default())
            .add_systems(Update, apply_gpu_readback);

        app.world()
            .resource::<GpuReadbackResult>()
            .store(GpuParticleReadbackPayload {
                particles: vec![stale_particle],
                particle_revision: 4,
            });

        app.update();

        let request = app.world().resource::<MpmFullParticleReadbackRequest>();
        let cache = app.world().resource::<MpmFullParticleReadbackCache>();
        assert!(request.requested);
        assert_eq!(request.min_particle_revision, 5);
        assert!(cache.payload.is_none());
    }

    #[test]
    fn apply_gpu_readback_caches_requested_payload_and_clears_request() {
        let current_particle = GpuParticle::from_cpu(
            cell_to_world_center(IVec2::new(6, 2)),
            Vec2::ZERO,
            1.0,
            1.0,
            Mat2::IDENTITY,
            Mat2::ZERO,
            0.0,
            MPM_PHASE_ID_WATER,
        );

        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(GpuReadbackResult::default())
            .insert_resource(MpmFullParticleReadbackRequest {
                requested: true,
                min_particle_revision: 6,
            })
            .insert_resource(MpmFullParticleReadbackCache::default())
            .add_systems(Update, apply_gpu_readback);

        app.world()
            .resource::<GpuReadbackResult>()
            .store(GpuParticleReadbackPayload {
                particles: vec![current_particle],
                particle_revision: 6,
            });

        app.update();

        let request = app.world().resource::<MpmFullParticleReadbackRequest>();
        let cache = app.world().resource::<MpmFullParticleReadbackCache>();
        let payload = cache.payload.as_ref().expect("full payload cached");
        assert!(!request.requested);
        assert_eq!(payload.particle_revision, 6);
        assert_eq!(payload.particles.len(), 1);
        assert_eq!(
            world_to_cell(Vec2::from_array(payload.particles[0].x)),
            IVec2::new(6, 2)
        );
    }

    #[test]
    fn consume_world_edit_add_ack_drops_acked_batches() {
        let mut pending_gpu_adds = PendingGpuParticleAdds::default();
        pending_gpu_adds.push_batch(PendingGpuParticleAddBatch {
            particle_revision: 3,
            added_particle_count: 4,
            expected_total_particle_count: 4,
            ops: vec![PendingGpuParticleAddOp {
                cell: IVec2::new(1, 0),
                material: ParticleMaterial::WaterLiquid,
                count_per_cell: 4,
            }],
        });
        pending_gpu_adds.push_batch(PendingGpuParticleAddBatch {
            particle_revision: 5,
            added_particle_count: 2,
            expected_total_particle_count: 6,
            ops: vec![PendingGpuParticleAddOp {
                cell: IVec2::new(2, 0),
                material: ParticleMaterial::WaterLiquid,
                count_per_cell: 2,
            }],
        });

        let ack = GpuWorldEditAddApplyAck::default();
        ack.signal_revision(3);

        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(ack)
            .insert_resource(pending_gpu_adds)
            .add_systems(Update, consume_world_edit_add_ack);

        app.update();

        let pending = app.world().resource::<PendingGpuParticleAdds>();
        assert_eq!(pending.latest_revision(), 5);
        assert_eq!(pending.total_added_particles(), 2);
    }

    #[test]
    fn world_edit_add_requeues_oldest_pending_batch_without_new_input() {
        let mut app = App::new();
        let mut pending_gpu_adds = PendingGpuParticleAdds::default();
        pending_gpu_adds.push_batch(PendingGpuParticleAddBatch {
            particle_revision: 7,
            added_particle_count: 4,
            expected_total_particle_count: 10,
            ops: vec![PendingGpuParticleAddOp {
                cell: IVec2::new(CHUNK_SIZE_I32 * 3 + 2, 1),
                material: ParticleMaterial::WaterLiquid,
                count_per_cell: 4,
            }],
        });
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(ActivePhysicsParams::default())
            .insert_resource(TerrainWorld::default())
            .insert_resource(MpmGpuUploadRequest::default())
            .insert_resource(MpmParticleReadbackStatus {
                particle_count: 10,
                particle_revision: 7,
            })
            .insert_resource(MpmFullParticleReadbackRequest::default())
            .insert_resource(MpmFullParticleReadbackCache::default())
            .insert_resource(DeferredGpuWorldEditRequests::default())
            .insert_resource(MpmChunkResidencyState {
                initialized: true,
                chunk_to_slot: HashMap::from([(IVec2::new(3, 0), 4)]),
                slot_to_chunk: vec![IVec2::ZERO; 5],
                slot_allocated: vec![false, false, false, false, true],
                slot_state: vec![ChunkSlotState::default(); 5],
                ..Default::default()
            })
            .insert_resource(pending_gpu_adds)
            .insert_resource(MpmGpuWorldEditAddQueueRequest::default())
            .insert_resource(MpmGpuWorldEditRemoveQueueRequest::default())
            .add_message::<GpuWorldEditRequest>()
            .add_systems(Update, apply_world_edit_requests);
        app.world_mut().resource_mut::<MpmChunkResidencyState>().slot_to_chunk[4] =
            IVec2::new(3, 0);

        app.update();

        let add_queue = app.world().resource::<MpmGpuWorldEditAddQueueRequest>();
        assert_eq!(add_queue.particle_revision, 7);
        assert_eq!(add_queue.base_particle_count, 6);
        assert_eq!(add_queue.added_particle_count, 4);
        assert_eq!(add_queue.ops.len(), 1);
        assert_eq!(add_queue.ops[0].slot_id, 4);
        assert_eq!(add_queue.ops[0].local_cell_x, 2);
        assert_eq!(add_queue.ops[0].local_cell_y, 1);
    }

    #[test]
    fn active_tile_mask_marks_neighbor_chunk_tiles() {
        let mut residency = MpmChunkResidencyState {
            initialized: true,
            resident_chunk_count: 2,
            chunk_to_slot: HashMap::from([(IVec2::ZERO, 0), (IVec2::new(1, 0), 1)]),
            slot_to_chunk: vec![IVec2::ZERO, IVec2::new(1, 0)],
            slot_allocated: vec![true, true],
            slot_state: vec![
                ChunkSlotState {
                    resident: true,
                    ..Default::default()
                },
                ChunkSlotState {
                    resident: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let particle = GpuParticle {
            x: [3.9, 1.0],
            home_chunk_slot_id: 0,
            ..GpuParticle::from_cpu(
                Vec2::new(3.9, 1.0),
                Vec2::ZERO,
                1.0,
                1.0,
                Mat2::IDENTITY,
                Mat2::ZERO,
                0.0,
                MPM_PHASE_ID_WATER,
            )
        };

        let masks = active_tile_mask_for_particles(&[particle], &residency);
        assert_eq!(masks[0], (1 << 3) | (1 << 7));
        assert_eq!(masks[1], (1 << 0) | (1 << 4));

        update_active_tile_stats(&mut residency, &[particle]);
        assert_eq!(residency.active_tile_count, 4);
        assert_eq!(residency.active_tile_capacity, 32);
        assert!((residency.inactive_skip_rate - 0.875).abs() < 1.0e-6);
    }
}
