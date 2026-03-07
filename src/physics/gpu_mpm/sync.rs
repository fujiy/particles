// CPU → GPU synchronization systems for MLS-MPM.
//
// Runs in main-world (not render-world): prepares upload requests,
// which are then executed in the render-world extract/prepare phase.

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

use super::buffers::{
    self, CHUNK_EVENT_KIND_FRONTIER_REQUEST, CHUNK_EVENT_KIND_NEWLY_EMPTY,
    CHUNK_EVENT_KIND_NEWLY_OCCUPIED, CHUNK_EVENT_KIND_SLOT_SNAPSHOT, GpuChunkMeta, GpuGridLayout,
    GpuMoverResult, GpuMpmParams, GpuParticle, GpuStatisticsScalars, INVALID_CHUNK_SLOT_ID,
};
use super::gpu_resources::{
    MAX_RESIDENT_CHUNK_SLOTS, MPM_CHUNK_NODE_DIM, MPM_CHUNK_NODES_PER_SLOT, MpmGpuControl,
    MpmGpuParamsRequest, MpmGpuRunRequest, MpmGpuStepClock, MpmGpuUploadRequest, world_grid_layout,
};
use super::phase::mpm_phase_id_for_particle;
use super::readback::{
    GpuChunkEventReadbackResult, GpuMoverApplyAck, GpuMoverReadbackResult, GpuReadbackResult,
    GpuStatisticsReadbackResult,
};
use crate::params::ActivePhysicsParams;
use crate::physics::material::{ParticleMaterial, particle_properties};
use crate::physics::state::{ReplayState, SimulationState};
use crate::physics::world::constants::{CELL_SIZE_M, CHUNK_SIZE_I32};
use crate::physics::world::terrain::{
    TerrainCell, TerrainWorld, cell_to_world_center, world_to_cell,
};
use crate::render::TerrainGeneratedChunkCache;

const CHUNK_HALO_RADIUS: i32 = 1;
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

#[derive(Resource, Debug, Default)]
pub struct MpmReadbackSnapshot {
    pub particles: Vec<GpuParticle>,
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
    mut edit_requests: MessageReader<GpuWorldEditRequest>,
    mut upload: ResMut<MpmGpuUploadRequest>,
    mut readback_snapshot: ResMut<MpmReadbackSnapshot>,
) {
    if control.init_only {
        let _ = edit_requests.read().count();
        return;
    }

    let rho0 = active_params.0.water.rho0.max(1.0e-6);
    let mut saw_request = false;
    let mut changed = false;

    for request in edit_requests.read() {
        saw_request = true;
        if request.cells.is_empty() {
            continue;
        }
        if !readback_snapshot.particles.is_empty() {
            upload.particles = readback_snapshot.particles.clone();
        } else if upload.particles.is_empty() {
            upload.particles = readback_snapshot.particles.clone();
        }
        match &request.command {
            GpuWorldEditCommand::AddParticles {
                material,
                count_per_cell,
            } => {
                if *count_per_cell == 0 || mpm_phase_id_for_particle(*material).is_none() {
                    continue;
                }
                let before = upload.particles.len();
                for &cell in &request.cells {
                    append_particles_in_cell(
                        &mut upload.particles,
                        cell,
                        *material,
                        *count_per_cell,
                        rho0,
                    );
                }
                changed |= upload.particles.len() != before;
            }
            GpuWorldEditCommand::RemoveParticles { max_count } => {
                if *max_count == 0 || upload.particles.is_empty() {
                    continue;
                }
                let removed_mpm_indices = remove_gpu_particles_in_cells(
                    &mut upload.particles,
                    &request.cells,
                    *max_count as usize,
                );
                if removed_mpm_indices.is_empty() {
                    continue;
                }
                changed = true;
            }
        }
    }

    if !saw_request {
        return;
    }

    if !changed {
        return;
    }

    readback_snapshot.particles = upload.particles.clone();
    upload.upload_particles = true;
}

/// System: keep GPU MPM run-state aligned to global simulation mode.
pub fn prepare_particle_upload(
    control: Res<MpmGpuControl>,
    readback_snapshot: Res<MpmReadbackSnapshot>,
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
        !upload.particles.is_empty() || !readback_snapshot.particles.is_empty()
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
    readback_snapshot: Res<MpmReadbackSnapshot>,
    mut residency: ResMut<MpmChunkResidencyState>,
    mut upload: ResMut<MpmGpuUploadRequest>,
) {
    // Default: no chunk/terrain upload this frame unless explicitly requested below.
    upload.upload_chunks = false;
    upload.upload_chunk_diffs = false;
    upload.upload_terrain = false;
    upload.upload_terrain_cell_slot_diffs = false;
    upload.chunk_meta_diffs.clear();
    upload.terrain_slot_ids.clear();
    upload.terrain_cell_solid_slot_diffs.clear();

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
            readback_snapshot.particles.len(),
            residency.runtime_rebuild_reason_new_chunk_oob,
            residency.runtime_rebuild_reason_invalid_old_slot,
            residency.runtime_rebuild_waiting_readback_count
        );
    }

    let particle_source = if upload.upload_particles_frame && !upload.particles.is_empty() {
        upload.particles.as_slice()
    } else {
        readback_snapshot.particles.as_slice()
    };
    if residency.rebuild_requested && !upload.upload_particles_frame && particle_source.is_empty() {
        // Rebuild requested from runtime movers, but fresh GPU readback has not arrived yet.
        // Keep current residency and retry next frame to avoid stale-particle rollback.
        residency.runtime_rebuild_waiting_readback_count = residency
            .runtime_rebuild_waiting_readback_count
            .saturating_add(1);
        if residency.runtime_rebuild_waiting_readback_count == 1 {
            bevy::log::warn!(
                "[gpu_mpm] runtime rebuild pending: waiting for GPU particle readback"
            );
        }
        return;
    }
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
    residency.chunk_sdf_samples = residency
        .resident_chunk_count
        .saturating_mul(MPM_CHUNK_NODES_PER_SLOT);
    residency.invalid_slot_access_count = 0;
    residency.mover_readback_flush_budget = 2;
    residency.rebuild_requested = false;
}

/// System: update MpmGpuParamsRequest from simulation state and PhysicsParams asset.
pub fn prepare_gpu_params(
    control: Res<MpmGpuControl>,
    active_params: Res<ActivePhysicsParams>,
    upload: Res<MpmGpuUploadRequest>,
    readback_snapshot: Res<MpmReadbackSnapshot>,
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
    let h = CELL_SIZE_M;
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
    let particle_count = if upload.upload_particles_frame || !upload.particles.is_empty() {
        upload.particles.len()
    } else {
        readback_snapshot.particles.len()
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
        boundary_friction_water: p.boundary.water,
        boundary_friction_granular: p.boundary.granular,
        _pad_friction: 0,
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

/// Diagnostics counter for GPU readback frames.
static GPU_READBACK_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// System: apply GPU readback results to snapshot cache.
pub fn apply_gpu_readback(
    control: Res<MpmGpuControl>,
    readback_result: Res<GpuReadbackResult>,
    sim_state: Res<SimulationState>,
    upload: Res<MpmGpuUploadRequest>,
    mut readback_snapshot: ResMut<MpmReadbackSnapshot>,
    mut residency: ResMut<MpmChunkResidencyState>,
) {
    if !control.readback_enabled {
        return;
    }
    let Some(particles) = readback_result.take() else {
        return;
    };
    if !sim_state.gpu_mpm_active && upload.particles.is_empty() {
        return;
    }
    readback_snapshot.particles = particles.clone();

    if residency.initialized {
        let mut invalid_particles = 0u64;
        for particle in &particles {
            let chunk = chunk_coord_from_world_pos(Vec2::from_array(particle.x));
            if !residency.chunk_to_slot.contains_key(&chunk) {
                invalid_particles = invalid_particles.saturating_add(1);
            }
        }
        residency.invalid_slot_access_count = residency
            .invalid_slot_access_count
            .saturating_add(invalid_particles);
    }

    let frame = GPU_READBACK_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let n = particles.len();
    // Periodic NaN / divergence check (every 60 readback frames).
    if frame % 60 == 0 {
        let mut nan_count = 0u32;
        let mut max_speed: f32 = 0.0;
        for i in 0..n {
            let p = &particles[i];
            if p.x[0].is_nan() || p.x[1].is_nan() || p.v[0].is_nan() || p.v[1].is_nan() {
                nan_count += 1;
            }
            let speed_sq = p.v[0] * p.v[0] + p.v[1] * p.v[1];
            if speed_sq > max_speed {
                max_speed = speed_sq;
            }
        }
        let max_speed = max_speed.sqrt();
        if nan_count > 0 {
            bevy::log::error!("[gpu_mpm] frame={frame} NaN detected in {nan_count}/{n} particles");
        } else {
            bevy::log::info!("[gpu_mpm] frame={frame} n={n} max_speed={max_speed:.2} m/s — OK");
        }
    }
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

    upload.chunk_meta_diffs.clear();
    upload.upload_chunk_diffs = false;
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
            upload.chunk_meta_diffs.push((slot_id as u32, meta));
        }
    }

    let slot_capacity = residency.resident_chunk_count.max(1);
    if !upload.chunk_meta_diffs.is_empty()
        && upload.chunk_meta_diffs.len() as u32 >= slot_capacity / 2
    {
        let slot_span = residency.resident_chunk_count as usize;
        upload.chunk_meta = residency.chunk_meta_cache[..slot_span].to_vec();
        upload.upload_chunks = true;
        upload.upload_chunk_diffs = false;
        upload.chunk_meta_diffs.clear();
    } else {
        upload.upload_chunk_diffs = !upload.chunk_meta_diffs.is_empty();
    }

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
        upload.chunk_meta_diffs.clear();
        upload.upload_chunk_diffs = false;
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
                upload.chunk_meta_diffs.push((slot_id as u32, meta));
            }
        }
        upload.upload_chunk_diffs = !upload.chunk_meta_diffs.is_empty();
        if upload.upload_chunk_diffs {
            let slot_span = residency.resident_chunk_count as usize;
            if upload.chunk_meta_diffs.len() >= slot_span / 2 {
                upload.chunk_meta = residency.chunk_meta_cache[..slot_span].to_vec();
                upload.upload_chunks = true;
                upload.upload_chunk_diffs = false;
                upload.chunk_meta_diffs.clear();
            }
        }
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

fn remove_gpu_particles_in_cells(
    upload_particles: &mut Vec<GpuParticle>,
    cells: &[IVec2],
    max_count: usize,
) -> Vec<usize> {
    let target_cells: HashSet<IVec2> = cells.iter().copied().collect();
    if target_cells.is_empty() {
        return Vec::new();
    }
    let mut removed_mpm_indices = Vec::new();
    let mut keep = vec![true; upload_particles.len()];
    for (i, particle) in upload_particles.iter().enumerate() {
        if removed_mpm_indices.len() >= max_count {
            break;
        }
        let cell = world_to_cell(Vec2::from_array(particle.x));
        if target_cells.contains(&cell) {
            keep[i] = false;
            removed_mpm_indices.push(i);
        }
    }
    if removed_mpm_indices.is_empty() {
        return removed_mpm_indices;
    }
    compact_vec_by_keep(upload_particles, &keep);
    removed_mpm_indices
}

fn compact_vec_by_keep<T: Copy>(data: &mut Vec<T>, keep: &[bool]) {
    let mut write = 0usize;
    for read in 0..data.len() {
        if !keep[read] {
            continue;
        }
        if write != read {
            data[write] = data[read];
        }
        write += 1;
    }
    data.truncate(write);
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

    if !is_allocated
        || !slot_state
            .get(slot_id as usize)
            .map(|slot| slot.resident)
            .unwrap_or(false)
    {
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
                let global_cell = chunk_cell_origin + IVec2::new(local_x as i32, local_y as i32);
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
    use crate::physics::gpu_mpm::readback::{GpuReadbackResult, GpuStatisticsReadbackResult};
    use crate::physics::material::ParticleMaterial;
    use crate::physics::state::SimulationState;
    use bevy::ecs::message::Messages;

    fn setup_edit_app() -> App {
        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(ActivePhysicsParams::default())
            .insert_resource(MpmGpuUploadRequest::default())
            .insert_resource(MpmReadbackSnapshot::default())
            .add_message::<GpuWorldEditRequest>()
            .add_systems(Update, apply_world_edit_requests);
        app
    }

    #[test]
    fn world_edit_add_particles_updates_upload_and_snapshot() {
        let mut app = setup_edit_app();
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
        let snapshot = app.world().resource::<MpmReadbackSnapshot>();

        assert_eq!(upload.particles.len(), 8);
        assert_eq!(snapshot.particles.len(), 8);
        assert!(upload.upload_particles);
        for p in &upload.particles {
            let cell = world_to_cell(Vec2::from_array(p.x));
            assert!(cells.contains(&cell));
        }
    }

    #[test]
    fn world_edit_remove_particles_respects_max_count() {
        let mut app = setup_edit_app();
        let target = IVec2::new(2, 3);

        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: vec![target],
                command: GpuWorldEditCommand::AddParticles {
                    material: ParticleMaterial::WaterLiquid,
                    count_per_cell: 9,
                },
            });
        app.update();

        app.world_mut()
            .resource_mut::<Messages<GpuWorldEditRequest>>()
            .write(GpuWorldEditRequest {
                cells: vec![target],
                command: GpuWorldEditCommand::RemoveParticles { max_count: 4 },
            });
        app.update();

        let upload = app.world().resource::<MpmGpuUploadRequest>();

        assert_eq!(upload.particles.len(), 5);
    }

    #[test]
    fn world_edit_add_particles_prefers_latest_readback_snapshot() {
        let mut app = setup_edit_app();
        let stale_particle = GpuParticle::from_cpu(
            Vec2::new(0.25, 0.25),
            Vec2::ZERO,
            1.0,
            1.0,
            Mat2::IDENTITY,
            Mat2::ZERO,
            0.0,
            MPM_PHASE_ID_WATER,
        );
        let latest_particle = GpuParticle::from_cpu(
            cell_to_world_center(IVec2::new(4, 5)),
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
            world.resource_mut::<MpmReadbackSnapshot>().particles = vec![latest_particle];
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
        let snapshot = app.world().resource::<MpmReadbackSnapshot>();

        assert_eq!(upload.particles.len(), 5);
        assert_eq!(snapshot.particles.len(), 5);
        assert!(
            upload
                .particles
                .iter()
                .any(|particle| world_to_cell(Vec2::from_array(particle.x)) == IVec2::new(4, 5))
        );
        assert!(
            upload
                .particles
                .iter()
                .all(|particle| world_to_cell(Vec2::from_array(particle.x)) != IVec2::new(0, 0))
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
            .insert_resource(MpmReadbackSnapshot {
                particles: vec![GpuParticle::from_cpu(
                    Vec2::new(1.0, 2.0),
                    Vec2::ZERO,
                    1.0,
                    1.0,
                    Mat2::IDENTITY,
                    Mat2::ZERO,
                    0.0,
                    MPM_PHASE_ID_WATER,
                )],
            })
            .insert_resource(MpmGpuUploadRequest {
                upload_particles: true,
                ..Default::default()
            })
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
    fn apply_gpu_readback_ignores_stale_when_gpu_path_inactive() {
        let mut app = App::new();
        app.insert_resource(MpmGpuControl::default())
            .insert_resource(GpuReadbackResult::default())
            .insert_resource(SimulationState::default())
            .insert_resource(MpmGpuUploadRequest::default())
            .insert_resource(MpmReadbackSnapshot::default())
            .insert_resource(MpmChunkResidencyState::default())
            .add_systems(Update, apply_gpu_readback);

        app.world()
            .resource::<GpuReadbackResult>()
            .store(vec![GpuParticle::from_cpu(
                Vec2::new(3.0, 4.0),
                Vec2::ZERO,
                1.0,
                1.0,
                Mat2::IDENTITY,
                Mat2::ZERO,
                0.0,
                MPM_PHASE_ID_WATER,
            )]);

        app.update();

        let snapshot = app.world().resource::<MpmReadbackSnapshot>();
        assert!(snapshot.particles.is_empty());
    }
}
