// CPU → GPU synchronization systems for MLS-MPM.
//
// Runs in main-world (not render-world): prepares upload requests,
// which are then executed in the render-world extract/prepare phase.

use bevy::prelude::*;
use std::collections::HashSet;

use super::buffers::{
    self, GpuChunkMeta, GpuGridLayout, GpuMpmParams, GpuParticle, GpuStatisticsScalars,
    INVALID_CHUNK_SLOT_ID,
};
use super::gpu_resources::{
    MAX_RESIDENT_CHUNK_SLOTS, MPM_CHUNK_NODE_DIM, MpmGpuControl, MpmGpuParamsRequest,
    MpmGpuRunRequest, MpmGpuStepClock, MpmGpuUploadRequest, world_grid_layout,
};
use super::phase::mpm_phase_id_for_particle;
use super::readback::{GpuReadbackResult, GpuStatisticsReadbackResult};
use crate::params::ActivePhysicsParams;
use crate::physics::material::{ParticleMaterial, particle_properties};
use crate::physics::state::{ReplayState, SimulationState};
use crate::physics::world::constants::{CELL_SIZE_M, CHUNK_SIZE_I32};
use crate::physics::world::terrain::{TerrainWorld, cell_to_world_center, world_to_cell};

const TERRAIN_SDF_DISABLED_DISTANCE_M: f32 = 1.0e6;
const CHUNK_HALO_RADIUS: i32 = 2;

#[derive(Resource, Debug, Default)]
pub struct MpmReadbackSnapshot {
    pub particles: Vec<GpuParticle>,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct MpmChunkResidencyState {
    pub initialized: bool,
    pub chunk_origin: IVec2,
    pub chunk_dims: UVec2,
    pub grid_layout: GpuGridLayout,
    pub resident_chunk_count: u32,
    pub chunk_sdf_samples: u32,
    pub invalid_slot_access_count: u64,
}

impl Default for MpmChunkResidencyState {
    fn default() -> Self {
        Self {
            initialized: false,
            chunk_origin: IVec2::ZERO,
            chunk_dims: UVec2::ZERO,
            grid_layout: world_grid_layout(),
            resident_chunk_count: 0,
            chunk_sdf_samples: 0,
            invalid_slot_access_count: 0,
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
        if upload.particles.is_empty() {
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
        upload.upload_chunks = false;
        sim_state.gpu_mpm_active = false;
        upload.particles.clear();
        return;
    }
    let upload_requested = upload.upload_particles;
    upload.upload_particles_frame = upload_requested;
    // Consume main-world request flag; explicit requests set this again when needed.
    upload.upload_particles = false;

    // Keep GPU run-state stable regardless of one-shot upload flag consumption.
    // `upload.particles` is the authoritative CPU-side snapshot for the current set.
    let has_particles = !upload.particles.is_empty() || !readback_snapshot.particles.is_empty();
    // Pure GPU mode: no CPU fallback; active when MPM is enabled and particles exist.
    sim_state.gpu_mpm_active = sim_state.mpm_enabled && has_particles;
}

/// System: build terrain SDF/normal upload request.
///
/// Only re-uploads when TerrainWorld changes.
pub fn prepare_terrain_upload(
    control: Res<MpmGpuControl>,
    sim_state: Res<SimulationState>,
    terrain: Res<TerrainWorld>,
    readback_snapshot: Res<MpmReadbackSnapshot>,
    mut residency: ResMut<MpmChunkResidencyState>,
    mut upload: ResMut<MpmGpuUploadRequest>,
) {
    // Default: no chunk/terrain upload this frame unless explicitly requested below.
    upload.upload_chunks = false;
    upload.upload_terrain = false;

    if control.init_only {
        upload.chunk_meta.clear();
        upload.terrain_sdf.clear();
        upload.terrain_normal.clear();
        upload.last_uploaded_terrain_version = None;
        residency.initialized = false;
        residency.resident_chunk_count = 0;
        residency.chunk_sdf_samples = 0;
        residency.invalid_slot_access_count = 0;
        return;
    }
    // When MLS-MPM stepping is disabled, terrain upload is unnecessary for overlay debug.
    if !sim_state.mpm_enabled {
        return;
    }

    // MPM-CHUNK-01: static residency.
    // Rebuild only on initial bring-up or explicit particle upload (scenario/reset/edit).
    let should_rebuild = !residency.initialized || upload.upload_particles_frame;
    if !should_rebuild {
        return;
    }

    let particle_source = if !upload.particles.is_empty() {
        upload.particles.as_slice()
    } else {
        readback_snapshot.particles.as_slice()
    };
    let Some(build) = build_static_chunk_upload(&terrain, particle_source) else {
        upload.chunk_meta.clear();
        upload.terrain_sdf.clear();
        upload.terrain_normal.clear();
        residency.initialized = false;
        residency.resident_chunk_count = 0;
        residency.chunk_sdf_samples = 0;
        return;
    };

    upload.chunk_meta = build.chunk_meta;
    upload.upload_chunks = true;
    upload.terrain_sdf = build.terrain_sdf;
    upload.terrain_normal = build.terrain_normal;
    upload.upload_terrain = true;

    residency.initialized = true;
    residency.chunk_origin = build.chunk_origin;
    residency.chunk_dims = build.chunk_dims;
    residency.grid_layout = build.grid_layout;
    residency.resident_chunk_count = residency.chunk_dims.x.saturating_mul(residency.chunk_dims.y);
    residency.chunk_sdf_samples = build.grid_layout.node_count() as u32;
    residency.invalid_slot_access_count = 0;
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
    let particle_count = if !upload.particles.is_empty() {
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
    mut readback_snapshot: ResMut<MpmReadbackSnapshot>,
    mut residency: ResMut<MpmChunkResidencyState>,
) {
    if !control.readback_enabled {
        return;
    }
    let Some(particles) = readback_result.take() else {
        return;
    };
    readback_snapshot.particles = particles.clone();

    if residency.initialized {
        let chunk_max = residency.chunk_origin + residency.chunk_dims.as_ivec2();
        let mut invalid_particles = 0u64;
        for particle in &particles {
            let chunk = chunk_coord_from_world_pos(Vec2::from_array(particle.x));
            if chunk.x < residency.chunk_origin.x
                || chunk.y < residency.chunk_origin.y
                || chunk.x >= chunk_max.x
                || chunk.y >= chunk_max.y
            {
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
    chunk_meta: Vec<GpuChunkMeta>,
    terrain_sdf: Vec<f32>,
    terrain_normal: Vec<[f32; 2]>,
}

fn chunk_coord_from_world_pos(world_pos: Vec2) -> IVec2 {
    let cell = world_to_cell(world_pos);
    IVec2::new(
        cell.x.div_euclid(CHUNK_SIZE_I32),
        cell.y.div_euclid(CHUNK_SIZE_I32),
    )
}

fn slot_id_from_chunk_coord(chunk: IVec2, origin: IVec2, dims: UVec2) -> Option<u32> {
    let lx = chunk.x - origin.x;
    let ly = chunk.y - origin.y;
    if lx < 0 || ly < 0 {
        return None;
    }
    let lx_u = lx as u32;
    let ly_u = ly as u32;
    if lx_u >= dims.x || ly_u >= dims.y {
        return None;
    }
    Some(ly_u * dims.x + lx_u)
}

fn build_static_chunk_upload(
    terrain: &TerrainWorld,
    particles: &[GpuParticle],
) -> Option<StaticChunkUploadBuild> {
    let mut occupied = HashSet::<IVec2>::default();
    for chunk in terrain.loaded_chunk_coords() {
        occupied.insert(chunk);
    }
    for particle in particles {
        occupied.insert(chunk_coord_from_world_pos(Vec2::from_array(particle.x)));
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
    let chunk_dims = chunk_dims_i.as_uvec2();
    let resident_chunk_count = chunk_dims.x.saturating_mul(chunk_dims.y);
    if resident_chunk_count > MAX_RESIDENT_CHUNK_SLOTS {
        bevy::log::warn!(
            "[gpu_mpm] static chunk residency exceeds slot capacity: requested={} limit={}",
            resident_chunk_count,
            MAX_RESIDENT_CHUNK_SLOTS
        );
        return None;
    }

    let mut chunk_meta = vec![GpuChunkMeta::default(); resident_chunk_count as usize];
    let neighbor_offsets = [
        IVec2::new(-1, -1),
        IVec2::new(0, -1),
        IVec2::new(1, -1),
        IVec2::new(-1, 0),
        IVec2::new(1, 0),
        IVec2::new(-1, 1),
        IVec2::new(0, 1),
        IVec2::new(1, 1),
    ];
    for cy in min_chunk.y..=max_chunk.y {
        for cx in min_chunk.x..=max_chunk.x {
            let coord = IVec2::new(cx, cy);
            let Some(slot_id) = slot_id_from_chunk_coord(coord, min_chunk, chunk_dims) else {
                continue;
            };
            let meta = &mut chunk_meta[slot_id as usize];
            meta.chunk_coord_x = cx;
            meta.chunk_coord_y = cy;
            meta.active_tile_mask = 0;
            for (i, offset) in neighbor_offsets.into_iter().enumerate() {
                let neighbor = coord + offset;
                meta.neighbor_slot_id[i] =
                    slot_id_from_chunk_coord(neighbor, min_chunk, chunk_dims)
                        .unwrap_or(INVALID_CHUNK_SLOT_ID);
            }
        }
    }

    let chunk_node_dim_i = MPM_CHUNK_NODE_DIM as i32;
    let grid_layout = GpuGridLayout {
        origin: IVec2::new(min_chunk.x * chunk_node_dim_i, min_chunk.y * chunk_node_dim_i),
        dims: UVec2::new(
            chunk_dims.x * MPM_CHUNK_NODE_DIM,
            chunk_dims.y * MPM_CHUNK_NODE_DIM,
        ),
    };
    let node_count = grid_layout.node_count();
    let mut terrain_sdf = vec![TERRAIN_SDF_DISABLED_DISTANCE_M; node_count];
    let mut terrain_normal = vec![[0.0_f32, 1.0_f32]; node_count];
    let chunk_node_dim_u = MPM_CHUNK_NODE_DIM as usize;
    let nodes_per_chunk = chunk_node_dim_u * chunk_node_dim_u;
    let chunk_dims_x = chunk_dims.x as usize;
    for y in 0..grid_layout.dims.y {
        for x in 0..grid_layout.dims.x {
            let node = IVec2::new(grid_layout.origin.x + x as i32, grid_layout.origin.y + y as i32);
            let world_pos = Vec2::new(node.x as f32 * CELL_SIZE_M, node.y as f32 * CELL_SIZE_M);
            if let Some((sdf, normal)) = terrain.sample_signed_distance_and_normal(world_pos) {
                let x_u = x as usize;
                let y_u = y as usize;
                let chunk_lx = x_u / chunk_node_dim_u;
                let chunk_ly = y_u / chunk_node_dim_u;
                let local_x = x_u % chunk_node_dim_u;
                let local_y = y_u % chunk_node_dim_u;
                let slot_id = chunk_ly * chunk_dims_x + chunk_lx;
                let idx = slot_id * nodes_per_chunk + local_y * chunk_node_dim_u + local_x;
                terrain_sdf[idx] = sdf;
                let n = normal.normalize_or_zero();
                terrain_normal[idx] = if n == Vec2::ZERO {
                    [0.0, 1.0]
                } else {
                    [n.x, n.y]
                };
            }
        }
    }

    Some(StaticChunkUploadBuild {
        chunk_origin: min_chunk,
        chunk_dims,
        grid_layout,
        chunk_meta,
        terrain_sdf,
        terrain_normal,
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
    use crate::physics::gpu_mpm::gpu_resources::MpmGpuUploadRequest;
    use crate::physics::gpu_mpm::readback::GpuStatisticsReadbackResult;
    use crate::physics::material::ParticleMaterial;
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
}
