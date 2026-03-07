// GPU buffer layout definitions for MLS-MPM.
//
// All data is GPU-resident. CPU only holds metadata and readback statistics.

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

pub const INVALID_CHUNK_SLOT_ID: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// GPU-side chunk metadata (static residency for MPM-CHUNK-01)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuChunkMeta {
    pub chunk_coord_x: i32,
    pub chunk_coord_y: i32,
    /// Moore neighborhood slot ids ordered as:
    /// (-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)
    pub neighbor_slot_id: [u32; 8],
    pub active_tile_mask: u32,
    pub _pad: u32,
}

impl Default for GpuChunkMeta {
    fn default() -> Self {
        Self {
            chunk_coord_x: 0,
            chunk_coord_y: 0,
            neighbor_slot_id: [INVALID_CHUNK_SLOT_ID; 8],
            active_tile_mask: 0,
            _pad: 0,
        }
    }
}

const _: () = assert!(std::mem::size_of::<GpuChunkMeta>() == 48);

// ---------------------------------------------------------------------------
// GPU-side particle layout (matches mpm_types.wgsl::GpuParticle, 72 bytes)
// Layout: x(8) v(8) mass(4) v0(4) f(16) c(16) v_vol(4) phase_id(4) phi_p(4) home_slot(4) = 72
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuParticle {
    pub x: [f32; 2],
    pub v: [f32; 2],
    pub mass: f32,
    pub v0: f32,
    /// F column-major: [f00, f01, f10, f11]
    pub f: [f32; 4],
    /// C column-major: [c00, c01, c10, c11]
    pub c: [f32; 4],
    /// Plastic volume correction scalar diff_log_J (granular only, water keeps 0.0).
    pub v_vol: f32,
    /// Continuum phase id (0=water, 1=soil, 2=sand).
    pub phase_id: u32,
    /// Water fill fraction φ_p interpolated from grid (water only). Written by G2P, read by P2G.
    pub phi_p: f32,
    /// Current occupied/home chunk slot id for this particle.
    pub home_chunk_slot_id: u32,
}

const _: () = assert!(std::mem::size_of::<GpuParticle>() == 72);

impl GpuParticle {
    pub fn from_cpu(
        pos: Vec2,
        vel: Vec2,
        mass: f32,
        rest_volume: f32,
        f: Mat2,
        c: Mat2,
        v_vol: f32,
        phase_id: u8,
    ) -> Self {
        Self {
            x: pos.to_array(),
            v: vel.to_array(),
            mass,
            v0: rest_volume,
            // Mat2 in glam is column-major: col0 = (x_axis.x, x_axis.y), col1 = (y_axis.x, y_axis.y)
            f: [f.x_axis.x, f.x_axis.y, f.y_axis.x, f.y_axis.y],
            c: [c.x_axis.x, c.x_axis.y, c.y_axis.x, c.y_axis.y],
            v_vol,
            phase_id: phase_id as u32,
            phi_p: 1.0,
            home_chunk_slot_id: INVALID_CHUNK_SLOT_ID,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU-side movers (MPM-CHUNK-02 staging)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuMoverRecord {
    pub particle_id: u32,
    pub old_home_slot_id: u32,
    pub new_chunk_coord_x: i32,
    pub new_chunk_coord_y: i32,
}

const _: () = assert!(std::mem::size_of::<GpuMoverRecord>() == 16);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuMoverResult {
    pub particle_id: u32,
    pub new_home_slot_id: u32,
    pub _pad_a: u32,
    pub _pad_b: u32,
}

const _: () = assert!(std::mem::size_of::<GpuMoverResult>() == 16);

// ---------------------------------------------------------------------------
// GPU-side grid node (matches mpm_types.wgsl::GpuGridNode, 32 bytes)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuGridNode {
    pub water_px: f32,
    pub water_py: f32,
    pub water_mass: f32,
    pub water_pad: f32,
    pub granular_px: f32,
    pub granular_py: f32,
    pub granular_mass: f32,
    pub granular_pad: f32,
}

const _: () = assert!(std::mem::size_of::<GpuGridNode>() == 32);

// ---------------------------------------------------------------------------
// Simulation parameters uniform (matches mpm_types.wgsl::MpmParams, 144 bytes)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMpmParams {
    pub dt: f32,
    pub gx: f32,
    pub gy: f32,
    pub rho0: f32,
    pub bulk_modulus: f32,
    pub h: f32,
    pub grid_origin_x: i32,
    pub grid_origin_y: i32,
    pub grid_width: u32,
    pub grid_height: u32,
    pub particle_count: u32,
    pub j_min: f32,
    pub j_max: f32,
    pub c_max_norm: f32,
    pub sdf_velocity_threshold_m: f32,
    /// Coulomb friction coefficient μ_b for water at terrain boundary.
    pub boundary_friction_water: f32,
    /// Coulomb friction coefficient μ_b for granular at terrain boundary.
    pub boundary_friction_granular: f32,
    pub _pad_friction: u32,
    pub dp_lambda_soil: f32,
    pub dp_mu_soil: f32,
    pub dp_alpha_soil: f32,
    pub dp_k_soil: f32,
    pub dp_hardening_soil: f32,
    pub dp_lambda_sand: f32,
    pub dp_mu_sand: f32,
    pub dp_alpha_sand: f32,
    pub dp_k_sand: f32,
    pub dp_hardening_sand: f32,
    pub granular_tensile_clamp: f32,
    /// Symmetric inter-phase drag gamma [Eq.39, physics.md]
    pub coupling_drag_gamma: f32,
    pub coupling_friction: f32,
    /// Minimum |∇phi_g| to enable interface normal/friction coupling [Eq.41, physics.md]
    pub coupling_interface_min_grad: f32,
    /// Epsilon for interface normal normalization denominator.
    pub coupling_interface_normal_eps: f32,
    /// APIC↔PIC blend coefficient for water [Eq.32, physics.md]. 1.0=full APIC, 0.0=full PIC.
    pub alpha_apic_water: f32,
    /// APIC↔PIC blend coefficient for granular [Eq.32, physics.md].
    pub alpha_apic_granular: f32,
    /// Statistics: tracked phase id (0=water,1=soil,2=sand, 0xFFFF_FFFF=all MPM phases).
    pub stats_tracked_phase_id: u32,
    /// Statistics: granular repose target phase id.
    pub stats_repose_phase_id: u32,
    /// Statistics: material interaction primary phase id.
    pub stats_interaction_primary_phase_id: u32,
    /// Statistics: material interaction secondary phase id.
    pub stats_interaction_secondary_phase_id: u32,
    /// Statistics: penetration threshold against terrain SDF (meters).
    pub stats_penetration_epsilon_m: f32,
    /// Statistics: fixed-point scale for position accumulations.
    pub stats_position_fp_scale: f32,
    /// Static chunk residency layout origin in chunk coords.
    pub chunk_origin_x: i32,
    pub chunk_origin_y: i32,
    /// Static chunk residency layout dimensions in chunks.
    pub chunk_dims_x: u32,
    pub chunk_dims_y: u32,
    /// Active resident chunk count for diagnostics and bounds checks.
    pub resident_chunk_count: u32,
    /// Nodes per chunk axis on current solver grid.
    pub chunk_node_dim: u32,
    pub _pad_tail: [u32; 1],
}

const _: () = assert!(std::mem::size_of::<GpuMpmParams>() == 192);

pub const GPU_STATS_LANE_TOTAL_PARTICLES: usize = 0;
pub const GPU_STATS_LANE_PHASE_WATER: usize = 1;
pub const GPU_STATS_LANE_PHASE_GRANULAR_SOIL: usize = 2;
pub const GPU_STATS_LANE_PHASE_GRANULAR_SAND: usize = 3;
pub const GPU_STATS_LANE_PHASE_UNKNOWN: usize = 4;
pub const GPU_STATS_LANE_MAX_SPEED_BITS: usize = 5;
pub const GPU_STATS_LANE_ALL_COUNT: usize = 6;
pub const GPU_STATS_LANE_ALL_PENETRATION_COUNT: usize = 7;
/// Signed fixed-point sum (i32 bit pattern in u32 lane).
pub const GPU_STATS_LANE_ALL_SUM_Y_FP: usize = 8;
pub const GPU_STATS_LANE_TRACKED_COUNT: usize = 9;
pub const GPU_STATS_LANE_TRACKED_PENETRATION_COUNT: usize = 10;
/// Signed fixed-point sum (i32 bit pattern in u32 lane).
pub const GPU_STATS_LANE_TRACKED_SUM_Y_FP: usize = 11;
/// i32 bit pattern (cell y index).
pub const GPU_STATS_LANE_WATER_SURFACE_P95_CELL_BITS: usize = 12;
pub const GPU_STATS_LANE_GRANULAR_REPOSE_ANGLE_BITS: usize = 13;
/// i32 bit pattern.
pub const GPU_STATS_LANE_GRANULAR_REPOSE_BASE_SPAN_BITS: usize = 14;
pub const GPU_STATS_LANE_INTERACTION_PRIMARY_COUNT: usize = 15;
pub const GPU_STATS_LANE_INTERACTION_CONTACT_COUNT: usize = 16;
/// Signed fixed-point sum (i32 bit pattern in u32 lane).
pub const GPU_STATS_LANE_INTERACTION_PRIMARY_SUM_Y_FP: usize = 17;
pub const GPU_STATS_LANE_INTERACTION_SECONDARY_COUNT: usize = 18;
/// Signed fixed-point sum (i32 bit pattern in u32 lane).
pub const GPU_STATS_LANE_INTERACTION_SECONDARY_SUM_Y_FP: usize = 19;
pub const GPU_STATS_LANE_INTERACTION_CONTACT_RATIO_BITS: usize = 20;
pub const GPU_STATS_LANE_INTERACTION_PRIMARY_CENTROID_Y_BITS: usize = 21;
pub const GPU_STATS_LANE_INTERACTION_SECONDARY_CENTROID_Y_BITS: usize = 22;
pub const GPU_STATS_LANE_REPOSE_COUNT: usize = 23;
pub const GPU_STATS_LANE_GRID_WATER_PHI_MAX_BITS: usize = 24;
pub const GPU_STATS_LANE_GRID_WATER_PHI_MEAN_NONZERO_BITS: usize = 25;
pub const GPU_STATS_LANE_GRID_WATER_PHI_P99_BITS: usize = 26;
pub const GPU_STATS_LANE_GRID_WATER_NONZERO_NODES: usize = 27;
pub const GPU_STATS_LANE_GRID_GRANULAR_PHI_MAX_BITS: usize = 28;
pub const GPU_STATS_LANE_GRID_GRANULAR_PHI_MEAN_NONZERO_BITS: usize = 29;
pub const GPU_STATS_LANE_GRID_GRANULAR_PHI_P99_BITS: usize = 30;
pub const GPU_STATS_LANE_GRID_GRANULAR_NONZERO_NODES: usize = 31;
/// Ordered-f32 bits for atomic min/max.
pub const GPU_STATS_LANE_REPOSE_X_MIN_ORDERED_BITS: usize = 32;
pub const GPU_STATS_LANE_REPOSE_X_MAX_ORDERED_BITS: usize = 33;
pub const GPU_STATS_LANE_REPOSE_Y_MIN_ORDERED_BITS: usize = 34;
pub const GPU_STATS_LANE_REPOSE_Y_MAX_ORDERED_BITS: usize = 35;
/// Ordered-f32 bits for tracked position bounds.
pub const GPU_STATS_LANE_TRACKED_X_MIN_ORDERED_BITS: usize = 36;
pub const GPU_STATS_LANE_TRACKED_X_MAX_ORDERED_BITS: usize = 37;
pub const GPU_STATS_LANE_TRACKED_Y_MIN_ORDERED_BITS: usize = 38;
pub const GPU_STATS_LANE_TRACKED_Y_MAX_ORDERED_BITS: usize = 39;

pub const GPU_STATS_SCALAR_BASE_LANES: usize = 64;
pub const GPU_STATS_WATER_SURFACE_HIST_BINS: usize = 256;
pub const GPU_STATS_LANE_WATER_SURFACE_HIST_BASE: usize = GPU_STATS_SCALAR_BASE_LANES;
pub const GPU_STATS_SCALAR_LANES: usize =
    GPU_STATS_SCALAR_BASE_LANES + GPU_STATS_WATER_SURFACE_HIST_BINS;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuStatisticsScalars {
    pub lanes: [u32; GPU_STATS_SCALAR_LANES],
}

impl Default for GpuStatisticsScalars {
    fn default() -> Self {
        Self {
            lanes: [0; GPU_STATS_SCALAR_LANES],
        }
    }
}

const _: () = assert!(std::mem::size_of::<GpuStatisticsScalars>() == 1280);

impl Default for GpuMpmParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            gx: 0.0,
            gy: -9.81,
            rho0: 1_000.0,
            bulk_modulus: 2.56e5,
            h: 0.25,
            grid_origin_x: 0,
            grid_origin_y: 0,
            grid_width: 0,
            grid_height: 0,
            particle_count: 0,
            j_min: 0.6,
            j_max: 1.4,
            c_max_norm: 80.0,
            sdf_velocity_threshold_m: 0.01,
            boundary_friction_water: 0.3,
            boundary_friction_granular: 0.6,
            _pad_friction: 0,
            dp_lambda_soil: 2.0e4,
            dp_mu_soil: 1.2e4,
            dp_alpha_soil: 0.2,
            dp_k_soil: 200.0,
            dp_hardening_soil: 1.5,
            dp_lambda_sand: 1.8e4,
            dp_mu_sand: 1.0e4,
            dp_alpha_sand: 0.25,
            dp_k_sand: 60.0,
            dp_hardening_sand: 1.0,
            granular_tensile_clamp: 0.0,
            coupling_drag_gamma: 3.5,
            coupling_friction: 0.6,
            coupling_interface_min_grad: 0.02,
            coupling_interface_normal_eps: 1.0e-6,
            alpha_apic_water: 0.95,
            alpha_apic_granular: 0.78,
            stats_tracked_phase_id: 0,
            stats_repose_phase_id: 1,
            stats_interaction_primary_phase_id: 2,
            stats_interaction_secondary_phase_id: 0,
            stats_penetration_epsilon_m: 1.0e-3,
            stats_position_fp_scale: 100.0,
            chunk_origin_x: 0,
            chunk_origin_y: 0,
            chunk_dims_x: 0,
            chunk_dims_y: 0,
            resident_chunk_count: 0,
            chunk_node_dim: 0,
            _pad_tail: [0; 1],
        }
    }
}

// ---------------------------------------------------------------------------
// Grid layout helper
// ---------------------------------------------------------------------------

/// Compute flat grid geometry from world constants.
///
/// Grid covers [world_min_node, world_max_node] inclusive with extra padding.
#[derive(Clone, Copy, Debug)]
pub struct GpuGridLayout {
    /// Node index of the grid origin (bottom-left corner).
    pub origin: IVec2,
    /// Grid dimensions in nodes.
    pub dims: UVec2,
}

impl GpuGridLayout {
    pub fn node_count(&self) -> usize {
        (self.dims.x as usize) * (self.dims.y as usize)
    }

    pub fn node_index(&self, node: IVec2) -> Option<u32> {
        let local = node - self.origin;
        if local.x < 0
            || local.y < 0
            || local.x >= self.dims.x as i32
            || local.y >= self.dims.y as i32
        {
            return None;
        }
        Some(local.y as u32 * self.dims.x + local.x as u32)
    }
}

// ---------------------------------------------------------------------------
// CPU-side readback metrics (minimal, polled async)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuMpmReadbackMetrics {
    pub particle_mass_sum: f32,
    pub max_particle_speed_mps: f32,
    pub boundary_penetrating_particle_ratio: f32,
}
