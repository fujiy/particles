// GPU buffer layout definitions for MLS-MPM.
//
// All data is GPU-resident. CPU only holds metadata and readback statistics.

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// GPU-side particle layout (matches mpm_types.wgsl::GpuParticle, 72 bytes)
// Layout: x(8) v(8) mass(4) v0(4) f(16) c(16) material_id(4) _pad(12) = 72
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
    pub material_id: u32,
    pub _pad: [u32; 3],
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
        material_id: u8,
    ) -> Self {
        Self {
            x: pos.to_array(),
            v: vel.to_array(),
            mass,
            v0: rest_volume,
            // Mat2 in glam is column-major: col0 = (x_axis.x, x_axis.y), col1 = (y_axis.x, y_axis.y)
            f: [f.x_axis.x, f.x_axis.y, f.y_axis.x, f.y_axis.y],
            c: [c.x_axis.x, c.x_axis.y, c.y_axis.x, c.y_axis.y],
            material_id: material_id as u32,
            _pad: [0; 3],
        }
    }
}

// ---------------------------------------------------------------------------
// GPU-side grid node (matches mpm_types.wgsl::GpuGridNode, 16 bytes)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuGridNode {
    pub px: f32,
    pub py: f32,
    pub mass: f32,
    pub _pad: f32,
}

const _: () = assert!(std::mem::size_of::<GpuGridNode>() == 16);

// ---------------------------------------------------------------------------
// Simulation parameters uniform (matches mpm_types.wgsl::MpmParams, 80 bytes)
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
    pub deep_push_gain_per_s: f32,
    pub deep_push_speed_cap_mps: f32,
    pub tangential_damping: f32,
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<GpuMpmParams>() == 80);

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
            deep_push_gain_per_s: 0.1,
            deep_push_speed_cap_mps: 1.0,
            tangential_damping: 0.05,
            _pad: [0; 2],
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
