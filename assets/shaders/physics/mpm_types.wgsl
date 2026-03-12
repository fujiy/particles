// MLS-MPM shared type definitions.
// Imported by each compute shader via #import.
#define_import_path particles::mpm_types

// Particle storage layout (one array of GpuParticle, 72 bytes each).
// Matches Rust GpuParticle: x(8)+v(8)+mass(4)+v0(4)+f(16)+c(16)+v_vol(4)+phase_id(4)+phi_p(4)+home_slot+material(4)=72
struct GpuParticle {
    // position (m)
    x: vec2<f32>,
    // velocity (m/s)
    v: vec2<f32>,
    // mass (kg)
    mass: f32,
    // rest volume (m^2)
    v_rest: f32,
    // deformation gradient F (column-major: col0=(a,b), col1=(c,d))
    f_a: f32,
    f_b: f32,
    f_c: f32,
    f_d: f32,
    // APIC affine matrix C (column-major)
    c_a: f32,
    c_b: f32,
    c_c: f32,
    c_d: f32,
    // Plastic volume correction scalar diff_log_J [Eq.21, physics.md]
    v_vol: f32,
    // phase id (0=water, 1=granular soil, 2=granular sand)
    phase_id: u32,
    // Water fill fraction φ_p [Eq.9, physics.md]. Written by G2P, read by P2G.
    phi_p: f32,
    // Packed occupied/home chunk slot id (low 24 bits) + material id (high 8 bits).
    home_chunk_slot_id: u32,
}

// Grid node layout.
struct GpuGridNode {
    // Water phase momentum / mass
    water_px: f32,
    water_py: f32,
    water_mass: f32,
    water_pad: f32,
    // Granular phase momentum / mass
    granular_px: f32,
    granular_py: f32,
    granular_mass: f32,
    granular_pad: f32,
}

// Simulation parameters (uniform).
struct MpmParams {
    // timestep (s)
    dt: f32,
    // gravity (m/s^2)
    gx: f32,
    gy: f32,
    // reference density (kg/m^3)
    rho_ref: f32,
    // bulk modulus (Pa)
    bulk_modulus: f32,
    // grid cell size (m)
    h: f32,
    // grid origin (node index)
    grid_origin_x: i32,
    grid_origin_y: i32,
    // grid dimensions in nodes
    grid_width: u32,
    grid_height: u32,
    // total particle count
    particle_count: u32,
    // deformation gradient clamps
    j_min: f32,
    j_max: f32,
    // APIC C matrix max norm (for clamping)
    c_max_norm: f32,
    // terrain boundary params
    sdf_velocity_threshold_m: f32,
    boundary_normal_projection_scale: f32,
    // Coulomb friction coefficient μ_b per material [Eq.29, physics.md]
    boundary_friction_water: f32,
    boundary_friction_granular: f32,
    // Drucker-Prager parameters (soil)
    dp_lambda_soil: f32,
    dp_mu_soil: f32,
    dp_alpha_soil: f32,
    dp_k_soil: f32,
    dp_hardening_soil: f32,
    // Drucker-Prager parameters (sand)
    dp_lambda_sand: f32,
    dp_mu_sand: f32,
    dp_alpha_sand: f32,
    dp_k_sand: f32,
    dp_hardening_sand: f32,
    // granular material / coupling params
    granular_tensile_clamp: f32,
    coupling_drag_gamma: f32,
    coupling_friction: f32,
    coupling_interface_min_grad: f32,
    coupling_interface_normal_eps: f32,
    // APIC<->PIC blend: 1.0=full APIC, 0.0=full PIC [Eq.32, physics.md]
    alpha_apic_water: f32,
    alpha_apic_granular: f32,
    // Statistics config
    stats_tracked_phase_id: u32,
    stats_repose_phase_id: u32,
    stats_interaction_primary_phase_id: u32,
    stats_interaction_secondary_phase_id: u32,
    stats_penetration_epsilon_m: f32,
    stats_position_fp_scale: f32,
    // Static chunk residency metadata.
    chunk_origin_x: i32,
    chunk_origin_y: i32,
    chunk_dims_x: u32,
    chunk_dims_y: u32,
    resident_chunk_count: u32,
    chunk_node_dim: u32,
    // padding
    pad_a: u32,
}

const PHASE_WATER: u32 = 0u;
const PHASE_GRANULAR_SOIL: u32 = 1u;
const PHASE_GRANULAR_SAND: u32 = 2u;
const PARTICLE_HOME_SLOT_MASK: u32 = 0x00FFFFFFu;
const PARTICLE_HOME_MATERIAL_SHIFT: u32 = 24u;
const INVALID_PARTICLE_SLOT: u32 = PARTICLE_HOME_SLOT_MASK;

fn phase_is_granular(phase_id: u32) -> bool {
    return phase_id == PHASE_GRANULAR_SOIL || phase_id == PHASE_GRANULAR_SAND;
}

fn particle_slot_id(p: GpuParticle) -> u32 {
    return p.home_chunk_slot_id & PARTICLE_HOME_SLOT_MASK;
}

fn particle_material_id(p: GpuParticle) -> u32 {
    return p.home_chunk_slot_id >> PARTICLE_HOME_MATERIAL_SHIFT;
}

fn repack_particle_home_slot(packed_home: u32, slot_id: u32) -> u32 {
    return (packed_home & ~PARTICLE_HOME_SLOT_MASK) | (slot_id & PARTICLE_HOME_SLOT_MASK);
}

fn pack_particle_home_slot(slot_id: u32, material_id: u32) -> u32 {
    return (material_id << PARTICLE_HOME_MATERIAL_SHIFT) | (slot_id & PARTICLE_HOME_SLOT_MASK);
}

// Quadratic B-spline kernel weight and gradient for 1D distance.
fn bspline_w_dw(dist: f32) -> vec2<f32> {
    let abs_d = abs(dist);
    if abs_d < 0.5 {
        let w = 0.75 - abs_d * abs_d;
        let dw = -2.0 * dist;
        return vec2<f32>(w, dw);
    } else if abs_d < 1.5 {
        let t = 1.5 - abs_d;
        let w = 0.5 * t * t;
        let dw = -sign(dist) * t;
        return vec2<f32>(w, dw);
    }
    return vec2<f32>(0.0, 0.0);
}

// Resident node capacity in slot-major layout.
fn node_capacity(params: MpmParams) -> u32 {
    let nodes_per_chunk = params.chunk_node_dim * params.chunk_node_dim;
    return params.resident_chunk_count * nodes_per_chunk;
}

fn chunk_node_dim_i(params: MpmParams) -> i32 {
    return i32(params.chunk_node_dim);
}

fn node_origin_from_chunks(params: MpmParams) -> vec2<i32> {
    let cdim = chunk_node_dim_i(params);
    return vec2<i32>(params.chunk_origin_x * cdim, params.chunk_origin_y * cdim);
}

fn node_extent_from_chunks(params: MpmParams) -> vec2<i32> {
    let cdim = chunk_node_dim_i(params);
    return vec2<i32>(i32(params.chunk_dims_x) * cdim, i32(params.chunk_dims_y) * cdim);
}

// Check if global node coordinate is within resident chunk bounds.
fn node_in_bounds(gx: i32, gy: i32, params: MpmParams) -> bool {
    if params.chunk_node_dim == 0u || params.chunk_dims_x == 0u || params.chunk_dims_y == 0u {
        return false;
    }
    let origin = node_origin_from_chunks(params);
    let extent = node_extent_from_chunks(params);
    let lx = gx - origin.x;
    let ly = gy - origin.y;
    return lx >= 0 && ly >= 0 && lx < extent.x && ly < extent.y;
}

// Node flat index from global node coordinates.
// Slot-major layout: slot_id * (chunk_node_dim^2) + local_node_index.
fn node_index(gx: i32, gy: i32, params: MpmParams) -> u32 {
    let cdim = chunk_node_dim_i(params);
    let origin = node_origin_from_chunks(params);
    let lx = gx - origin.x;
    let ly = gy - origin.y;

    let chunk_lx = u32(lx / cdim);
    let chunk_ly = u32(ly / cdim);
    let local_x = u32(lx - i32(chunk_lx) * cdim);
    let local_y = u32(ly - i32(chunk_ly) * cdim);

    let slot_id = chunk_ly * params.chunk_dims_x + chunk_lx;
    let nodes_per_chunk = params.chunk_node_dim * params.chunk_node_dim;
    return slot_id * nodes_per_chunk + local_y * params.chunk_node_dim + local_x;
}

// Determinant of a 2x2 matrix stored as (m00, m01, m10, m11).
fn mat2_det(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
    return m00 * m11 - m01 * m10;
}
