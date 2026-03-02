// MLS-MPM shared type definitions.
// Imported by each compute shader via #import.
#define_import_path particles::mpm_types

// Particle storage layout (one array of GpuParticle, 72 bytes each).
// Matches Rust GpuParticle: x(8)+v(8)+mass(4)+v0(4)+f(16)+c(16)+v_vol(4)+phase_id(4)+pad(8)=72
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
    // padding to 72 bytes
    pad_b: u32,
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
    // Coulomb friction coefficient μ_b per material [Eq.29, physics.md]
    boundary_friction_water: f32,
    boundary_friction_granular: f32,
    pad_friction: u32,
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
    // padding
    pad_a: u32,
}

const PHASE_WATER: u32 = 0u;
const PHASE_GRANULAR_SOIL: u32 = 1u;
const PHASE_GRANULAR_SAND: u32 = 2u;

fn phase_is_granular(phase_id: u32) -> bool {
    return phase_id == PHASE_GRANULAR_SOIL || phase_id == PHASE_GRANULAR_SAND;
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

// Node flat index from grid coordinates.
fn node_index(gx: i32, gy: i32, params: MpmParams) -> u32 {
    let lx = gx - params.grid_origin_x;
    let ly = gy - params.grid_origin_y;
    return u32(ly) * params.grid_width + u32(lx);
}

// Check if grid coordinate is within bounds.
fn node_in_bounds(gx: i32, gy: i32, params: MpmParams) -> bool {
    let lx = gx - params.grid_origin_x;
    let ly = gy - params.grid_origin_y;
    return lx >= 0 && ly >= 0
        && u32(lx) < params.grid_width
        && u32(ly) < params.grid_height;
}

// Determinant of a 2x2 matrix stored as (m00, m01, m10, m11).
fn mat2_det(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
    return m00 * m11 - m01 * m10;
}
