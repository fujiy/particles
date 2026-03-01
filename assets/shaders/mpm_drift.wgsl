// Debug compute pass: move particles at a constant velocity on GPU.
// This path intentionally bypasses MLS-MPM (clear/p2g/grid_update/g2p).

struct MpmParams {
    dt: f32,
    gx: f32,
    gy: f32,
    rho0: f32,
    bulk_modulus: f32,
    h: f32,
    grid_origin_x: i32,
    grid_origin_y: i32,
    grid_width: u32,
    grid_height: u32,
    particle_count: u32,
    j_min: f32,
    j_max: f32,
    c_max_norm: f32,
    sdf_velocity_threshold_m: f32,
    deep_push_gain_per_s: f32,
    deep_push_speed_cap_mps: f32,
    tangential_damping: f32,
    _pad0: u32,
    _pad1: u32,
}

struct GpuParticle {
    x: vec2<f32>,
    v: vec2<f32>,
    mass: f32,
    v_0: f32,
    f_00: f32,
    f_01: f32,
    f_10: f32,
    f_11: f32,
    c_00: f32,
    c_01: f32,
    c_10: f32,
    c_11: f32,
    material_id: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> particles: array<GpuParticle>;

const DEBUG_VEL: vec2<f32> = vec2<f32>(0.35, 0.0);

@compute @workgroup_size(64)
fn drift_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }

    var p = particles[pid];
    p.x = p.x + DEBUG_VEL * params.dt;
    p.v = DEBUG_VEL;
    particles[pid] = p;
}
