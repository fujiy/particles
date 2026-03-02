#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;
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
    dp_lambda_soil: f32,
    dp_mu_soil: f32,
    dp_alpha_soil: f32,
    dp_k_soil: f32,
    dp_hardening_soil: f32,
    dp_lambda_sand: f32,
    dp_mu_sand: f32,
    dp_alpha_sand: f32,
    dp_k_sand: f32,
    dp_hardening_sand: f32,
    granular_tensile_clamp: f32,
    coupling_normal_stiffness: f32,
    coupling_tangent_drag: f32,
    coupling_friction: f32,
    coupling_max_impulse_ratio: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
    jp: f32,
    phase_id: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(1) var<uniform> params: MpmParams;
@group(0) @binding(2) var<storage, read> particles: array<GpuParticle>;

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    local_xy: vec2<f32>,
    @location(1)
    color: vec4<f32>,
};

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    if vertex_index == 0u { return vec2<f32>(-1.0, -1.0); }
    if vertex_index == 1u { return vec2<f32>( 1.0, -1.0); }
    if vertex_index == 2u { return vec2<f32>( 1.0,  1.0); }
    if vertex_index == 3u { return vec2<f32>(-1.0, -1.0); }
    if vertex_index == 4u { return vec2<f32>( 1.0,  1.0); }
    return vec2<f32>(-1.0,  1.0);
}

fn particle_color(phase_id: u32) -> vec4<f32> {
    if phase_id == 0u {
        return vec4<f32>(0.30, 0.63, 0.95, 0.92);
    }
    if phase_id == 1u {
        return vec4<f32>(0.76, 0.56, 0.38, 0.94);
    }
    if phase_id == 2u {
        return vec4<f32>(0.86, 0.77, 0.56, 0.94);
    }
    return vec4<f32>(1.0, 1.0, 1.0, 0.92);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    let corner = quad_corner(vertex_index);

    var out: VertexOut;
    out.local_xy = corner;
    let p = particles[instance_index];
    let r = max(params.h * 0.35, 0.01);
    let world_xy = p.x + corner * r;

    out.clip_position = view.clip_from_world * vec4<f32>(world_xy.x, world_xy.y, 0.0, 1.0);
    out.color = particle_color(p.phase_id);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let d = length(in.local_xy);
    if d > 1.0 {
        discard;
    }
    let edge = smoothstep(1.0, 0.88, d);
    return vec4<f32>(in.color.rgb, in.color.a * edge);
}
