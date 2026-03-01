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

@group(0) @binding(1) var<uniform> params: MpmParams;
@group(0) @binding(2) var<storage, read> particles: array<GpuParticle>;

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    local_xy: vec2<f32>,
    @location(1)
    color: vec4<f32>,
    @location(2)
    is_background: f32,
};

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    if vertex_index == 0u { return vec2<f32>(-1.0, -1.0); }
    if vertex_index == 1u { return vec2<f32>( 1.0, -1.0); }
    if vertex_index == 2u { return vec2<f32>( 1.0,  1.0); }
    if vertex_index == 3u { return vec2<f32>(-1.0, -1.0); }
    if vertex_index == 4u { return vec2<f32>( 1.0,  1.0); }
    return vec2<f32>(-1.0,  1.0);
}

fn particle_color(material_id: u32) -> vec4<f32> {
    if material_id == 0u {
        return vec4<f32>(0.10, 0.80, 0.95, 0.88);
    }
    return vec4<f32>(0.80, 0.65, 0.36, 0.88);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    let corner = quad_corner(vertex_index);

    var out: VertexOut;
    out.local_xy = corner;

    // Always draw background tint as instance 0.
    if instance_index == 0u {
        out.clip_position = vec4<f32>(corner.x, corner.y, 0.0, 1.0);
        out.color = vec4<f32>(0.22, 0.32, 0.46, 0.40);
        out.is_background = 1.0;
        return out;
    }

    let p = particles[instance_index - 1u];
    let r = max(params.h * 0.35, 0.01);
    let world_xy = p.x + corner * r;

    out.clip_position = view.clip_from_world * vec4<f32>(world_xy.x, world_xy.y, 0.0, 1.0);
    out.color = particle_color(p.material_id);
    out.is_background = 0.0;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if in.is_background > 0.5 {
        return in.color;
    }

    let d = length(in.local_xy);
    if d > 1.0 {
        discard;
    }
    let edge = smoothstep(1.0, 0.88, d);
    return vec4<f32>(in.color.rgb, in.color.a * edge);
}
