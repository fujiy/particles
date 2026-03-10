#import bevy_render::view::View
#import particles::mpm_types::{GpuParticle, MpmParams}

@group(0) @binding(0) var<uniform> view: View;
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
    let aa = max(fwidth(d), 1.0e-4);
    let inner_radius = max(1.0 - 2.5 * aa, 0.0);
    let outer_alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, d);
    let inner_alpha = smoothstep(inner_radius - aa, inner_radius + aa, d);
    let ring_alpha = outer_alpha * inner_alpha;
    if ring_alpha <= 1.0e-3 {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * ring_alpha);
}
