#import bevy_render::view::View
#import particles::mpm_types::MpmParams

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: MpmParams;
@group(0) @binding(2) var<storage, read> terrain_sdf: array<f32>;

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    local_uv: vec2<f32>,
};

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    if vertex_index == 0u { return vec2<f32>(0.0, 0.0); }
    if vertex_index == 1u { return vec2<f32>(1.0, 0.0); }
    if vertex_index == 2u { return vec2<f32>(1.0, 1.0); }
    if vertex_index == 3u { return vec2<f32>(0.0, 0.0); }
    if vertex_index == 4u { return vec2<f32>(1.0, 1.0); }
    return vec2<f32>(0.0, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let uv = quad_corner(vertex_index);
    let grid_w = max(f32(params.grid_width), 1.0);
    let grid_h = max(f32(params.grid_height), 1.0);
    let span = vec2<f32>(max(grid_w - 1.0, 1.0), max(grid_h - 1.0, 1.0));
    let grid_min = vec2<f32>(f32(params.grid_origin_x), f32(params.grid_origin_y)) * params.h;
    let grid_max = grid_min + span * params.h;
    let world_xy = mix(grid_min, grid_max, uv);

    var out: VertexOut;
    out.clip_position = view.clip_from_world * vec4<f32>(world_xy, 0.0, 1.0);
    out.local_uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if params.grid_width == 0u || params.grid_height == 0u {
        discard;
    }

    let grid_w_f = f32(params.grid_width);
    let grid_h_f = f32(params.grid_height);
    let gx = u32(clamp(round(in.local_uv.x * max(grid_w_f - 1.0, 0.0)), 0.0, grid_w_f - 1.0));
    let gy = u32(clamp(round(in.local_uv.y * max(grid_h_f - 1.0, 0.0)), 0.0, grid_h_f - 1.0));
    let idx = gy * params.grid_width + gx;

    let sdf = terrain_sdf[idx];
    let range_m = max(params.h * 6.0, 1.0e-4);
    if sdf <= 0.0 {
        let depth = clamp((-sdf) / range_m, 0.0, 1.0);
        return vec4<f32>(
            0.92,
            0.22 + 0.30 * depth,
            0.10 + 0.12 * depth,
            0.24 + 0.54 * depth
        );
    }

    let sdf_norm = clamp(sdf / range_m, -1.0, 1.0);
    if sdf_norm >= 1.0 {
        discard;
    }
    let near = 1.0 - sdf_norm;
    return vec4<f32>(0.10, 0.72, 0.95, 0.06 + 0.24 * near);
}
