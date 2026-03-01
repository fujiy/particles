#import bevy_render::view::View

struct WaterDotParams {
    origin_x: f32,
    origin_y: f32,
    dot_size_m: f32,
    density_threshold: f32,
    atomic_scale: f32,
    splat_radius_dots: f32,
    blur_sigma_dots: f32,
    _pad0: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    blur_radius_dots: u32,
    palette_seed: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: WaterDotParams;
@group(0) @binding(2) var<storage, read> blurred_density: array<f32>;

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    world_xy: vec2<f32>,
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    if vertex_index == 0u { return vec2<f32>(-1.0, -1.0); }
    if vertex_index == 1u { return vec2<f32>( 1.0, -1.0); }
    if vertex_index == 2u { return vec2<f32>( 1.0,  1.0); }
    if vertex_index == 3u { return vec2<f32>(-1.0, -1.0); }
    if vertex_index == 4u { return vec2<f32>( 1.0,  1.0); }
    return vec2<f32>(-1.0,  1.0);
}

fn dot_index(x: i32, y: i32) -> u32 {
    return u32(y) * params.width + u32(x);
}

fn in_bounds(x: i32, y: i32) -> bool {
    return x >= 0 && y >= 0 && u32(x) < params.width && u32(y) < params.height;
}

fn sample_density(dot_cell: vec2<i32>) -> f32 {
    let x = clamp(dot_cell.x, 0, i32(params.width) - 1);
    let y = clamp(dot_cell.y, 0, i32(params.height) - 1);
    return blurred_density[dot_index(x, y)];
}

fn srgb_channel_to_linear(x: f32) -> f32 {
    if x <= 0.04045 {
        return x / 12.92;
    }
    return pow((x + 0.055) / 1.055, 2.4);
}

fn srgb8_to_linear(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_channel_to_linear(rgb.x),
        srgb_channel_to_linear(rgb.y),
        srgb_channel_to_linear(rgb.z),
    );
}

fn deterministic_palette_index(x: i32, y: i32, seed: u32) -> u32 {
    var state = (u32(x) * 0x45d9f3bu);
    state = state ^ (u32(y) * 0x27d4eb2du);
    state = state ^ seed;
    state = state ^ (state >> 16u);
    state = state * 0x7feb352du;
    state = state ^ (state >> 15u);
    state = state * 0x846ca68bu;
    state = state ^ (state >> 16u);
    return state & 0x3u;
}

fn water_palette_color(x: i32, y: i32, seed: u32) -> vec3<f32> {
    let idx = deterministic_palette_index(x, y, seed);
    if idx == 0u {
        return srgb8_to_linear(vec3<f32>(42.0 / 255.0, 120.0 / 255.0, 202.0 / 255.0));
    }
    if idx == 1u {
        return srgb8_to_linear(vec3<f32>(52.0 / 255.0, 136.0 / 255.0, 218.0 / 255.0));
    }
    if idx == 2u {
        return srgb8_to_linear(vec3<f32>(65.0 / 255.0, 152.0 / 255.0, 228.0 / 255.0));
    }
    return srgb8_to_linear(vec3<f32>(78.0 / 255.0, 167.0 / 255.0, 238.0 / 255.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let corner = quad_corner(vertex_index);

    var out: VertexOut;
    out.clip_position = vec4<f32>(corner, 0.0, 1.0);

    let world = view.world_from_clip * vec4<f32>(corner, 0.0, 1.0);
    out.world_xy = world.xy / max(world.w, 1.0e-6);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dot_pos = (in.world_xy - vec2<f32>(params.origin_x, params.origin_y)) / params.dot_size_m;

    let dot_cell = vec2<i32>(i32(floor(dot_pos.x)), i32(floor(dot_pos.y)));
    if !in_bounds(dot_cell.x, dot_cell.y) {
        discard;
    }

    // Sample density per dot-cell (nearest) so edge decision is dot-stable.
    let density = sample_density(dot_cell);
    if density < params.density_threshold {
        discard;
    }

    // Keep each dot shape (square) with hard edge to avoid dark seam lines.
    let local = fract(dot_pos) - vec2<f32>(0.5, 0.5);
    let edge = max(abs(local.x), abs(local.y));
    if edge >= 0.5 {
        discard;
    }

    let color = water_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    return vec4<f32>(color, 1.0);
}
