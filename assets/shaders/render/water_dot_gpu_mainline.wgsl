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
    world_dot_origin_x: i32,
    world_dot_origin_y: i32,
    palette_seed: u32,
    _pad1: u32,
}

struct WaterDotPalette {
    water: array<vec4<f32>, 4>,
    stone: array<vec4<f32>, 4>,
    soil: array<vec4<f32>, 4>,
    sand: array<vec4<f32>, 4>,
    grass: array<vec4<f32>, 4>,
}

const WINNER_MATERIAL_MASK: u32 = 0xffu;
const WINNER_EMPTY: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: WaterDotParams;
@group(0) @binding(2) var<storage, read> coverage_atomic: array<u32>;
@group(0) @binding(3) var<storage, read> winner_packed: array<u32>;
@group(0) @binding(4) var<uniform> palette: WaterDotPalette;

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

fn palette_color(material_id: u32, x: i32, y: i32, seed: u32) -> vec3<f32> {
    if material_id == 0u {
        return palette.water[deterministic_palette_index(x, y, seed)].rgb;
    }
    if material_id == 2u {
        return palette.stone[deterministic_palette_index(x + 13, y + 29, seed ^ 0x2b1du)].rgb;
    }
    if material_id == 4u {
        return palette.soil[deterministic_palette_index(x + 31, y - 17, seed ^ 0x59du)].rgb;
    }
    if material_id == 6u {
        return palette.sand[deterministic_palette_index(x - 19, y + 23, seed ^ 0x7f4au)].rgb;
    }
    return palette.grass[deterministic_palette_index(x - 7, y + 41, seed ^ 0x4c91u)].rgb;
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

    let idx = dot_index(dot_cell.x, dot_cell.y);
    let coverage = f32(coverage_atomic[idx]) / max(params.atomic_scale, 1.0e-6);
    if coverage < params.density_threshold {
        discard;
    }
    let winner = winner_packed[idx];
    if winner == WINNER_EMPTY {
        discard;
    }

    let local = fract(dot_pos) - vec2<f32>(0.5, 0.5);
    let edge = max(abs(local.x), abs(local.y));
    if edge >= 0.5 {
        discard;
    }

    let material_id = winner & WINNER_MATERIAL_MASK;
    let world_dot = vec2<i32>(
        params.world_dot_origin_x + dot_cell.x,
        params.world_dot_origin_y + dot_cell.y,
    );
    let color = palette_color(material_id, world_dot.x, world_dot.y, params.palette_seed);
    return vec4<f32>(color, 1.0);
}
