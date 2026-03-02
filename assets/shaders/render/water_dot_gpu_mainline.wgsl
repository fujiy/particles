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

struct WaterDotPalette {
    water: array<vec4<f32>, 4>,
    soil: array<vec4<f32>, 4>,
    sand: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: WaterDotParams;
@group(0) @binding(2) var<storage, read> blurred_density_water: array<f32>;
@group(0) @binding(3) var<storage, read> blurred_density_soil: array<f32>;
@group(0) @binding(4) var<storage, read> blurred_density_sand: array<f32>;
@group(0) @binding(5) var<uniform> palette: WaterDotPalette;

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

fn sample_density_water(dot_cell: vec2<i32>) -> f32 {
    let x = clamp(dot_cell.x, 0, i32(params.width) - 1);
    let y = clamp(dot_cell.y, 0, i32(params.height) - 1);
    return blurred_density_water[dot_index(x, y)];
}

fn sample_density_soil(dot_cell: vec2<i32>) -> f32 {
    let x = clamp(dot_cell.x, 0, i32(params.width) - 1);
    let y = clamp(dot_cell.y, 0, i32(params.height) - 1);
    return blurred_density_soil[dot_index(x, y)];
}

fn sample_density_sand(dot_cell: vec2<i32>) -> f32 {
    let x = clamp(dot_cell.x, 0, i32(params.width) - 1);
    let y = clamp(dot_cell.y, 0, i32(params.height) - 1);
    return blurred_density_sand[dot_index(x, y)];
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

fn deterministic_rand01(x: i32, y: i32, seed: u32) -> f32 {
    var state = (u32(x) * 0x9e3779b9u);
    state = state ^ (u32(y) * 0x85ebca6bu);
    state = state ^ seed;
    state = state ^ (state >> 15u);
    state = state * 0x2c1b3c6du;
    state = state ^ (state >> 12u);
    state = state * 0x297a2d39u;
    state = state ^ (state >> 15u);
    return f32(state & 0x00ffffffu) / 16777215.0;
}

fn water_palette_color(x: i32, y: i32, seed: u32) -> vec3<f32> {
    let idx = deterministic_palette_index(x, y, seed);
    return palette.water[idx].rgb;
}

fn soil_palette_color(x: i32, y: i32, seed: u32) -> vec3<f32> {
    let idx = deterministic_palette_index(x + 31, y - 17, seed ^ 0x59du);
    return palette.soil[idx].rgb;
}

fn sand_palette_color(x: i32, y: i32, seed: u32) -> vec3<f32> {
    let idx = deterministic_palette_index(x - 19, y + 23, seed ^ 0x7f4au);
    return palette.sand[idx].rgb;
}

fn choose_phase(
    water_density: f32,
    soil_density: f32,
    sand_density: f32,
    x: i32,
    y: i32,
    seed: u32,
) -> u32 {
    let sum_density = max(water_density + soil_density + sand_density, 1.0e-6);
    let r = deterministic_rand01(x, y, seed ^ 0x13a7u);
    let water_ratio = water_density / sum_density;
    let soil_ratio = soil_density / sum_density;
    if r < water_ratio {
        return 0u;
    }
    if r < water_ratio + soil_ratio {
        return 1u;
    }
    return 2u;
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
    let water_density = sample_density_water(dot_cell);
    let soil_density = sample_density_soil(dot_cell);
    let sand_density = sample_density_sand(dot_cell);
    let total_density = water_density + soil_density + sand_density;
    if total_density < params.density_threshold {
        discard;
    }

    // Keep each dot shape (square) with hard edge to avoid dark seam lines.
    let local = fract(dot_pos) - vec2<f32>(0.5, 0.5);
    let edge = max(abs(local.x), abs(local.y));
    if edge >= 0.5 {
        discard;
    }

    let phase = choose_phase(
        water_density,
        soil_density,
        sand_density,
        dot_cell.x,
        dot_cell.y,
        params.palette_seed,
    );
    var color = water_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    if phase == 1u {
        color = soil_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    } else if phase == 2u {
        color = sand_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    }
    return vec4<f32>(color, 1.0);
}
