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
    stone: array<vec4<f32>, 4>,
    soil: array<vec4<f32>, 4>,
    sand: array<vec4<f32>, 4>,
    grass: array<vec4<f32>, 4>,
}

struct TerrainComposeParams {
    cell_size_m: f32,
    dot_size_m: f32,
    back_min_screen_resolution_divisor: f32,
    back_scale_multiplier: f32,
    front_mpp_cells_per_px: f32,
    back_atmosphere_tint: f32,
    far_origin_x: i32,
    far_origin_y: i32,
    back_origin_x: i32,
    back_origin_y: i32,
    near_origin_x: i32,
    near_origin_y: i32,
    palette_seed: u32,
    dots_per_cell: u32,
    ring_offset_x: i32,
    ring_offset_y: i32,
    near_enabled: u32,
    far_downsample: u32,
    back_downsample: u32,
    _pad1: u32,
    sky_color_r: f32,
    sky_color_g: f32,
    sky_color_b: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: WaterDotParams;
@group(0) @binding(2) var<storage, read> blurred_density_water: array<f32>;
@group(0) @binding(3) var<storage, read> blurred_density_stone: array<f32>;
@group(0) @binding(4) var<storage, read> blurred_density_soil: array<f32>;
@group(0) @binding(5) var<storage, read> blurred_density_sand: array<f32>;
@group(0) @binding(6) var<storage, read> blurred_density_grass: array<f32>;
@group(0) @binding(7) var<uniform> palette: WaterDotPalette;
@group(0) @binding(8) var<uniform> terrain_params: TerrainComposeParams;
@group(0) @binding(9) var near_base_tex: texture_2d<u32>;
@group(0) @binding(10) var near_override_tex: texture_2d<u32>;

const NEAR_OVERRIDE_EMPTY_SENTINEL: u32 = 65535u;

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

fn positive_mod(x: i32, m: i32) -> i32 {
    let r = x % m;
    return select(r + m, r, r >= 0);
}

fn terrain_occludes(world_cell: vec2<i32>) -> bool {
    if terrain_params.near_enabled == 0u {
        return false;
    }
    let near_size = vec2<i32>(textureDimensions(near_base_tex));
    let near_logical = world_cell - vec2<i32>(terrain_params.near_origin_x, terrain_params.near_origin_y);
    if near_logical.x < 0 || near_logical.y < 0 || near_logical.x >= near_size.x || near_logical.y >= near_size.y {
        return false;
    }
    let tc = vec2<i32>(
        positive_mod(near_logical.x + terrain_params.ring_offset_x, near_size.x),
        positive_mod(near_logical.y + terrain_params.ring_offset_y, near_size.y),
    );
    let near_override = textureLoad(near_override_tex, tc, 0).r;
    let near_base = textureLoad(near_base_tex, tc, 0).r;
    let near_material = select(near_base, near_override, near_override != NEAR_OVERRIDE_EMPTY_SENTINEL);
    return near_material != 0u;
}

fn sample_density_water(dot_cell: vec2<i32>) -> f32 {
    let x = clamp(dot_cell.x, 0, i32(params.width) - 1);
    let y = clamp(dot_cell.y, 0, i32(params.height) - 1);
    return blurred_density_water[dot_index(x, y)];
}

fn sample_density_stone(dot_cell: vec2<i32>) -> f32 {
    let x = clamp(dot_cell.x, 0, i32(params.width) - 1);
    let y = clamp(dot_cell.y, 0, i32(params.height) - 1);
    return blurred_density_stone[dot_index(x, y)];
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

fn sample_density_grass(dot_cell: vec2<i32>) -> f32 {
    let x = clamp(dot_cell.x, 0, i32(params.width) - 1);
    let y = clamp(dot_cell.y, 0, i32(params.height) - 1);
    return blurred_density_grass[dot_index(x, y)];
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

fn stone_palette_color(x: i32, y: i32, seed: u32) -> vec3<f32> {
    let idx = deterministic_palette_index(x + 13, y + 29, seed ^ 0x2b1du);
    return palette.stone[idx].rgb;
}

fn soil_palette_color(x: i32, y: i32, seed: u32) -> vec3<f32> {
    let idx = deterministic_palette_index(x + 31, y - 17, seed ^ 0x59du);
    return palette.soil[idx].rgb;
}

fn sand_palette_color(x: i32, y: i32, seed: u32) -> vec3<f32> {
    let idx = deterministic_palette_index(x - 19, y + 23, seed ^ 0x7f4au);
    return palette.sand[idx].rgb;
}

fn grass_palette_color(x: i32, y: i32, seed: u32) -> vec3<f32> {
    let idx = deterministic_palette_index(x - 7, y + 41, seed ^ 0x4c91u);
    return palette.grass[idx].rgb;
}

fn choose_phase(
    water_density: f32,
    stone_density: f32,
    soil_density: f32,
    sand_density: f32,
    grass_density: f32,
    x: i32,
    y: i32,
    seed: u32,
) -> u32 {
    let sum_density =
        max(water_density + stone_density + soil_density + sand_density + grass_density, 1.0e-6);
    let r = deterministic_rand01(x, y, seed ^ 0x13a7u);
    let water_ratio = water_density / sum_density;
    let stone_ratio = stone_density / sum_density;
    let soil_ratio = soil_density / sum_density;
    let sand_ratio = sand_density / sum_density;
    if r < water_ratio {
        return 0u;
    }
    if r < water_ratio + stone_ratio {
        return 1u;
    }
    if r < water_ratio + stone_ratio + soil_ratio {
        return 2u;
    }
    if r < water_ratio + stone_ratio + soil_ratio + sand_ratio {
        return 3u;
    }
    return 4u;
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
    let world_cell = vec2<i32>(
        i32(floor(in.world_xy.x / terrain_params.cell_size_m)),
        i32(floor(in.world_xy.y / terrain_params.cell_size_m)),
    );

    // Sample density per dot-cell (nearest) so edge decision is dot-stable.
    if terrain_occludes(world_cell) {
        discard;
    }
    let water_density = sample_density_water(dot_cell);
    let stone_density = sample_density_stone(dot_cell);
    let soil_density = sample_density_soil(dot_cell);
    let sand_density = sample_density_sand(dot_cell);
    let grass_density = sample_density_grass(dot_cell);
    let total_density = water_density + stone_density + soil_density + sand_density + grass_density;
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
        stone_density,
        soil_density,
        sand_density,
        grass_density,
        dot_cell.x,
        dot_cell.y,
        params.palette_seed,
    );
    var color = water_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    if phase == 1u {
        color = stone_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    } else if phase == 2u {
        color = soil_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    } else if phase == 3u {
        color = sand_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    } else if phase == 4u {
        color = grass_palette_color(dot_cell.x, dot_cell.y, params.palette_seed);
    }
    return vec4<f32>(color, 1.0);
}
