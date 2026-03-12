/// TerrainCompose fragment pass.
///
/// Reads material IDs from `near_tex` (R16Uint, one texel == one world cell) and
/// renders terrain with deterministic dot-art palette variation (8×8 dots/cell).
#import bevy_render::view::View

struct ComposeParams {
    cell_size_m:   f32,  // metres per cell (CELL_SIZE_M = 0.25)
    dot_size_m:    f32,  // metres per dot (cell_size_m / dots_per_cell)
    back_min_screen_resolution_divisor: f32,
    back_scale_multiplier: f32,
    front_mpp_cells_per_px: f32,
    back_atmosphere_tint: f32,
    far_origin_x:  i32,  // world cell X of far texel block (0,0)
    far_origin_y:  i32,  // world cell Y of far texel block (0,0)
    back_origin_x: i32,  // world cell X of back texel block (0,0)
    back_origin_y: i32,  // world cell Y of back texel block (0,0)
    near_origin_x: i32,  // world cell X of near texel (0,0)
    near_origin_y: i32,  // world cell Y of near texel (0,0)
    palette_seed:  u32,
    dots_per_cell: u32,
    ring_offset_x: i32,  // ring buffer offset in physical texture X
    ring_offset_y: i32,  // ring buffer offset in physical texture Y
    near_enabled:  u32,  // 0: disable near overlay, 1: enable
    far_downsample: u32, // world-cell pitch of far texel
    back_downsample: u32, // world-cell pitch of back texel
    _pad1:         u32,
    sky_color_r:   f32,   // sRGB 0..255
    sky_color_g:   f32,   // sRGB 0..255
    sky_color_b:   f32,   // sRGB 0..255
    _pad2:         f32,
}

struct TerrainPalette {
    stone: array<vec4<f32>, 4>,
    soil:  array<vec4<f32>, 4>,
    sand:  array<vec4<f32>, 4>,
    grass: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> view:    View;
@group(0) @binding(1) var<uniform> params:  ComposeParams;
@group(0) @binding(2) var<uniform> palette: TerrainPalette;
@group(0) @binding(3) var          near_base_tex: texture_2d<u32>;
@group(0) @binding(4) var          near_override_tex: texture_2d<u32>;
@group(0) @binding(5) var          far_tex:  texture_2d<u32>;
@group(0) @binding(6) var          back_tex: texture_2d<u32>;

const NEAR_OVERRIDE_EMPTY_SENTINEL: u32 = 65535u;
const GRASS_EDGE_DEPTH_BASE_DOTS: i32 = 4;
const GRASS_EDGE_DEPTH_VARIATION_DOTS: i32 = 1;
const SURFACE_NOISE_PERIOD_DOTS: f32 = 3.0;
const SURFACE_HOLE_THRESHOLD: f32 = 0.72;

// ── Vertex (full-screen quad) ─────────────────────────────────────────────────

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       world_xy:      vec2<f32>,
}

fn quad_corner(v: u32) -> vec2<f32> {
    if v == 0u { return vec2<f32>(-1.0, -1.0); }
    if v == 1u { return vec2<f32>( 1.0, -1.0); }
    if v == 2u { return vec2<f32>( 1.0,  1.0); }
    if v == 3u { return vec2<f32>(-1.0, -1.0); }
    if v == 4u { return vec2<f32>( 1.0,  1.0); }
    return vec2<f32>(-1.0,  1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    let c = quad_corner(vi);
    var out: VertexOut;
    out.clip_position = vec4<f32>(c, 0.0, 1.0);
    let w = view.world_from_clip * vec4<f32>(c, 0.0, 1.0);
    out.world_xy = w.xy / max(w.w, 1e-6);
    return out;
}

// ── Colour helpers ────────────────────────────────────────────────────────────

fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.04045 { return x / 12.92; }
    return pow((x + 0.055) / 1.055, 2.4);
}

fn deterministic_hash_u32(x: i32, y: i32, seed: u32) -> u32 {
    var s = u32(x) * 0x45d9f3bu;
    s = s ^ (u32(y) * 0x27d4eb2du);
    s = s ^ seed;
    s = s ^ (s >> 16u);
    s = s * 0x7feb352du;
    s = s ^ (s >> 15u);
    s = s * 0x846ca68bu;
    s = s ^ (s >> 16u);
    return s;
}

fn deterministic_palette_index(x: i32, y: i32, seed: u32) -> u32 {
    return deterministic_hash_u32(x, y, seed) & 0x3u;
}

fn hash_to_unit01(h: u32) -> f32 {
    return f32(h & 0x00ffffffu) / f32(0x01000000u);
}

fn smoothstep01(t: f32) -> f32 {
    let c = clamp(t, 0.0, 1.0);
    return c * c * (3.0 - 2.0 * c);
}

fn value_noise_2d(p: vec2<f32>, seed: u32) -> f32 {
    let x0 = i32(floor(p.x));
    let y0 = i32(floor(p.y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let tx = smoothstep01(p.x - f32(x0));
    let ty = smoothstep01(p.y - f32(y0));

    let v00 = hash_to_unit01(deterministic_hash_u32(x0, y0, seed));
    let v10 = hash_to_unit01(deterministic_hash_u32(x1, y0, seed));
    let v01 = hash_to_unit01(deterministic_hash_u32(x0, y1, seed));
    let v11 = hash_to_unit01(deterministic_hash_u32(x1, y1, seed));

    let vx0 = mix(v00, v10, tx);
    let vx1 = mix(v01, v11, tx);
    return mix(vx0, vx1, ty);
}

fn stone_color(idx: u32) -> vec3<f32> {
    return palette.stone[idx].rgb;
}

fn soil_color(idx: u32) -> vec3<f32> {
    return palette.soil[idx].rgb;
}

fn sand_color(idx: u32) -> vec3<f32> {
    return palette.sand[idx].rgb;
}

fn grass_color(idx: u32) -> vec3<f32> {
    return palette.grass[idx].rgb;
}

fn fill_palette_index(dot_x: i32, dot_y: i32, seed: u32) -> u32 {
    return deterministic_palette_index(dot_x, dot_y, seed);
}

fn grass_palette_index(dot_x: i32, dot_y: i32, seed: u32) -> u32 {
    return deterministic_palette_index(dot_x - 7, dot_y + 41, seed ^ 0x4c91u);
}

fn terrain_fill_color(material: u32, dot_x: i32, dot_y: i32, seed: u32) -> vec3<f32> {
    let shade = fill_palette_index(dot_x, dot_y, seed);
    if material == 1u { return stone_color(shade); }
    if material == 2u { return soil_color(shade); }
    if material == 3u { return sand_color(shade); }
    return soil_color(shade);
}

fn grass_edge_color(dot_x: i32, dot_y: i32, seed: u32) -> vec3<f32> {
    return grass_color(grass_palette_index(dot_x, dot_y, seed));
}

fn grass_depth_from_noise(noise: f32, dots_per_cell: i32) -> i32 {
    let centered = noise * 2.0 - 1.0;
    let offset = select(select(0, 1, centered > 0.35), -1, centered < -0.35);
    return clamp(
        GRASS_EDGE_DEPTH_BASE_DOTS + offset * GRASS_EDGE_DEPTH_VARIATION_DOTS,
        1,
        dots_per_cell,
    );
}

fn surface_hole_alpha(
    dot: vec2<i32>,
    left_surface: bool,
    right_surface: bool,
    bottom_surface: bool,
    top_surface: bool,
    seed: u32,
) -> f32 {
    if !(left_surface || right_surface || bottom_surface || top_surface) {
        return 1.0;
    }
    if (left_surface && top_surface)
        || (right_surface && top_surface)
        || (left_surface && bottom_surface)
        || (right_surface && bottom_surface)
    {
        return 0.0;
    }
    let noise = value_noise_2d(
        vec2<f32>(
            f32(dot.x) / SURFACE_NOISE_PERIOD_DOTS + 17.3,
            f32(dot.y) / SURFACE_NOISE_PERIOD_DOTS - 11.7,
        ),
        seed ^ 0x61c7u,
    );
    if noise >= SURFACE_HOLE_THRESHOLD {
        return 0.0;
    }
    return 1.0;
}

fn positive_mod(x: i32, m: i32) -> i32 {
    let r = x % m;
    return select(r + m, r, r >= 0);
}

fn sample_weighted_color(
    tex: texture_2d<u32>,
    texel: vec2<i32>,
    w: f32,
    dot: vec2<i32>,
    seed: u32,
    inout_color: ptr<function, vec3<f32>>,
    inout_alpha: ptr<function, f32>,
) {
    let sample = textureLoad(tex, texel, 0);
    let material = sample.r;
    if material == 0u {
        return;
    }
    let a = f32(sample.b) / 255.0;
    if a <= 0.0 {
        return;
    }
    let weighted = w * a;
    (*inout_color) = (*inout_color) + terrain_fill_color(material, dot.x, dot.y, seed) * weighted;
    (*inout_alpha) = (*inout_alpha) + weighted;
}

fn sample_layer(
    tex: texture_2d<u32>,
    origin: vec2<i32>,
    downsample: u32,
    world_cell_center: vec2<f32>,
    dot: vec2<i32>,
    seed: u32,
) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(tex));
    if size.x <= 0 || size.y <= 0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let downsample_f = f32(max(downsample, 1u));
    let coord =
        (world_cell_center - vec2<f32>(f32(origin.x), f32(origin.y)))
            / downsample_f
        - vec2<f32>(0.5, 0.5);
    let base = vec2<i32>(i32(floor(coord.x)), i32(floor(coord.y)));
    let frac = coord - vec2<f32>(f32(base.x), f32(base.y));
    let wx0 = 1.0 - clamp(frac.x, 0.0, 1.0);
    let wx1 = clamp(frac.x, 0.0, 1.0);
    let wy0 = 1.0 - clamp(frac.y, 0.0, 1.0);
    let wy1 = clamp(frac.y, 0.0, 1.0);

    var color_acc = vec3<f32>(0.0, 0.0, 0.0);
    var alpha_acc = 0.0;
    let tx00 = base;
    let tx10 = base + vec2<i32>(1, 0);
    let tx01 = base + vec2<i32>(0, 1);
    let tx11 = base + vec2<i32>(1, 1);
    let clamped00 = vec2<i32>(clamp(tx00.x, 0, size.x - 1), clamp(tx00.y, 0, size.y - 1));
    let clamped10 = vec2<i32>(clamp(tx10.x, 0, size.x - 1), clamp(tx10.y, 0, size.y - 1));
    let clamped01 = vec2<i32>(clamp(tx01.x, 0, size.x - 1), clamp(tx01.y, 0, size.y - 1));
    let clamped11 = vec2<i32>(clamp(tx11.x, 0, size.x - 1), clamp(tx11.y, 0, size.y - 1));
    sample_weighted_color(tex, clamped00, wx0 * wy0, dot, seed, &color_acc, &alpha_acc);
    sample_weighted_color(tex, clamped10, wx1 * wy0, dot, seed, &color_acc, &alpha_acc);
    sample_weighted_color(tex, clamped01, wx0 * wy1, dot, seed, &color_acc, &alpha_acc);
    sample_weighted_color(tex, clamped11, wx1 * wy1, dot, seed, &color_acc, &alpha_acc);

    if alpha_acc <= 1.0e-6 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(color_acc / alpha_acc, clamp(alpha_acc, 0.0, 1.0));
}

fn sample_layer_nearest(
    tex: texture_2d<u32>,
    origin: vec2<i32>,
    downsample: u32,
    world_cell_center: vec2<f32>,
    dot: vec2<i32>,
    seed: u32,
) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(tex));
    if size.x <= 0 || size.y <= 0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let downsample_f = f32(max(downsample, 1u));
    let coord =
        (world_cell_center - vec2<f32>(f32(origin.x), f32(origin.y)))
            / downsample_f
        - vec2<f32>(0.5, 0.5);
    let nearest = vec2<i32>(i32(round(coord.x)), i32(round(coord.y)));
    let clamped = vec2<i32>(
        clamp(nearest.x, 0, size.x - 1),
        clamp(nearest.y, 0, size.y - 1),
    );
    var color_acc = vec3<f32>(0.0, 0.0, 0.0);
    var alpha_acc = 0.0;
    sample_weighted_color(tex, clamped, 1.0, dot, seed, &color_acc, &alpha_acc);
    if alpha_acc <= 1.0e-6 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(color_acc / alpha_acc, clamp(alpha_acc, 0.0, 1.0));
}

fn layer_over(bg: vec3<f32>, fg: vec3<f32>, fg_alpha: f32) -> vec3<f32> {
    let a = clamp(fg_alpha, 0.0, 1.0);
    return bg * (1.0 - a) + fg * a;
}

fn near_material_at(world_cell: vec2<i32>) -> u32 {
    if params.near_enabled == 0u {
        return 0u;
    }
    let near_size = vec2<i32>(textureDimensions(near_base_tex));
    let near_logical = world_cell - vec2<i32>(params.near_origin_x, params.near_origin_y);
    if near_logical.x < 0
        || near_logical.y < 0
        || near_logical.x >= near_size.x
        || near_logical.y >= near_size.y
    {
        return 0u;
    }
    let tc = vec2<i32>(
        positive_mod(near_logical.x + params.ring_offset_x, near_size.x),
        positive_mod(near_logical.y + params.ring_offset_y, near_size.y),
    );
    let near_override = textureLoad(near_override_tex, tc, 0).r;
    if near_override != NEAR_OVERRIDE_EMPTY_SENTINEL {
        return near_override;
    }
    return textureLoad(near_base_tex, tc, 0).r;
}

fn near_contains_world_cell(world_cell: vec2<i32>) -> bool {
    if params.near_enabled == 0u {
        return false;
    }
    let near_size = vec2<i32>(textureDimensions(near_base_tex));
    let near_logical = world_cell - vec2<i32>(params.near_origin_x, params.near_origin_y);
    return near_logical.x >= 0
        && near_logical.y >= 0
        && near_logical.x < near_size.x
        && near_logical.y < near_size.y;
}

fn terrain_near_color(
    material: u32,
    world_cell: vec2<i32>,
    dot: vec2<i32>,
    seed: u32,
) -> vec4<f32> {
    let dots_per_cell = max(i32(params.dots_per_cell), 1);
    let cell_dot = vec2<i32>(
        positive_mod(dot.x, dots_per_cell),
        positive_mod(dot.y, dots_per_cell),
    );
    let left_neighbor_empty = near_material_at(world_cell + vec2<i32>(-1, 0)) == 0u;
    let right_neighbor_empty = near_material_at(world_cell + vec2<i32>(1, 0)) == 0u;
    let bottom_neighbor_empty = near_material_at(world_cell + vec2<i32>(0, -1)) == 0u;
    let top_neighbor_empty = near_material_at(world_cell + vec2<i32>(0, 1)) == 0u;
    let left_surface = left_neighbor_empty && cell_dot.x == 0;
    let right_surface = right_neighbor_empty && cell_dot.x == dots_per_cell - 1;
    let bottom_surface = bottom_neighbor_empty && cell_dot.y == 0;
    let top_surface = top_neighbor_empty && cell_dot.y == dots_per_cell - 1;
    let alpha = surface_hole_alpha(
        dot,
        left_surface,
        right_surface,
        bottom_surface,
        top_surface,
        seed,
    );
    if alpha <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if material != 4u {
        return vec4<f32>(terrain_fill_color(material, dot.x, dot.y, seed), alpha);
    }

    let left_edge_depth = grass_depth_from_noise(
        value_noise_2d(
            vec2<f32>(
                f32(dot.y) / SURFACE_NOISE_PERIOD_DOTS + 2.1,
                f32(world_cell.x) * 0.5 - 7.3,
            ),
            seed ^ 0x11a9u,
        ),
        dots_per_cell,
    );
    let right_edge_depth = grass_depth_from_noise(
        value_noise_2d(
            vec2<f32>(
                f32(dot.y) / SURFACE_NOISE_PERIOD_DOTS - 4.7,
                f32(world_cell.x) * 0.5 + 9.1,
            ),
            seed ^ 0x2f53u,
        ),
        dots_per_cell,
    );
    let bottom_edge_depth = grass_depth_from_noise(
        value_noise_2d(
            vec2<f32>(
                f32(dot.x) / SURFACE_NOISE_PERIOD_DOTS + 5.4,
                f32(world_cell.y) * 0.5 - 3.9,
            ),
            seed ^ 0x45d1u,
        ),
        dots_per_cell,
    );
    let top_edge_depth = grass_depth_from_noise(
        value_noise_2d(
            vec2<f32>(
                f32(dot.x) / SURFACE_NOISE_PERIOD_DOTS - 6.2,
                f32(world_cell.y) * 0.5 + 8.6,
            ),
            seed ^ 0x5b7du,
        ),
        dots_per_cell,
    );

    let left_exposed = left_neighbor_empty && cell_dot.x < left_edge_depth;
    let right_exposed =
        right_neighbor_empty && cell_dot.x >= dots_per_cell - right_edge_depth;
    let bottom_exposed = bottom_neighbor_empty && cell_dot.y < bottom_edge_depth;
    let top_exposed = top_neighbor_empty && cell_dot.y >= dots_per_cell - top_edge_depth;

    if left_exposed || right_exposed || bottom_exposed || top_exposed {
        return vec4<f32>(grass_edge_color(dot.x, dot.y, seed), alpha);
    }
    return vec4<f32>(soil_color(fill_palette_index(dot.x, dot.y, seed)), alpha);
}

fn terrain_background_layer(
    world_xy: vec2<f32>,
    world_cell: vec2<i32>,
    dot: vec2<i32>,
    cell_f: vec2<f32>,
) -> vec4<f32> {
    let world_cell_coord = cell_f;
    let near_contains_cell = near_contains_world_cell(world_cell);
    var far_layer = select(
        sample_layer(
            far_tex,
            vec2<i32>(params.far_origin_x, params.far_origin_y),
            params.far_downsample,
            world_cell_coord,
            dot,
            params.palette_seed,
        ),
        sample_layer_nearest(
            far_tex,
            vec2<i32>(params.far_origin_x, params.far_origin_y),
            params.far_downsample,
            world_cell_coord,
            dot,
            params.palette_seed,
        ),
        params.near_enabled != 0u,
    );
    if near_contains_cell {
        far_layer = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let front_mpp_cells_per_px = max(params.front_mpp_cells_per_px, 1.0e-6);
    let scale_excess =
        params.back_scale_multiplier
            / max(params.back_min_screen_resolution_divisor * front_mpp_cells_per_px, 1.0e-6);
    let back_scale = 1.0 + max(scale_excess, 0.0);
    let camera_world = view.world_from_clip * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let camera_cell = (camera_world.xy / max(camera_world.w, 1.0e-6)) / params.cell_size_m;
    let back_world_cell_coord = camera_cell + (world_cell_coord - camera_cell) * back_scale;
    let back_dots_per_cell = f32(max(params.dots_per_cell, 1u));
    let back_dot = vec2<i32>(
        i32(floor(back_world_cell_coord.x * back_dots_per_cell)),
        i32(floor(back_world_cell_coord.y * back_dots_per_cell)),
    );

    let back_layer = sample_layer(
        back_tex,
        vec2<i32>(params.back_origin_x, params.back_origin_y),
        params.back_downsample,
        back_world_cell_coord,
        back_dot,
        params.palette_seed,
    );

    let sky_color = vec3<f32>(
        srgb_to_linear(params.sky_color_r / 255.0),
        srgb_to_linear(params.sky_color_g / 255.0),
        srgb_to_linear(params.sky_color_b / 255.0),
    );
    var color = sky_color;
    let back_tinted = layer_over(back_layer.rgb, sky_color, params.back_atmosphere_tint);
    color = layer_over(color, back_tinted, back_layer.a);
    color = layer_over(color, far_layer.rgb, far_layer.a);
    return vec4<f32>(color, 1.0);
}

fn terrain_front_layer(
    world_cell: vec2<i32>,
    dot: vec2<i32>,
) -> vec4<f32> {
    if params.near_enabled == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let near_logical = world_cell - vec2<i32>(params.near_origin_x, params.near_origin_y);
    let near_size = vec2<i32>(textureDimensions(near_base_tex));
    if near_logical.x < 0 || near_logical.y < 0
        || near_logical.x >= near_size.x || near_logical.y >= near_size.y {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let tc = vec2<i32>(
        positive_mod(near_logical.x + params.ring_offset_x, near_size.x),
        positive_mod(near_logical.y + params.ring_offset_y, near_size.y),
    );
    let near_override = textureLoad(near_override_tex, tc, 0).r;
    let near_material = select(
        textureLoad(near_base_tex, tc, 0).r,
        near_override,
        near_override != NEAR_OVERRIDE_EMPTY_SENTINEL,
    );
    let has_override = near_override != NEAR_OVERRIDE_EMPTY_SENTINEL;
    if has_override {
        if near_material != 0u {
            return terrain_near_color(near_material, world_cell, dot, params.palette_seed);
        }
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if near_material != 0u {
        return terrain_near_color(near_material, world_cell, dot, params.palette_seed);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_back_main(in: VertexOut) -> @location(0) vec4<f32> {
    let cell_f = in.world_xy / params.cell_size_m;
    let world_cell = vec2<i32>(i32(floor(cell_f.x)), i32(floor(cell_f.y)));
    let dot_f = in.world_xy / params.dot_size_m;
    let dot = vec2<i32>(i32(floor(dot_f.x)), i32(floor(dot_f.y)));
    return terrain_background_layer(in.world_xy, world_cell, dot, cell_f);
}

@fragment
fn fs_front_main(in: VertexOut) -> @location(0) vec4<f32> {
    let cell_f = in.world_xy / params.cell_size_m;
    let world_cell = vec2<i32>(i32(floor(cell_f.x)), i32(floor(cell_f.y)));
    let dot_f = in.world_xy / params.dot_size_m;
    let dot = vec2<i32>(i32(floor(dot_f.x)), i32(floor(dot_f.y)));
    return terrain_front_layer(world_cell, dot);
}
