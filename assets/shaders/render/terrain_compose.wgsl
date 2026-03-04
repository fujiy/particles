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

@group(0) @binding(0) var<uniform> view:    View;
@group(0) @binding(1) var<uniform> params:  ComposeParams;
@group(0) @binding(2) var          near_tex: texture_2d<u32>;
@group(0) @binding(3) var          far_tex:  texture_2d<u32>;
@group(0) @binding(4) var          back_tex: texture_2d<u32>;

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

fn srgb8(r: f32, g: f32, b: f32) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear(r / 255.0),
                     srgb_to_linear(g / 255.0),
                     srgb_to_linear(b / 255.0));
}

fn deterministic_palette_index(x: i32, y: i32, seed: u32) -> u32 {
    var s = u32(x) * 0x45d9f3bu;
    s = s ^ (u32(y) * 0x27d4eb2du);
    s = s ^ seed;
    s = s ^ (s >> 16u);
    s = s * 0x7feb352du;
    s = s ^ (s >> 15u);
    s = s * 0x846ca68bu;
    s = s ^ (s >> 16u);
    return s & 0x3u;
}

fn stone_color(idx: u32) -> vec3<f32> {
    if idx == 0u { return srgb8( 70.0,  67.0,  63.0); }
    if idx == 1u { return srgb8( 83.0,  79.0,  74.0); }
    if idx == 2u { return srgb8( 95.0,  90.0,  84.0); }
    return          srgb8(108.0, 103.0,  96.0);
}

fn soil_color(idx: u32) -> vec3<f32> {
    if idx == 0u { return srgb8(105.0,  79.0,  56.0); }
    if idx == 1u { return srgb8(119.0,  91.0,  67.0); }
    if idx == 2u { return srgb8(133.0, 103.0,  78.0); }
    return          srgb8(147.0, 115.0,  88.0);
}

fn sand_color(idx: u32) -> vec3<f32> {
    if idx == 0u { return srgb8(170.0, 150.0, 110.0); }
    if idx == 1u { return srgb8(186.0, 166.0, 124.0); }
    if idx == 2u { return srgb8(201.0, 181.0, 137.0); }
    return          srgb8(216.0, 196.0, 150.0);
}

fn terrain_color(material: u32, dot_x: i32, dot_y: i32, seed: u32) -> vec3<f32> {
    let shade = deterministic_palette_index(dot_x, dot_y, seed);
    if material == 1u { return stone_color(shade); }
    if material == 2u { return soil_color(shade); }
    return sand_color(shade);
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
    (*inout_color) = (*inout_color) + terrain_color(material, dot.x, dot.y, seed) * weighted;
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

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // World position → global cell coordinate (world_xy / cell_size_m, floored).
    let cell_f = in.world_xy / params.cell_size_m;
    let world_cell = vec2<i32>(i32(floor(cell_f.x)), i32(floor(cell_f.y)));

    // Dot coordinates for palette variation (8×8 sub-cell grid, global indexing).
    let dot_f = in.world_xy / params.dot_size_m;
    let dot = vec2<i32>(i32(floor(dot_f.x)), i32(floor(dot_f.y)));

    // Continuous cell-space coordinate (not quantized to integer cell centers).
    // This avoids visible cell-step jitter in distant layers while keeping dot pattern stable.
    let world_cell_coord = cell_f;
    let far_layer = select(
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
    let front_mpp_cells_per_px = max(params.front_mpp_cells_per_px, 1.0e-6);
    let scale_excess =
        params.back_scale_multiplier
            / max(params.back_min_screen_resolution_divisor * front_mpp_cells_per_px, 1.0e-6);
    // Asymptotically approaches Far (1.0) toward zoom-out; never hard-clamps at 1.
    let back_scale = 1.0 + max(scale_excess, 0.0);
    // Derive camera pivot from the same view matrix used for world_xy, so Back
    // scale/pan stays frame-coherent with zoom and avoids 1-frame drift.
    let camera_world = view.world_from_clip * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let camera_cell = (camera_world.xy / max(camera_world.w, 1.0e-6)) / params.cell_size_m;
    let back_world_cell_coord = camera_cell + (world_cell_coord - camera_cell) * back_scale;
    // Back dot phase uses sub-cell indexing to avoid 1-cell stepwise motion during pan.
    // At high zoom this still stays <= Near-sized dots because Back itself is scaled down.
    let back_dots_per_cell = f32(max(params.dots_per_cell, 1u));
    let back_dot = vec2<i32>(
        i32(floor(back_world_cell_coord.x * back_dots_per_cell)),
        i32(floor(back_world_cell_coord.y * back_dots_per_cell)),
    );

    let back_layer = sample_layer(
        back_tex,
        vec2<i32>(params.back_origin_x, params.back_origin_y),
        params.back_downsample,
        // Keep Back texel magnification capped to ~1 dot (2 px) while preserving Far-like behavior.
        back_world_cell_coord,
        back_dot,
        params.palette_seed,
    );

    let sky_color = srgb8(params.sky_color_r, params.sky_color_g, params.sky_color_b);
    var color = sky_color;
    // Air-perspective tint for distant backdrop.
    let back_tinted = layer_over(back_layer.rgb, sky_color, params.back_atmosphere_tint);
    color = layer_over(color, back_tinted, back_layer.a);
    color = layer_over(color, far_layer.rgb, far_layer.a);

    // Near overlays Far when enabled.
    let near_logical = world_cell - vec2<i32>(params.near_origin_x, params.near_origin_y);
    let near_size = vec2<i32>(textureDimensions(near_tex));
    if params.near_enabled != 0u
        && near_logical.x >= 0 && near_logical.y >= 0
        && near_logical.x < near_size.x && near_logical.y < near_size.y {
        let tc = vec2<i32>(
            positive_mod(near_logical.x + params.ring_offset_x, near_size.x),
            positive_mod(near_logical.y + params.ring_offset_y, near_size.y),
        );
        let near_material = textureLoad(near_tex, tc, 0).r;
        if near_material != 0u {
            color = terrain_color(near_material, dot.x, dot.y, params.palette_seed);
        }
    }

    return vec4<f32>(color, 1.0);
}
