/// TerrainCompose fragment pass.
///
/// Reads material IDs from `near_tex` (R16Uint, one texel == one world cell) and
/// renders terrain with deterministic dot-art palette variation (8×8 dots/cell).
#import bevy_render::view::View

struct ComposeParams {
    cell_size_m:   f32,  // metres per cell (CELL_SIZE_M = 0.25)
    dot_size_m:    f32,  // metres per dot (cell_size_m / dots_per_cell)
    near_origin_x: i32,  // world cell X of near texel (0,0)
    near_origin_y: i32,  // world cell Y of near texel (0,0)
    far_origin_x:  i32,  // world cell X of far texel block (0,0)
    far_origin_y:  i32,  // world cell Y of far texel block (0,0)
    palette_seed:  u32,
    dots_per_cell: u32,
    ring_offset_x: i32,  // ring buffer offset in physical texture X
    ring_offset_y: i32,  // ring buffer offset in physical texture Y
    near_enabled:  u32,  // 0: disable near overlay, 1: enable
    far_downsample: u32, // world-cell pitch of far texel
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
    _pad3:         u32,
}

@group(0) @binding(0) var<uniform> view:    View;
@group(0) @binding(1) var<uniform> params:  ComposeParams;
@group(0) @binding(2) var          near_tex: texture_2d<u32>;
@group(0) @binding(3) var          far_tex:  texture_2d<u32>;

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

fn floor_div_positive(x: i32, d: i32) -> i32 {
    let q = x / d;
    let r = x % d;
    return select(q - 1, q, r >= 0);
}

fn far_sample_weighted_color(
    texel: vec2<i32>,
    w: f32,
    dot: vec2<i32>,
    seed: u32,
    inout_color: ptr<function, vec3<f32>>,
    inout_alpha: ptr<function, f32>,
) {
    let sample = textureLoad(far_tex, texel, 0);
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

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // World position → global cell coordinate (world_xy / cell_size_m, floored).
    let cell_f = in.world_xy / params.cell_size_m;
    let world_cell = vec2<i32>(i32(floor(cell_f.x)), i32(floor(cell_f.y)));

    // Dot coordinates for palette variation (8×8 sub-cell grid, global indexing).
    let dot_f = in.world_xy / params.dot_size_m;
    let dot = vec2<i32>(i32(floor(dot_f.x)), i32(floor(dot_f.y)));

    // Far is always rendered as background (manual bilinear-like blend across 4 texels).
    let far_size = vec2<i32>(textureDimensions(far_tex));
    let downsample_f = f32(max(params.far_downsample, 1u));
    // Sample Far using world-cell centers. This keeps Far aligned with cell boundaries
    // when downsample == 1 (no half-cell drift from continuous world sampling).
    let world_cell_center = vec2<f32>(f32(world_cell.x) + 0.5, f32(world_cell.y) + 0.5);
    let far_coord =
        (world_cell_center - vec2<f32>(f32(params.far_origin_x), f32(params.far_origin_y)))
            / downsample_f
        - vec2<f32>(0.5, 0.5);
    let far_base = vec2<i32>(i32(floor(far_coord.x)), i32(floor(far_coord.y)));
    let frac = far_coord - vec2<f32>(f32(far_base.x), f32(far_base.y));
    let wx0 = 1.0 - clamp(frac.x, 0.0, 1.0);
    let wx1 = clamp(frac.x, 0.0, 1.0);
    let wy0 = 1.0 - clamp(frac.y, 0.0, 1.0);
    let wy1 = clamp(frac.y, 0.0, 1.0);

    var color_acc = vec3<f32>(0.0, 0.0, 0.0);
    var alpha_acc = 0.0;
    let tx00 = far_base;
    let tx10 = far_base + vec2<i32>(1, 0);
    let tx01 = far_base + vec2<i32>(0, 1);
    let tx11 = far_base + vec2<i32>(1, 1);

    if tx00.x >= 0 && tx00.y >= 0 && tx00.x < far_size.x && tx00.y < far_size.y {
        far_sample_weighted_color(tx00, wx0 * wy0, dot, params.palette_seed, &color_acc, &alpha_acc);
    }
    if tx10.x >= 0 && tx10.y >= 0 && tx10.x < far_size.x && tx10.y < far_size.y {
        far_sample_weighted_color(tx10, wx1 * wy0, dot, params.palette_seed, &color_acc, &alpha_acc);
    }
    if tx01.x >= 0 && tx01.y >= 0 && tx01.x < far_size.x && tx01.y < far_size.y {
        far_sample_weighted_color(tx01, wx0 * wy1, dot, params.palette_seed, &color_acc, &alpha_acc);
    }
    if tx11.x >= 0 && tx11.y >= 0 && tx11.x < far_size.x && tx11.y < far_size.y {
        far_sample_weighted_color(tx11, wx1 * wy1, dot, params.palette_seed, &color_acc, &alpha_acc);
    }

    var color = vec3<f32>(0.0, 0.0, 0.0);
    var alpha = 0.0;
    if alpha_acc > 1.0e-6 {
        color = color_acc / alpha_acc;
        alpha = alpha_acc;
    }

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
            alpha = 1.0;
        }
    }

    if alpha <= 1.0e-6 {
        discard;
    }
    return vec4<f32>(color, alpha);
}
