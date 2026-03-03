/// TerrainCompose fragment pass.
///
/// Reads material IDs from `near_tex` (R16Uint, one texel == one world cell) and
/// renders terrain with deterministic dot-art palette variation (8×8 dots/cell).
#import bevy_render::view::View

struct ComposeParams {
    origin_x:      f32,  // world-space X of cache origin (metres)
    origin_y:      f32,  // world-space Y of cache origin (metres)
    cell_size_m:   f32,  // metres per cell (CELL_SIZE_M = 0.25)
    dot_size_m:    f32,  // metres per dot (cell_size_m / dots_per_cell)
    cache_origin_x: i32, // world cell X of texture texel (0,0)
    cache_origin_y: i32, // world cell Y of texture texel (0,0)
    palette_seed:  u32,
    dots_per_cell: u32,
}

@group(0) @binding(0) var<uniform> view:    View;
@group(0) @binding(1) var<uniform> params:  ComposeParams;
@group(0) @binding(2) var          near_tex: texture_2d<u32>;

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

// ── Fragment ──────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // World position → global cell coordinate (world_xy / cell_size_m, floored).
    let cell_f = in.world_xy / params.cell_size_m;
    let world_cell = vec2<i32>(i32(floor(cell_f.x)), i32(floor(cell_f.y)));

    // Global cell → texture coordinate (subtract cache origin; no ring offset in this initial version).
    let tc = world_cell - vec2<i32>(params.cache_origin_x, params.cache_origin_y);
    let tex_size = vec2<i32>(textureDimensions(near_tex));
    if tc.x < 0 || tc.y < 0 || tc.x >= tex_size.x || tc.y >= tex_size.y {
        discard;
    }

    let material = textureLoad(near_tex, tc, 0).r;
    if material == 0u {
        discard;
    }

    // Dot coordinates for palette variation (8×8 sub-cell grid, global indexing).
    let dot_f = in.world_xy / params.dot_size_m;
    let dot = vec2<i32>(i32(floor(dot_f.x)), i32(floor(dot_f.y)));

    let color = terrain_color(material, dot.x, dot.y, params.palette_seed);
    return vec4<f32>(color, 1.0);
}
