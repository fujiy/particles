#import bevy_render::view::View

struct TerrainDotParams {
    origin_x: f32,
    origin_y: f32,
    cell_size_m: f32,
    dot_size_m: f32,
    width: u32,
    height: u32,
    palette_seed: u32,
    dots_per_cell: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: TerrainDotParams;
@group(0) @binding(2) var<storage, read> terrain_cells: array<u32>;

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

fn cell_index(x: i32, y: i32) -> u32 {
    return u32(y) * params.width + u32(x);
}

fn in_bounds(x: i32, y: i32) -> bool {
    return x >= 0 && y >= 0 && u32(x) < params.width && u32(y) < params.height;
}

fn in_dot_bounds(x: i32, y: i32) -> bool {
    let dot_w = params.width * params.dots_per_cell;
    let dot_h = params.height * params.dots_per_cell;
    return x >= 0 && y >= 0 && u32(x) < dot_w && u32(y) < dot_h;
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

fn stone_palette(idx: u32) -> vec3<f32> {
    if idx == 0u {
        return srgb8_to_linear(vec3<f32>(70.0 / 255.0, 67.0 / 255.0, 63.0 / 255.0));
    }
    if idx == 1u {
        return srgb8_to_linear(vec3<f32>(83.0 / 255.0, 79.0 / 255.0, 74.0 / 255.0));
    }
    if idx == 2u {
        return srgb8_to_linear(vec3<f32>(95.0 / 255.0, 90.0 / 255.0, 84.0 / 255.0));
    }
    return srgb8_to_linear(vec3<f32>(108.0 / 255.0, 103.0 / 255.0, 96.0 / 255.0));
}

fn soil_palette(idx: u32) -> vec3<f32> {
    if idx == 0u {
        return srgb8_to_linear(vec3<f32>(105.0 / 255.0, 79.0 / 255.0, 56.0 / 255.0));
    }
    if idx == 1u {
        return srgb8_to_linear(vec3<f32>(119.0 / 255.0, 91.0 / 255.0, 67.0 / 255.0));
    }
    if idx == 2u {
        return srgb8_to_linear(vec3<f32>(133.0 / 255.0, 103.0 / 255.0, 78.0 / 255.0));
    }
    return srgb8_to_linear(vec3<f32>(147.0 / 255.0, 115.0 / 255.0, 88.0 / 255.0));
}

fn sand_palette(idx: u32) -> vec3<f32> {
    if idx == 0u {
        return srgb8_to_linear(vec3<f32>(170.0 / 255.0, 150.0 / 255.0, 110.0 / 255.0));
    }
    if idx == 1u {
        return srgb8_to_linear(vec3<f32>(186.0 / 255.0, 166.0 / 255.0, 124.0 / 255.0));
    }
    if idx == 2u {
        return srgb8_to_linear(vec3<f32>(201.0 / 255.0, 181.0 / 255.0, 137.0 / 255.0));
    }
    return srgb8_to_linear(vec3<f32>(216.0 / 255.0, 196.0 / 255.0, 150.0 / 255.0));
}

fn terrain_color(material: u32, cell_x: i32, cell_y: i32) -> vec3<f32> {
    let shade = deterministic_palette_index(cell_x, cell_y, params.palette_seed);
    if material == 1u {
        return stone_palette(shade);
    }
    if material == 2u {
        return soil_palette(shade);
    }
    return sand_palette(shade);
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
    let dot = vec2<i32>(i32(floor(dot_pos.x)), i32(floor(dot_pos.y)));
    if !in_dot_bounds(dot.x, dot.y) {
        discard;
    }

    let dots_per_cell = i32(params.dots_per_cell);
    let cell = vec2<i32>(dot.x / dots_per_cell, dot.y / dots_per_cell);
    if !in_bounds(cell.x, cell.y) {
        discard;
    }

    let material = terrain_cells[cell_index(cell.x, cell.y)];
    if material == 0u {
        discard;
    }

    let color = terrain_color(material, dot.x, dot.y);
    return vec4<f32>(color, 1.0);
}
