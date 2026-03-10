#import bevy_render::view::View
#import particles::mpm_types::MpmParams

struct GpuChunkMeta {
    chunk_coord_x: i32,
    chunk_coord_y: i32,
    neighbor_slot_id: array<u32, 8>,
    active_tile_mask: u32,
    particle_count_curr: u32,
    particle_count_next: u32,
    occupied_bit_curr: u32,
    occupied_bit_next: u32,
}

struct ChunkOverlayColors {
    occupied_edge: vec4<f32>,
    occupied_grid: vec4<f32>,
    halo_edge: vec4<f32>,
    halo_grid: vec4<f32>,
    free_edge: vec4<f32>,
    free_grid: vec4<f32>,
    active_tile_fill: vec4<f32>,
    active_tile_edge: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: MpmParams;
@group(0) @binding(2) var source_texture: texture_2d<f32>;
@group(0) @binding(3) var source_sampler: sampler;
@group(0) @binding(4) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(5) var<storage, read> terrain_node_solid: array<u32>;
@group(0) @binding(6) var<uniform> colors: ChunkOverlayColors;

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    local_uv: vec2<f32>,
    @location(1)
    edge_color: vec4<f32>,
    @location(2)
    grid_color: vec4<f32>,
    @location(3)
    active_tile_mask: u32,
    @location(4)
    screen_uv: vec2<f32>,
    @location(5)
    slot_id: u32,
};

const ACTIVE_TILE_NODE_DIM: u32 = 8u;

fn node_index_from_slot_local(slot_id: u32, local_x: u32, local_y: u32) -> u32 {
    let nodes_per_chunk = params.chunk_node_dim * params.chunk_node_dim;
    return slot_id * nodes_per_chunk + local_y * params.chunk_node_dim + local_x;
}

fn slot_local_node_is_fluid(slot_id: u32, local_x: u32, local_y: u32) -> bool {
    if params.chunk_node_dim == 0u {
        return true;
    }
    let lx = min(local_x, params.chunk_node_dim - 1u);
    let ly = min(local_y, params.chunk_node_dim - 1u);
    let idx = node_index_from_slot_local(slot_id, lx, ly);
    return terrain_node_solid[idx] == 0u;
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    if vertex_index == 0u { return vec2<f32>(0.0, 0.0); }
    if vertex_index == 1u { return vec2<f32>(1.0, 0.0); }
    if vertex_index == 2u { return vec2<f32>(1.0, 1.0); }
    if vertex_index == 3u { return vec2<f32>(0.0, 0.0); }
    if vertex_index == 4u { return vec2<f32>(1.0, 1.0); }
    return vec2<f32>(0.0, 1.0);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    let uv = quad_corner(vertex_index);
    let chunk_entry = chunk_meta[instance_index];

    let chunk_world_size = max(f32(params.chunk_node_dim) * params.h, params.h);
    let chunk_min = vec2<f32>(
        f32(chunk_entry.chunk_coord_x) * chunk_world_size,
        f32(chunk_entry.chunk_coord_y) * chunk_world_size,
    );
    let world_xy = chunk_min + uv * chunk_world_size;

    let is_occupied = chunk_entry.occupied_bit_curr != 0u;
    let is_halo_only = !is_occupied && (chunk_entry.occupied_bit_next != 0u);
    var edge_color = colors.free_edge;
    var grid_color = colors.free_grid;
    if is_halo_only {
        edge_color = colors.halo_edge;
        grid_color = colors.halo_grid;
    }
    if is_occupied {
        edge_color = colors.occupied_edge;
        grid_color = colors.occupied_grid;
    }

    var out: VertexOut;
    out.clip_position = view.clip_from_world * vec4<f32>(world_xy.x, world_xy.y, 0.0, 1.0);
    out.local_uv = uv;
    out.edge_color = edge_color;
    out.grid_color = grid_color;
    out.active_tile_mask = chunk_entry.active_tile_mask;
    out.screen_uv = out.clip_position.xy * vec2<f32>(0.5, -0.5) / out.clip_position.w + vec2<f32>(0.5, 0.5);
    out.slot_id = instance_index;
    return out;
}

fn luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn invert_luminance(rgb: vec3<f32>) -> vec3<f32> {
    let lum = luminance(rgb);
    let inverted_luma = 1.0 - lum;
    if lum <= 1.0e-5 {
        return vec3<f32>(inverted_luma);
    }
    return clamp(rgb * (inverted_luma / lum), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn adapt_overlay_color(color: vec4<f32>, background_luma: f32) -> vec4<f32> {
    let bright_mix = smoothstep(0.35, 0.65, background_luma);
    let dark_rgb = invert_luminance(color.rgb);
    return vec4<f32>(mix(dark_rgb, color.rgb, bright_mix), color.a);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let background = textureSample(
        source_texture,
        source_sampler,
        clamp(in.screen_uv, vec2<f32>(0.0), vec2<f32>(1.0)),
    );
    let background_luma = luminance(background.rgb);
    let edge_color = adapt_overlay_color(in.edge_color, background_luma);
    let grid_color = adapt_overlay_color(in.grid_color, background_luma);
    let edge_du = min(in.local_uv.x, 1.0 - in.local_uv.x);
    let edge_dv = min(in.local_uv.y, 1.0 - in.local_uv.y);
    let edge_px = min(
        edge_du / max(fwidth(in.local_uv.x), 1.0e-6),
        edge_dv / max(fwidth(in.local_uv.y), 1.0e-6),
    );
    let edge_alpha = 1.0 - smoothstep(0.5, 1.5, edge_px);

    let divisions = max(f32(params.chunk_node_dim), 1.0);
    let grid_uv = in.local_uv * divisions;
    let grid_frac = fract(grid_uv);
    let grid_du = min(grid_frac.x, 1.0 - grid_frac.x);
    let grid_dv = min(grid_frac.y, 1.0 - grid_frac.y);
    let grid_px = min(
        grid_du / max(fwidth(grid_uv.x), 1.0e-6),
        grid_dv / max(fwidth(grid_uv.y), 1.0e-6),
    );
    let grid_alpha_base = 1.0 - smoothstep(0.5, 1.5, grid_px);
    let nearest_node = clamp(
        vec2<u32>(floor(grid_uv + vec2<f32>(0.5, 0.5))),
        vec2<u32>(0u),
        vec2<u32>(max(params.chunk_node_dim, 1u) - 1u),
    );
    let fluid_node = slot_local_node_is_fluid(
        in.slot_id,
        nearest_node.x,
        nearest_node.y,
    );

    let edge_term = edge_color.a * edge_alpha;
    let tiles_per_axis = max((params.chunk_node_dim + ACTIVE_TILE_NODE_DIM - 1u) / ACTIVE_TILE_NODE_DIM, 1u);
    let tile_uv = in.local_uv * f32(tiles_per_axis);
    let tile_clamped = clamp(vec2<u32>(tile_uv), vec2<u32>(0u), vec2<u32>(tiles_per_axis - 1u));
    let tile_id = tile_clamped.y * tiles_per_axis + tile_clamped.x;
    let tile_active = tile_id < 32u && (in.active_tile_mask & (1u << tile_id)) != 0u;

    let grid_term = select(0.0, grid_color.a * grid_alpha_base, tile_active && fluid_node);
    if edge_term <= 1.0e-4 && grid_term <= 1.0e-4 {
        discard;
    }

    var out_rgb = grid_color.rgb;
    var out_alpha = grid_term;
    if edge_term > out_alpha {
        out_rgb = edge_color.rgb;
        out_alpha = edge_term;
    }
    return vec4<f32>(out_rgb, out_alpha);
}
