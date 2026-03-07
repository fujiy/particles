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
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: MpmParams;
@group(0) @binding(2) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(3) var<uniform> colors: ChunkOverlayColors;

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    local_uv: vec2<f32>,
    @location(1)
    edge_color: vec4<f32>,
    @location(2)
    grid_color: vec4<f32>,
};

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
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
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
    let grid_alpha = 1.0 - smoothstep(0.5, 1.5, grid_px);

    let edge_term = in.edge_color.a * edge_alpha;
    let grid_term = in.grid_color.a * grid_alpha;
    if edge_term <= 1.0e-4 && grid_term <= 1.0e-4 {
        discard;
    }

    if edge_term >= grid_term {
        return vec4<f32>(in.edge_color.rgb, edge_term);
    }
    return vec4<f32>(in.grid_color.rgb, grid_term);
}
