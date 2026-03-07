#import bevy_render::view::View
#import particles::mpm_types::MpmParams

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: MpmParams;
@group(0) @binding(2) var<storage, read> terrain_sdf: array<f32>;
@group(0) @binding(3) var<storage, read> chunk_meta: array<GpuChunkMeta>;

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

const INVALID_SLOT: u32 = 0xffffffffu;

fn floor_div_i32(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        return q - 1;
    }
    return q;
}

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    local_uv: vec2<f32>,
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
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let uv = quad_corner(vertex_index);
    let grid_w = max(f32(params.grid_width), 1.0);
    let grid_h = max(f32(params.grid_height), 1.0);
    let span = vec2<f32>(max(grid_w - 1.0, 1.0), max(grid_h - 1.0, 1.0));
    let grid_min = vec2<f32>(f32(params.grid_origin_x), f32(params.grid_origin_y)) * params.h;
    let grid_max = grid_min + span * params.h;
    let world_xy = mix(grid_min, grid_max, uv);

    var out: VertexOut;
    out.clip_position = view.clip_from_world * vec4<f32>(world_xy, 0.0, 1.0);
    out.local_uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if params.grid_width == 0u
        || params.grid_height == 0u
        || params.chunk_node_dim == 0u
        || params.chunk_dims_x == 0u
        || params.chunk_dims_y == 0u
    {
        discard;
    }

    let grid_w_f = f32(params.grid_width);
    let grid_h_f = f32(params.grid_height);
    let gx_local =
        u32(clamp(round(in.local_uv.x * max(grid_w_f - 1.0, 0.0)), 0.0, grid_w_f - 1.0));
    let gy_local =
        u32(clamp(round(in.local_uv.y * max(grid_h_f - 1.0, 0.0)), 0.0, grid_h_f - 1.0));
    let cdim = params.chunk_node_dim;
    let cdim_i = i32(cdim);
    let node_x = i32(gx_local) + params.grid_origin_x;
    let node_y = i32(gy_local) + params.grid_origin_y;
    let chunk_x = floor_div_i32(node_x, cdim_i);
    let chunk_y = floor_div_i32(node_y, cdim_i);
    let local_x = u32(node_x - chunk_x * cdim_i);
    let local_y = u32(node_y - chunk_y * cdim_i);

    var slot_id = INVALID_SLOT;
    for (var s = 0u; s < params.resident_chunk_count; s++) {
        let ch = chunk_meta[s];
        if ch.chunk_coord_x == chunk_x
            && ch.chunk_coord_y == chunk_y
            && ch.occupied_bit_next != 0u
        {
            slot_id = s;
            break;
        }
    }
    if slot_id == INVALID_SLOT {
        discard;
    }

    let nodes_per_chunk = cdim * cdim;
    let idx = slot_id * nodes_per_chunk + local_y * cdim + local_x;

    let sdf = terrain_sdf[idx];
    let range_m = max(params.h * 6.0, 1.0e-4);
    if sdf <= 0.0 {
        let depth = clamp((-sdf) / range_m, 0.0, 1.0);
        return vec4<f32>(
            0.92,
            0.22 + 0.30 * depth,
            0.10 + 0.12 * depth,
            0.24 + 0.54 * depth
        );
    }

    let sdf_norm = clamp(sdf / range_m, -1.0, 1.0);
    if sdf_norm >= 1.0 {
        discard;
    }
    let near = 1.0 - sdf_norm;
    return vec4<f32>(0.10, 0.72, 0.95, 0.06 + 0.24 * near);
}
