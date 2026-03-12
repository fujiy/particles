#import bevy_render::view::View
#import particles::mpm_types::{GpuGridNode, MpmParams}

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

struct MassOverlayStyle {
    low: vec4<f32>,
    mid: vec4<f32>,
    high: vec4<f32>,
    opacity: f32,
    max_ref_cell_mass_scale: f32,
    pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: MpmParams;
@group(0) @binding(2) var<storage, read> grid: array<GpuGridNode>;
@group(0) @binding(3) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(4) var<uniform> style: MassOverlayStyle;

const INVALID_SLOT: u32 = 0xffffffffu;
const NODES_PER_CELL: u32 = 2u;
const MASS_EPSILON: f32 = 1.0e-5;

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    local_uv: vec2<f32>,
};

fn floor_div_i32(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        return q - 1;
    }
    return q;
}

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    if vertex_index == 0u { return vec2<f32>(0.0, 0.0); }
    if vertex_index == 1u { return vec2<f32>(1.0, 0.0); }
    if vertex_index == 2u { return vec2<f32>(1.0, 1.0); }
    if vertex_index == 3u { return vec2<f32>(0.0, 0.0); }
    if vertex_index == 4u { return vec2<f32>(1.0, 1.0); }
    return vec2<f32>(0.0, 1.0);
}

fn node_index_from_slot_local(slot_id: u32, local_x: u32, local_y: u32) -> u32 {
    let nodes_per_chunk = params.chunk_node_dim * params.chunk_node_dim;
    return slot_id * nodes_per_chunk + local_y * params.chunk_node_dim + local_x;
}

fn node_total_mass(node_index: u32) -> f32 {
    return max(grid[node_index].water_mass, 0.0) + max(grid[node_index].granular_mass, 0.0);
}

fn cell_mass_from_slot_local(slot_id: u32, local_cell_x: u32, local_cell_y: u32) -> f32 {
    let base_x = local_cell_x * NODES_PER_CELL;
    let base_y = local_cell_y * NODES_PER_CELL;
    let idx00 = node_index_from_slot_local(slot_id, base_x, base_y);
    let idx10 = node_index_from_slot_local(slot_id, base_x + 1u, base_y);
    let idx01 = node_index_from_slot_local(slot_id, base_x, base_y + 1u);
    let idx11 = node_index_from_slot_local(slot_id, base_x + 1u, base_y + 1u);
    return 0.25 * (
        node_total_mass(idx00)
        + node_total_mass(idx10)
        + node_total_mass(idx01)
        + node_total_mass(idx11)
    );
}

fn mass_gradient(t: f32) -> vec4<f32> {
    let clamped_t = clamp(t, 0.0, 1.0);
    if clamped_t <= 0.5 {
        return mix(style.low, style.mid, clamped_t * 2.0);
    }
    return mix(style.mid, style.high, (clamped_t - 0.5) * 2.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let uv = quad_corner(vertex_index);
    let cell_size = params.h * f32(NODES_PER_CELL);
    let cell_count_x = max((params.grid_width + NODES_PER_CELL - 1u) / NODES_PER_CELL, 1u);
    let cell_count_y = max((params.grid_height + NODES_PER_CELL - 1u) / NODES_PER_CELL, 1u);
    let cell_origin = vec2<f32>(
        f32(floor_div_i32(params.grid_origin_x, i32(NODES_PER_CELL))),
        f32(floor_div_i32(params.grid_origin_y, i32(NODES_PER_CELL))),
    ) * cell_size;
    let world_xy = cell_origin + uv * vec2<f32>(f32(cell_count_x), f32(cell_count_y)) * cell_size;

    var out: VertexOut;
    out.clip_position = view.clip_from_world * vec4<f32>(world_xy, 0.0, 1.0);
    out.local_uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if params.grid_width == 0u
        || params.grid_height == 0u
        || params.chunk_node_dim < NODES_PER_CELL
        || params.resident_chunk_count == 0u
    {
        discard;
    }

    let cell_count_x = max((params.grid_width + NODES_PER_CELL - 1u) / NODES_PER_CELL, 1u);
    let cell_count_y = max((params.grid_height + NODES_PER_CELL - 1u) / NODES_PER_CELL, 1u);
    let cell_x_local = min(u32(in.local_uv.x * f32(cell_count_x)), cell_count_x - 1u);
    let cell_y_local = min(u32(in.local_uv.y * f32(cell_count_y)), cell_count_y - 1u);
    let cell_origin_x = floor_div_i32(params.grid_origin_x, i32(NODES_PER_CELL));
    let cell_origin_y = floor_div_i32(params.grid_origin_y, i32(NODES_PER_CELL));
    let global_cell_x = i32(cell_x_local) + cell_origin_x;
    let global_cell_y = i32(cell_y_local) + cell_origin_y;

    let chunk_cell_dim = max(params.chunk_node_dim / NODES_PER_CELL, 1u);
    let chunk_cell_dim_i = i32(chunk_cell_dim);
    let chunk_x = floor_div_i32(global_cell_x, chunk_cell_dim_i);
    let chunk_y = floor_div_i32(global_cell_y, chunk_cell_dim_i);

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

    let local_cell_x = u32(global_cell_x - chunk_x * chunk_cell_dim_i);
    let local_cell_y = u32(global_cell_y - chunk_y * chunk_cell_dim_i);
    let cell_mass = cell_mass_from_slot_local(slot_id, local_cell_x, local_cell_y);
    if cell_mass <= MASS_EPSILON {
        discard;
    }

    let cell_world_size = params.h * f32(NODES_PER_CELL);
    let ref_cell_mass =
        max(params.rho_ref * cell_world_size * cell_world_size * style.max_ref_cell_mass_scale, 1.0e-6);
    let t = clamp(cell_mass / ref_cell_mass, 0.0, 1.0);
    let color = mass_gradient(t);
    let alpha = clamp(style.opacity, 0.0, 1.0) * color.a * smoothstep(0.0, 0.06, t);
    if alpha <= 1.0e-4 {
        discard;
    }
    return vec4<f32>(color.rgb, alpha);
}
