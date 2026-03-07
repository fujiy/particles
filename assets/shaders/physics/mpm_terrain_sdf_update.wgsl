// Recompute terrain SDF/normal on GPU from slot-local solid-cell occupancy.

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

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(2) var<storage, read> terrain_cell_solid: array<u32>;
@group(0) @binding(3) var<storage, read_write> terrain_sdf: array<f32>;
@group(0) @binding(4) var<storage, read_write> terrain_normal: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> terrain_update_slot_count: array<u32>;
@group(0) @binding(6) var<storage, read> terrain_update_slots: array<u32>;

const INVALID_SLOT: u32 = 0xffffffffu;
const SDF_QUERY_RADIUS_CELLS: i32 = 10;
const SDF_INF: f32 = 1.0e9;

fn floor_div_i32(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        return q - 1;
    }
    return q;
}

fn neighbor_slot(base_slot: u32, dx_chunk: i32, dy_chunk: i32) -> u32 {
    if dx_chunk == 0 && dy_chunk == 0 {
        return base_slot;
    }
    if abs(dx_chunk) > 1 || abs(dy_chunk) > 1 {
        return INVALID_SLOT;
    }
    var idx: u32 = INVALID_SLOT;
    if dx_chunk == -1 && dy_chunk == -1 { idx = 0u; }
    if dx_chunk == 0 && dy_chunk == -1 { idx = 1u; }
    if dx_chunk == 1 && dy_chunk == -1 { idx = 2u; }
    if dx_chunk == -1 && dy_chunk == 0 { idx = 3u; }
    if dx_chunk == 1 && dy_chunk == 0 { idx = 4u; }
    if dx_chunk == -1 && dy_chunk == 1 { idx = 5u; }
    if dx_chunk == 0 && dy_chunk == 1 { idx = 6u; }
    if dx_chunk == 1 && dy_chunk == 1 { idx = 7u; }
    if idx == INVALID_SLOT {
        return INVALID_SLOT;
    }
    return chunk_meta[base_slot].neighbor_slot_id[idx];
}

fn terrain_cell_is_solid(base_slot: u32, cell_x: i32, cell_y: i32) -> bool {
    let cdim_i = i32(params.chunk_node_dim);
    let target_chunk_x = floor_div_i32(cell_x, cdim_i);
    let target_chunk_y = floor_div_i32(cell_y, cdim_i);
    let base_chunk_x = chunk_meta[base_slot].chunk_coord_x;
    let base_chunk_y = chunk_meta[base_slot].chunk_coord_y;
    let dx_chunk = target_chunk_x - base_chunk_x;
    let dy_chunk = target_chunk_y - base_chunk_y;
    let slot_id = neighbor_slot(base_slot, dx_chunk, dy_chunk);
    if slot_id == INVALID_SLOT || slot_id >= params.resident_chunk_count {
        return false;
    }
    let local_x = cell_x - target_chunk_x * cdim_i;
    let local_y = cell_y - target_chunk_y * cdim_i;
    if local_x < 0 || local_y < 0 || local_x >= cdim_i || local_y >= cdim_i {
        return false;
    }
    let cdim = params.chunk_node_dim;
    let nodes_per_chunk = cdim * cdim;
    let idx = slot_id * nodes_per_chunk + u32(local_y) * cdim + u32(local_x);
    return terrain_cell_solid[idx] != 0u;
}

fn signed_distance_at_world(base_slot: u32, world_pos: vec2<f32>) -> f32 {
    let h = max(params.h, 1.0e-6);
    let center_cell = vec2<i32>(
        i32(floor(world_pos.x / h)),
        i32(floor(world_pos.y / h)),
    );
    let inside = terrain_cell_is_solid(base_slot, center_cell.x, center_cell.y);
    var best = SDF_INF;
    for (var oy = -SDF_QUERY_RADIUS_CELLS; oy <= SDF_QUERY_RADIUS_CELLS; oy++) {
        for (var ox = -SDF_QUERY_RADIUS_CELLS; ox <= SDF_QUERY_RADIUS_CELLS; ox++) {
            let cx = center_cell.x + ox;
            let cy = center_cell.y + oy;
            let cell_is_solid = terrain_cell_is_solid(base_slot, cx, cy);
            if cell_is_solid == inside {
                continue;
            }
            let cell_min = vec2<f32>(f32(cx) * h, f32(cy) * h);
            let cell_max = cell_min + vec2<f32>(h, h);
            let closest = clamp(world_pos, cell_min, cell_max);
            let d = distance(world_pos, closest);
            best = min(best, d);
        }
    }
    if best >= SDF_INF * 0.5 {
        best = (f32(SDF_QUERY_RADIUS_CELLS) + 1.0) * h;
    }
    return select(best, -best, inside);
}

fn sdf_normal_at_world(base_slot: u32, world_pos: vec2<f32>) -> vec2<f32> {
    let eps = max(0.5 * params.h, 1.0e-4);
    let dx = signed_distance_at_world(base_slot, world_pos + vec2<f32>(eps, 0.0))
        - signed_distance_at_world(base_slot, world_pos - vec2<f32>(eps, 0.0));
    let dy = signed_distance_at_world(base_slot, world_pos + vec2<f32>(0.0, eps))
        - signed_distance_at_world(base_slot, world_pos - vec2<f32>(0.0, eps));
    let g = vec2<f32>(dx, dy);
    let g_len = length(g);
    if g_len > 1.0e-6 {
        return g / g_len;
    }
    return vec2<f32>(0.0, 1.0);
}

@compute @workgroup_size(64)
fn update_terrain_sdf_slots(@builtin(global_invocation_id) gid: vec3<u32>) {
    let update_count = terrain_update_slot_count[0];
    if update_count == 0u || params.chunk_node_dim == 0u {
        return;
    }
    let nodes_per_chunk = params.chunk_node_dim * params.chunk_node_dim;
    let update_index = gid.x / nodes_per_chunk;
    if update_index >= update_count {
        return;
    }
    let slot_id = terrain_update_slots[update_index];
    if slot_id >= params.resident_chunk_count {
        return;
    }

    let local_index = gid.x - update_index * nodes_per_chunk;
    let local_x = local_index % params.chunk_node_dim;
    let local_y = local_index / params.chunk_node_dim;
    let cdim_i = i32(params.chunk_node_dim);
    let chunk = chunk_meta[slot_id];
    let node_x = chunk.chunk_coord_x * cdim_i + i32(local_x);
    let node_y = chunk.chunk_coord_y * cdim_i + i32(local_y);
    let world_pos = vec2<f32>(f32(node_x) * params.h, f32(node_y) * params.h);

    let sdf = signed_distance_at_world(slot_id, world_pos);
    let normal = sdf_normal_at_world(slot_id, world_pos);
    let idx = slot_id * nodes_per_chunk + local_index;
    terrain_sdf[idx] = sdf;
    terrain_normal[idx] = normal;
}
