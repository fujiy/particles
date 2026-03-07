// Grid clear pass: zero out active-tile grid nodes.
// One workgroup per active tile, one thread per node within the 8x8 tile.

#import particles::mpm_types::{GpuGridNode, MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<GpuGridNode>;
@group(0) @binding(2) var<storage, read> active_tiles: array<vec2<u32>>;

const ACTIVE_TILE_NODE_DIM: u32 = 8u;

fn node_index_from_slot_local(slot_id: u32, local_x: u32, local_y: u32, params: MpmParams) -> u32 {
    let nodes_per_chunk = params.chunk_node_dim * params.chunk_node_dim;
    return slot_id * nodes_per_chunk + local_y * params.chunk_node_dim + local_x;
}

@compute @workgroup_size(64)
fn clear_grid(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let record = active_tiles[workgroup_id.x];
    let slot_id = record.x;
    let tile_id = record.y;
    let tiles_per_axis = max((params.chunk_node_dim + ACTIVE_TILE_NODE_DIM - 1u) / ACTIVE_TILE_NODE_DIM, 1u);
    let tile_x = tile_id % tiles_per_axis;
    let tile_y = tile_id / tiles_per_axis;
    let local_x = tile_x * ACTIVE_TILE_NODE_DIM + (local_idx % ACTIVE_TILE_NODE_DIM);
    let local_y = tile_y * ACTIVE_TILE_NODE_DIM + (local_idx / ACTIVE_TILE_NODE_DIM);
    if local_x >= params.chunk_node_dim || local_y >= params.chunk_node_dim {
        return;
    }
    let idx = node_index_from_slot_local(slot_id, local_x, local_y, params);
    grid[idx].water_px = 0.0;
    grid[idx].water_py = 0.0;
    grid[idx].water_mass = 0.0;
    grid[idx].water_pad = 0.0;
    grid[idx].granular_px = 0.0;
    grid[idx].granular_py = 0.0;
    grid[idx].granular_mass = 0.0;
    grid[idx].granular_pad = 0.0;
}
