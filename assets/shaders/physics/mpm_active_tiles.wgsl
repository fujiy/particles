// Active-tile reconstruction for sparse clear/grid-update dispatch.

#import particles::mpm_types::{GpuParticle, MpmParams, INVALID_PARTICLE_SLOT, particle_slot_id}

struct GpuChunkMetaAtomic {
    chunk_coord_x: i32,
    chunk_coord_y: i32,
    neighbor_slot_id: array<u32, 8>,
    active_tile_mask: atomic<u32>,
    particle_count_curr: u32,
    particle_count_next: u32,
    occupied_bit_curr: u32,
    occupied_bit_next: u32,
}

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

struct GpuActiveTileRecord {
    slot_id: u32,
    tile_id: u32,
}

struct DispatchArgs {
    x: u32,
    y: u32,
    z: u32,
}

const INVALID_SLOT: u32 = 0xffffffffu;
const ACTIVE_TILE_NODE_DIM: u32 = 8u;
const MAX_ACTIVE_TILE_RECORDS: u32 = 4096u;

fn active_tile_count_per_axis(params: MpmParams) -> u32 {
    return max((params.chunk_node_dim + ACTIVE_TILE_NODE_DIM - 1u) / ACTIVE_TILE_NODE_DIM, 1u);
}

fn neighbor_index_from_delta(dx: i32, dy: i32) -> u32 {
    if dx == -1 && dy == -1 { return 0u; }
    if dx ==  0 && dy == -1 { return 1u; }
    if dx ==  1 && dy == -1 { return 2u; }
    if dx == -1 && dy ==  0 { return 3u; }
    if dx ==  1 && dy ==  0 { return 4u; }
    if dx == -1 && dy ==  1 { return 5u; }
    if dx ==  0 && dy ==  1 { return 6u; }
    if dx ==  1 && dy ==  1 { return 7u; }
    return INVALID_SLOT;
}

@group(0) @binding(0) var<uniform> params_clear: MpmParams;
@group(0) @binding(1) var<storage, read_write> chunk_meta_clear: array<GpuChunkMetaAtomic>;

@compute @workgroup_size(64)
fn clear_active_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_id = gid.x;
    if slot_id >= params_clear.resident_chunk_count {
        return;
    }
    atomicStore(&chunk_meta_clear[slot_id].active_tile_mask, 0u);
}

@group(0) @binding(0) var<uniform> params_mark: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> chunk_meta_mark: array<GpuChunkMetaAtomic>;

@compute @workgroup_size(64)
fn mark_active_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params_mark.particle_count {
        return;
    }

    let p = particles[pid];
    let home_slot = particle_slot_id(p);
    if home_slot >= params_mark.resident_chunk_count || home_slot == INVALID_PARTICLE_SLOT {
        return;
    }
    let cdim_i = i32(params_mark.chunk_node_dim);
    if cdim_i <= 0 {
        return;
    }

    let home_node_origin = vec2<i32>(
        chunk_meta_mark[home_slot].chunk_coord_x * cdim_i,
        chunk_meta_mark[home_slot].chunk_coord_y * cdim_i,
    );
    let inv_h = 1.0 / max(params_mark.h, 1.0e-6);
    let grid_pos = p.x * inv_h;
    let base = vec2<i32>(
        i32(floor(grid_pos.x - 0.5)),
        i32(floor(grid_pos.y - 0.5)),
    );
    let tiles_per_axis = active_tile_count_per_axis(params_mark);

    for (var oy: i32 = 0; oy < 3; oy++) {
        for (var ox: i32 = 0; ox < 3; ox++) {
            let node = base + vec2<i32>(ox, oy);
            let local = node - home_node_origin;
            var delta_chunk_x = 0;
            var delta_chunk_y = 0;
            if local.x < 0 {
                delta_chunk_x = -1;
            } else if local.x >= cdim_i {
                delta_chunk_x = 1;
            }
            if local.y < 0 {
                delta_chunk_y = -1;
            } else if local.y >= cdim_i {
                delta_chunk_y = 1;
            }
            if abs(delta_chunk_x) > 1 || abs(delta_chunk_y) > 1 {
                continue;
            }

            var slot_id = home_slot;
            if delta_chunk_x != 0 || delta_chunk_y != 0 {
                let neighbor_idx = neighbor_index_from_delta(delta_chunk_x, delta_chunk_y);
                if neighbor_idx == INVALID_SLOT {
                    continue;
                }
                let neighbor_slot = chunk_meta_mark[home_slot].neighbor_slot_id[neighbor_idx];
                if neighbor_slot == INVALID_SLOT {
                    continue;
                }
                slot_id = neighbor_slot;
            }

            let local_x = local.x - delta_chunk_x * cdim_i;
            let local_y = local.y - delta_chunk_y * cdim_i;
            if local_x < 0 || local_y < 0 || local_x >= cdim_i || local_y >= cdim_i {
                continue;
            }

            let tile_x = u32(local_x) / ACTIVE_TILE_NODE_DIM;
            let tile_y = u32(local_y) / ACTIVE_TILE_NODE_DIM;
            let tile_id = tile_y * tiles_per_axis + tile_x;
            if tile_id >= 32u {
                continue;
            }
            atomicOr(&chunk_meta_mark[slot_id].active_tile_mask, 1u << tile_id);
        }
    }
}

@group(0) @binding(0) var<uniform> params_compact: MpmParams;
@group(0) @binding(1) var<storage, read> chunk_meta_compact: array<GpuChunkMeta>;
@group(0) @binding(2) var<storage, read_write> active_tile_count_rw: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> active_tile_list: array<GpuActiveTileRecord>;

@compute @workgroup_size(64)
fn compact_active_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_id = gid.x;
    if slot_id >= params_compact.resident_chunk_count {
        return;
    }

    let mask = chunk_meta_compact[slot_id].active_tile_mask;
    if mask == 0u {
        return;
    }

    let tiles_per_axis = active_tile_count_per_axis(params_compact);
    let tile_count = min(tiles_per_axis * tiles_per_axis, 32u);
    for (var tile_id = 0u; tile_id < tile_count; tile_id++) {
        if (mask & (1u << tile_id)) == 0u {
            continue;
        }
        let out_idx = atomicAdd(&active_tile_count_rw, 1u);
        if out_idx >= MAX_ACTIVE_TILE_RECORDS {
            continue;
        }
        active_tile_list[out_idx] = GpuActiveTileRecord(slot_id, tile_id);
    }
}

@group(0) @binding(0) var<storage, read> active_tile_count_ro: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch_args: DispatchArgs;

@compute @workgroup_size(1)
fn finalize_active_tile_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }
    dispatch_args.x = atomicLoad(&active_tile_count_ro);
    dispatch_args.y = 1u;
    dispatch_args.z = 1u;
}
