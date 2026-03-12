// GPU-side chunk occupancy/resident flag maintenance for overlay and diagnostics.

#import particles::mpm_types::{GpuParticle, MpmParams, INVALID_PARTICLE_SLOT, particle_slot_id}

struct GpuChunkMetaAtomic {
    chunk_coord_x: i32,
    chunk_coord_y: i32,
    neighbor_slot_id: array<u32, 8>,
    active_tile_mask: u32,
    particle_count_curr: atomic<u32>,
    particle_count_next: atomic<u32>,
    occupied_bit_curr: atomic<u32>,
    occupied_bit_next: atomic<u32>,
}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> chunk_meta: array<GpuChunkMetaAtomic>;
const INVALID_SLOT: u32 = 0xffffffffu;

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

fn floor_div_i32(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        return q - 1;
    }
    return q;
}

@compute @workgroup_size(64)
fn clear_chunk_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_id = gid.x;
    if slot_id >= params.resident_chunk_count {
        return;
    }
    // Preserve previous occupancy count in `curr`, then clear `next` for this frame accumulation.
    let prev_count = atomicLoad(&chunk_meta[slot_id].particle_count_next);
    atomicStore(&chunk_meta[slot_id].particle_count_curr, prev_count);
    atomicStore(&chunk_meta[slot_id].particle_count_next, 0u);
}

@compute @workgroup_size(64)
fn accumulate_chunk_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let p = particles[pid];
    let home_slot = particle_slot_id(p);
    if home_slot >= params.resident_chunk_count || home_slot == INVALID_PARTICLE_SLOT {
        return;
    }

    let inv_h = 1.0 / max(params.h, 1.0e-6);
    let node_x = i32(floor(p.x.x * inv_h));
    let node_y = i32(floor(p.x.y * inv_h));
    let cdim = i32(params.chunk_node_dim);
    let new_chunk_x = floor_div_i32(node_x, cdim);
    let new_chunk_y = floor_div_i32(node_y, cdim);

    var slot_id = home_slot;
    let dx = new_chunk_x - chunk_meta[home_slot].chunk_coord_x;
    let dy = new_chunk_y - chunk_meta[home_slot].chunk_coord_y;
    if dx != 0 || dy != 0 {
        if abs(dx) <= 1 && abs(dy) <= 1 {
            let neighbor_index = neighbor_index_from_delta(dx, dy);
            if neighbor_index != INVALID_SLOT {
                let neighbor_slot = chunk_meta[home_slot].neighbor_slot_id[neighbor_index];
                if neighbor_slot != INVALID_SLOT {
                    slot_id = neighbor_slot;
                }
            }
        }
    }
    atomicAdd(&chunk_meta[slot_id].particle_count_next, 1u);
}

@compute @workgroup_size(64)
fn finalize_chunk_flags(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_id = gid.x;
    if slot_id >= params.resident_chunk_count {
        return;
    }
    let self_count = atomicLoad(&chunk_meta[slot_id].particle_count_next);
    let occupied = self_count > 0u;

    var resident = occupied;
    if !resident {
        for (var i = 0u; i < 8u; i++) {
            let nslot = chunk_meta[slot_id].neighbor_slot_id[i];
            if nslot == INVALID_SLOT {
                continue;
            }
            if atomicLoad(&chunk_meta[nslot].particle_count_next) > 0u {
                resident = true;
                break;
            }
        }
    }

    // Keep particle_count_curr as "previous frame", particle_count_next as "current frame".
    atomicStore(&chunk_meta[slot_id].occupied_bit_curr, select(0u, 1u, occupied));
    atomicStore(&chunk_meta[slot_id].occupied_bit_next, select(0u, 1u, resident));
}
