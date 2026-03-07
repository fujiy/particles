// GPU-side chunk occupancy/resident flag maintenance for overlay and diagnostics.

#import particles::mpm_types::{GpuParticle, MpmParams}

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
    let slot_id = p.home_chunk_slot_id;
    if slot_id >= params.resident_chunk_count || slot_id == INVALID_SLOT {
        return;
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
