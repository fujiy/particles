// Extract moved particles by comparing previous home slot's chunk coord and current position.
// One thread per particle.

#import particles::mpm_types::{GpuParticle, MpmParams}

struct GpuChunkMeta {
    chunk_coord_x: i32,
    chunk_coord_y: i32,
    neighbor_slot_id: array<u32, 8>,
    active_tile_mask: u32,
    _pad: u32,
}

struct GpuMoverRecord {
    particle_id: u32,
    old_home_slot_id: u32,
    new_chunk_coord_x: i32,
    new_chunk_coord_y: i32,
}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(3) var<storage, read_write> mover_count: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> movers: array<GpuMoverRecord>;

fn floor_div_i32(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        return q - 1;
    }
    return q;
}

@compute @workgroup_size(64)
fn extract_movers(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    if params.chunk_node_dim == 0u || params.resident_chunk_count == 0u {
        return;
    }

    let p = particles[pid];
    let old_slot = p.home_chunk_slot_id;
    if old_slot >= params.resident_chunk_count {
        return;
    }
    let old_chunk = chunk_meta[old_slot];

    let inv_h = 1.0 / max(params.h, 1.0e-6);
    let node_x = i32(floor(p.x.x * inv_h));
    let node_y = i32(floor(p.x.y * inv_h));
    let cdim = i32(params.chunk_node_dim);
    let new_chunk_x = floor_div_i32(node_x, cdim);
    let new_chunk_y = floor_div_i32(node_y, cdim);

    if new_chunk_x == old_chunk.chunk_coord_x && new_chunk_y == old_chunk.chunk_coord_y {
        return;
    }

    let idx = atomicAdd(&mover_count, 1u);
    if idx >= params.particle_count {
        return;
    }
    movers[idx] = GpuMoverRecord(pid, old_slot, new_chunk_x, new_chunk_y);
}
