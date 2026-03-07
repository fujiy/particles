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

fn floor_div_i32(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        return q - 1;
    }
    return q;
}

fn chunk_slot_from_world_pos(pos: vec2<f32>) -> u32 {
    if params.chunk_node_dim == 0u || params.chunk_dims_x == 0u || params.chunk_dims_y == 0u {
        return 0xffffffffu;
    }
    let inv_h = 1.0 / max(params.h, 1.0e-6);
    let node_x = i32(floor(pos.x * inv_h));
    let node_y = i32(floor(pos.y * inv_h));
    let cdim = i32(params.chunk_node_dim);
    let chunk_x = floor_div_i32(node_x, cdim);
    let chunk_y = floor_div_i32(node_y, cdim);
    let lx = chunk_x - params.chunk_origin_x;
    let ly = chunk_y - params.chunk_origin_y;
    if lx < 0 || ly < 0 {
        return 0xffffffffu;
    }
    let lx_u = u32(lx);
    let ly_u = u32(ly);
    if lx_u >= params.chunk_dims_x || ly_u >= params.chunk_dims_y {
        return 0xffffffffu;
    }
    return ly_u * params.chunk_dims_x + lx_u;
}

@compute @workgroup_size(64)
fn clear_chunk_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_id = gid.x;
    if slot_id >= params.resident_chunk_count {
        return;
    }
    atomicStore(&chunk_meta[slot_id].particle_count_next, 0u);
}

@compute @workgroup_size(64)
fn accumulate_chunk_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let p = particles[pid];
    let slot_id = chunk_slot_from_world_pos(p.x);
    if slot_id == 0xffffffffu {
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
    if params.chunk_dims_x == 0u || params.chunk_dims_y == 0u {
        return;
    }

    let dims_x = params.chunk_dims_x;
    let dims_y = params.chunk_dims_y;
    let lx = slot_id % dims_x;
    let ly = slot_id / dims_x;
    let self_count = atomicLoad(&chunk_meta[slot_id].particle_count_next);
    let occupied = self_count > 0u;

    var resident = occupied;
    if !resident {
        let lx_i = i32(lx);
        let ly_i = i32(ly);
        for (var oy = -1; oy <= 1 && !resident; oy++) {
            for (var ox = -1; ox <= 1; ox++) {
                if ox == 0 && oy == 0 {
                    continue;
                }
                let nx = lx_i + ox;
                let ny = ly_i + oy;
                if nx < 0 || ny < 0 || nx >= i32(dims_x) || ny >= i32(dims_y) {
                    continue;
                }
                let nslot = u32(ny) * dims_x + u32(nx);
                if atomicLoad(&chunk_meta[nslot].particle_count_next) > 0u {
                    resident = true;
                    break;
                }
            }
        }
    }

    atomicStore(&chunk_meta[slot_id].particle_count_curr, self_count);
    atomicStore(&chunk_meta[slot_id].occupied_bit_curr, select(0u, 1u, occupied));
    atomicStore(&chunk_meta[slot_id].occupied_bit_next, select(0u, 1u, resident));
}
