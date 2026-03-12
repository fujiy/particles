// Apply incremental particle-add requests directly on GPU particle storage.
// One invocation per add-op record; each invocation writes up to count_per_cell particles.

#import particles::mpm_types::{GpuParticle, MpmParams, pack_particle_home_slot}

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

struct GpuWorldEditAddParams {
    base_particle_count: u32,
    op_count: u32,
    _pad0: vec2<u32>,
}

struct GpuWorldEditAddOp {
    slot_id: u32,
    local_cell_x: u32,
    local_cell_y: u32,
    count_per_cell: u32,
    particle_offset: u32,
    phase_id: u32,
    material_id: u32,
    mass: f32,
    v0: f32,
}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<uniform> add_params: GpuWorldEditAddParams;
@group(0) @binding(2) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(3) var<storage, read> add_ops: array<GpuWorldEditAddOp>;
@group(0) @binding(4) var<storage, read_write> particles: array<GpuParticle>;

fn particle_grid_axis(count: u32) -> u32 {
    return u32(ceil(sqrt(f32(max(count, 1u)))));
}

@compute @workgroup_size(64)
fn apply_world_edit_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let op_id = gid.x;
    if op_id >= add_params.op_count {
        return;
    }

    let op = add_ops[op_id];
    let axis = particle_grid_axis(op.count_per_cell);
    let cell_size_m = params.h * 2.0;
    let chunk_cell_dim = i32(params.chunk_node_dim / 2u);
    let chunk = chunk_meta[op.slot_id];
    let chunk_cell_origin = vec2<i32>(
        chunk.chunk_coord_x * chunk_cell_dim,
        chunk.chunk_coord_y * chunk_cell_dim,
    );
    let global_cell = chunk_cell_origin + vec2<i32>(i32(op.local_cell_x), i32(op.local_cell_y));
    let cell_min = vec2<f32>(global_cell) * cell_size_m;
    let spacing = cell_size_m / max(f32(axis), 1.0);

    var spawned = 0u;
    for (var y = 0u; y < axis; y++) {
        for (var x = 0u; x < axis; x++) {
            if spawned >= op.count_per_cell {
                break;
            }
            let particle_index = add_params.base_particle_count + op.particle_offset + spawned;
            let offset = vec2<f32>((f32(x) + 0.5) * spacing, (f32(y) + 0.5) * spacing);
            particles[particle_index] = GpuParticle(
                cell_min + offset,
                vec2<f32>(0.0, 0.0),
                op.mass,
                op.v0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                op.phase_id,
                1.0,
                pack_particle_home_slot(op.slot_id, op.material_id),
            );
            spawned += 1u;
        }
    }
}
