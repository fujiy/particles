// Apply CPU-resolved mover results to particle home slots.
// One thread per mover result record.

#import particles::mpm_types::{GpuParticle, MpmParams}

struct GpuMoverResult {
    particle_id: u32,
    new_home_slot_id: u32,
    _pad_a: u32,
    _pad_b: u32,
}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read> mover_result_count: array<u32>;
@group(0) @binding(3) var<storage, read> mover_results: array<GpuMoverResult>;

@compute @workgroup_size(64)
fn apply_mover_results(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rid = gid.x;
    let count = mover_result_count[0];
    if rid >= count {
        return;
    }
    let record = mover_results[rid];
    if record.particle_id >= params.particle_count {
        return;
    }
    particles[record.particle_id].home_chunk_slot_id = record.new_home_slot_id;
}
