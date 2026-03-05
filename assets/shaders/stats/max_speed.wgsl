// Computes max particle speed (m/s) as float bits in scalar lane.

#import particles::mpm_types::{GpuParticle, MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_MAX_SPEED_BITS: u32 = 5u;

@compute @workgroup_size(64)
fn compute_max_speed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let v = particles[pid].v;
    let speed = sqrt(max(dot(v, v), 0.0));
    let bits = bitcast<u32>(speed);
    atomicMax(&stats_scalars[LANE_MAX_SPEED_BITS], bits);
}
