// Counts total active particles.

#import particles::mpm_types::{GpuParticle, MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_TOTAL_PARTICLES: u32 = 0u;

@compute @workgroup_size(64)
fn count_total_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let _phase_id = particles[pid].phase_id;
    atomicAdd(&stats_scalars[LANE_TOTAL_PARTICLES], 1u);
}
