// Counts particle phases/material groups.

#import particles::mpm_types::{GpuParticle, MpmParams, PHASE_WATER, PHASE_GRANULAR_SOIL, PHASE_GRANULAR_SAND}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_PHASE_WATER: u32 = 1u;
const LANE_PHASE_GRANULAR_SOIL: u32 = 2u;
const LANE_PHASE_GRANULAR_SAND: u32 = 3u;
const LANE_PHASE_UNKNOWN: u32 = 4u;

fn lane_for_phase(phase_id: u32) -> u32 {
    if phase_id == PHASE_WATER {
        return LANE_PHASE_WATER;
    }
    if phase_id == PHASE_GRANULAR_SOIL {
        return LANE_PHASE_GRANULAR_SOIL;
    }
    if phase_id == PHASE_GRANULAR_SAND {
        return LANE_PHASE_GRANULAR_SAND;
    }
    return LANE_PHASE_UNKNOWN;
}

@compute @workgroup_size(64)
fn count_phase_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let lane = lane_for_phase(particles[pid].phase_id);
    atomicAdd(&stats_scalars[lane], 1u);
}
