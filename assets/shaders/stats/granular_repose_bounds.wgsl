// Collects bounding values for granular repose approximation.

#import particles::mpm_types::{GpuParticle, MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_REPOSE_COUNT: u32 = 23u;
const LANE_REPOSE_X_MIN_ORDERED_BITS: u32 = 32u;
const LANE_REPOSE_X_MAX_ORDERED_BITS: u32 = 33u;
const LANE_REPOSE_Y_MIN_ORDERED_BITS: u32 = 34u;
const LANE_REPOSE_Y_MAX_ORDERED_BITS: u32 = 35u;

fn float_to_ordered_u32(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    if (bits & 0x80000000u) != 0u {
        return ~bits;
    }
    return bits ^ 0x80000000u;
}

@compute @workgroup_size(64)
fn collect_granular_repose_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let p = particles[pid];
    if p.phase_id != params.stats_repose_phase_id {
        return;
    }

    atomicAdd(&stats_scalars[LANE_REPOSE_COUNT], 1u);
    let ox = float_to_ordered_u32(p.x.x);
    let oy = float_to_ordered_u32(p.x.y);
    atomicMin(&stats_scalars[LANE_REPOSE_X_MIN_ORDERED_BITS], ox);
    atomicMax(&stats_scalars[LANE_REPOSE_X_MAX_ORDERED_BITS], ox);
    atomicMin(&stats_scalars[LANE_REPOSE_Y_MIN_ORDERED_BITS], oy);
    atomicMax(&stats_scalars[LANE_REPOSE_Y_MAX_ORDERED_BITS], oy);
}
