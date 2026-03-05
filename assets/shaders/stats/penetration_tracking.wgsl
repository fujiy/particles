// Computes tracked/all counts, mean-y accumulations, and terrain penetration counts.

#import particles::mpm_types::{GpuParticle, MpmParams, node_index, node_in_bounds}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read> terrain_sdf: array<f32>;
@group(0) @binding(3) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_ALL_COUNT: u32 = 6u;
const LANE_ALL_PENETRATION_COUNT: u32 = 7u;
const LANE_ALL_SUM_Y_FP: u32 = 8u;
const LANE_TRACKED_COUNT: u32 = 9u;
const LANE_TRACKED_PENETRATION_COUNT: u32 = 10u;
const LANE_TRACKED_SUM_Y_FP: u32 = 11u;
const LANE_TRACKED_X_MIN_ORDERED_BITS: u32 = 36u;
const LANE_TRACKED_X_MAX_ORDERED_BITS: u32 = 37u;
const LANE_TRACKED_Y_MIN_ORDERED_BITS: u32 = 38u;
const LANE_TRACKED_Y_MAX_ORDERED_BITS: u32 = 39u;

fn float_to_ordered_u32(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    if (bits & 0x80000000u) != 0u {
        return ~bits;
    }
    return bits ^ 0x80000000u;
}

fn is_tracked_phase(phase_id: u32, tracked_phase_id: u32) -> bool {
    if tracked_phase_id == 0xffffffffu {
        return true;
    }
    return phase_id == tracked_phase_id;
}

fn sample_terrain_penetration(pos: vec2<f32>) -> bool {
    let inv_h = 1.0 / max(params.h, 1.0e-6);
    let gp = pos * inv_h;
    let gx = i32(round(gp.x));
    let gy = i32(round(gp.y));
    if !node_in_bounds(gx, gy, params) {
        return false;
    }
    let idx = node_index(gx, gy, params);
    return terrain_sdf[idx] < -max(params.stats_penetration_epsilon_m, 1.0e-6);
}

@compute @workgroup_size(64)
fn compute_penetration_tracking(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }

    let p = particles[pid];
    let y_fp = i32(round(p.x.y * params.stats_position_fp_scale));
    let y_bits = bitcast<u32>(y_fp);
    let penetrates = sample_terrain_penetration(p.x);

    atomicAdd(&stats_scalars[LANE_ALL_COUNT], 1u);
    atomicAdd(&stats_scalars[LANE_ALL_SUM_Y_FP], y_bits);
    if penetrates {
        atomicAdd(&stats_scalars[LANE_ALL_PENETRATION_COUNT], 1u);
    }

    if is_tracked_phase(p.phase_id, params.stats_tracked_phase_id) {
        atomicAdd(&stats_scalars[LANE_TRACKED_COUNT], 1u);
        atomicAdd(&stats_scalars[LANE_TRACKED_SUM_Y_FP], y_bits);
        if penetrates {
            atomicAdd(&stats_scalars[LANE_TRACKED_PENETRATION_COUNT], 1u);
        }
        let x_ordered = float_to_ordered_u32(p.x.x);
        let y_ordered = float_to_ordered_u32(p.x.y);
        atomicMin(&stats_scalars[LANE_TRACKED_X_MIN_ORDERED_BITS], x_ordered);
        atomicMax(&stats_scalars[LANE_TRACKED_X_MAX_ORDERED_BITS], x_ordered);
        atomicMin(&stats_scalars[LANE_TRACKED_Y_MIN_ORDERED_BITS], y_ordered);
        atomicMax(&stats_scalars[LANE_TRACKED_Y_MAX_ORDERED_BITS], y_ordered);
    }
}
