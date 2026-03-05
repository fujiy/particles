// Computes contact ratio numerator/denominator for primary phase against secondary occupancy.

#import particles::mpm_types::{GpuParticle, MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> cell_flags: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_INTERACTION_PRIMARY_COUNT: u32 = 15u;
const LANE_INTERACTION_CONTACT_COUNT: u32 = 16u;
const LANE_INTERACTION_PRIMARY_SUM_Y_FP: u32 = 17u;

fn cell_xy_from_position(pos: vec2<f32>) -> vec2<i32> {
    let h = max(params.h, 1.0e-6);
    let cx = i32(floor(pos.x / h));
    let cy = i32(floor(pos.y / h));
    return vec2<i32>(cx, cy);
}

fn cell_index(cx: i32, cy: i32) -> i32 {
    let lx = cx - params.grid_origin_x;
    let ly = cy - params.grid_origin_y;
    if lx < 0 || ly < 0 || u32(lx) >= params.grid_width || u32(ly) >= params.grid_height {
        return -1;
    }
    return i32(u32(ly) * params.grid_width + u32(lx));
}

@compute @workgroup_size(64)
fn compute_interaction_primary_contact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let p = particles[pid];
    if p.phase_id != params.stats_interaction_primary_phase_id {
        return;
    }

    atomicAdd(&stats_scalars[LANE_INTERACTION_PRIMARY_COUNT], 1u);
    let y_fp = i32(round(p.x.y * params.stats_position_fp_scale));
    atomicAdd(
        &stats_scalars[LANE_INTERACTION_PRIMARY_SUM_Y_FP],
        bitcast<u32>(y_fp),
    );

    let c = cell_xy_from_position(p.x);
    var touching = false;
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let idx = cell_index(c.x + ox, c.y + oy);
            if idx < 0 {
                continue;
            }
            if atomicLoad(&cell_flags[u32(idx)]) > 0u {
                touching = true;
                break;
            }
        }
        if touching {
            break;
        }
    }

    if touching {
        atomicAdd(&stats_scalars[LANE_INTERACTION_CONTACT_COUNT], 1u);
    }
}
