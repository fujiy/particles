// Marks occupancy for secondary material phase and accumulates centroid terms.

#import particles::mpm_types::{GpuParticle, MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> cell_flags: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_INTERACTION_SECONDARY_COUNT: u32 = 18u;
const LANE_INTERACTION_SECONDARY_SUM_Y_FP: u32 = 19u;

fn cell_index_from_position(pos: vec2<f32>) -> i32 {
    let h = max(params.h, 1.0e-6);
    let cx = i32(floor(pos.x / h));
    let cy = i32(floor(pos.y / h));
    let lx = cx - params.grid_origin_x;
    let ly = cy - params.grid_origin_y;
    if lx < 0 || ly < 0 || u32(lx) >= params.grid_width || u32(ly) >= params.grid_height {
        return -1;
    }
    return i32(u32(ly) * params.grid_width + u32(lx));
}

@compute @workgroup_size(64)
fn mark_interaction_secondary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let p = particles[pid];
    if p.phase_id != params.stats_interaction_secondary_phase_id {
        return;
    }

    let idx = cell_index_from_position(p.x);
    if idx >= 0 {
        atomicStore(&cell_flags[u32(idx)], 1u);
    }

    let y_fp = i32(round(p.x.y * params.stats_position_fp_scale));
    atomicAdd(&stats_scalars[LANE_INTERACTION_SECONDARY_COUNT], 1u);
    atomicAdd(
        &stats_scalars[LANE_INTERACTION_SECONDARY_SUM_Y_FP],
        bitcast<u32>(y_fp),
    );
}
