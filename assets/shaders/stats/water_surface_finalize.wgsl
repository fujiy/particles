// Reduces water y-cell histogram into p95 cell index.

#import particles::mpm_types::{MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_WATER_SURFACE_P95_CELL_BITS: u32 = 12u;
const HIST_BASE: u32 = 64u;
const HIST_BINS: u32 = 256u;

@compute @workgroup_size(1)
fn finalize_water_surface_p95(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }

    var total = 0u;
    for (var i = 0u; i < HIST_BINS; i++) {
        total += atomicLoad(&stats_scalars[HIST_BASE + i]);
    }
    if total == 0u {
        atomicStore(&stats_scalars[LANE_WATER_SURFACE_P95_CELL_BITS], 0x80000000u);
        return;
    }

    let target_index = u32(floor(f32(total) * 0.95));
    var cumulative = 0u;
    var selected_bin = 0u;
    for (var i = 0u; i < HIST_BINS; i++) {
        cumulative += atomicLoad(&stats_scalars[HIST_BASE + i]);
        if cumulative > target_index {
            selected_bin = i;
            break;
        }
    }

    let cell_y = i32(selected_bin) + params.grid_origin_y;
    atomicStore(
        &stats_scalars[LANE_WATER_SURFACE_P95_CELL_BITS],
        bitcast<u32>(cell_y),
    );
}
