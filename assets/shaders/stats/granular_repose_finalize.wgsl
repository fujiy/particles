// Finalizes granular repose angle/base-span from collected bounds.

#import particles::mpm_types::{MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_GRANULAR_REPOSE_ANGLE_BITS: u32 = 13u;
const LANE_GRANULAR_REPOSE_BASE_SPAN_BITS: u32 = 14u;
const LANE_REPOSE_COUNT: u32 = 23u;
const LANE_REPOSE_X_MIN_ORDERED_BITS: u32 = 32u;
const LANE_REPOSE_X_MAX_ORDERED_BITS: u32 = 33u;
const LANE_REPOSE_Y_MIN_ORDERED_BITS: u32 = 34u;
const LANE_REPOSE_Y_MAX_ORDERED_BITS: u32 = 35u;
const NAN_BITS: u32 = 0x7fc00000u;

const PI: f32 = 3.141592653589793;

fn ordered_u32_to_float(ordered: u32) -> f32 {
    var bits = 0u;
    if (ordered & 0x80000000u) != 0u {
        bits = ordered ^ 0x80000000u;
    } else {
        bits = ~ordered;
    }
    return bitcast<f32>(bits);
}

@compute @workgroup_size(1)
fn finalize_granular_repose(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }

    let count = atomicLoad(&stats_scalars[LANE_REPOSE_COUNT]);
    if count == 0u {
        atomicStore(&stats_scalars[LANE_GRANULAR_REPOSE_ANGLE_BITS], NAN_BITS);
        atomicStore(&stats_scalars[LANE_GRANULAR_REPOSE_BASE_SPAN_BITS], 0x80000000u);
        return;
    }

    let x_min = ordered_u32_to_float(atomicLoad(&stats_scalars[LANE_REPOSE_X_MIN_ORDERED_BITS]));
    let x_max = ordered_u32_to_float(atomicLoad(&stats_scalars[LANE_REPOSE_X_MAX_ORDERED_BITS]));
    let y_min = ordered_u32_to_float(atomicLoad(&stats_scalars[LANE_REPOSE_Y_MIN_ORDERED_BITS]));
    let y_max = ordered_u32_to_float(atomicLoad(&stats_scalars[LANE_REPOSE_Y_MAX_ORDERED_BITS]));

    let span = max(x_max - x_min, 0.0);
    let height = max(y_max - y_min, 0.0);
    let half_span = max(0.5 * span, 1.0e-6);
    let angle_deg = atan2(height, half_span) * (180.0 / PI);
    let base_span_cells = i32(round(span / max(params.h, 1.0e-6)));

    atomicStore(&stats_scalars[LANE_GRANULAR_REPOSE_ANGLE_BITS], bitcast<u32>(angle_deg));
    atomicStore(
        &stats_scalars[LANE_GRANULAR_REPOSE_BASE_SPAN_BITS],
        bitcast<u32>(base_span_cells),
    );
}
