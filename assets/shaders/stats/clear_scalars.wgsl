// Clears all statistics lanes and initializes extrema sentinels.

@group(0) @binding(0) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const TOTAL_LANES: u32 = 320u;

const LANE_WATER_SURFACE_P95_CELL_BITS: u32 = 12u;
const LANE_GRANULAR_REPOSE_ANGLE_BITS: u32 = 13u;
const LANE_GRANULAR_REPOSE_BASE_SPAN_BITS: u32 = 14u;
const LANE_INTERACTION_PRIMARY_CENTROID_Y_BITS: u32 = 21u;
const LANE_INTERACTION_SECONDARY_CENTROID_Y_BITS: u32 = 22u;

const LANE_REPOSE_X_MIN_ORDERED_BITS: u32 = 32u;
const LANE_REPOSE_X_MAX_ORDERED_BITS: u32 = 33u;
const LANE_REPOSE_Y_MIN_ORDERED_BITS: u32 = 34u;
const LANE_REPOSE_Y_MAX_ORDERED_BITS: u32 = 35u;
const LANE_TRACKED_X_MIN_ORDERED_BITS: u32 = 36u;
const LANE_TRACKED_X_MAX_ORDERED_BITS: u32 = 37u;
const LANE_TRACKED_Y_MIN_ORDERED_BITS: u32 = 38u;
const LANE_TRACKED_Y_MAX_ORDERED_BITS: u32 = 39u;

const ORDERED_POS_INF: u32 = 0xff800000u;
const ORDERED_NEG_INF: u32 = 0x007fffffu;
const NAN_BITS: u32 = 0x7fc00000u;

@compute @workgroup_size(64)
fn clear_stats_scalars(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    if lane >= TOTAL_LANES {
        return;
    }
    var value = 0u;
    if lane == LANE_WATER_SURFACE_P95_CELL_BITS || lane == LANE_GRANULAR_REPOSE_BASE_SPAN_BITS {
        value = 0x80000000u; // i32::MIN sentinel
    }
    if lane == LANE_GRANULAR_REPOSE_ANGLE_BITS
        || lane == LANE_INTERACTION_PRIMARY_CENTROID_Y_BITS
        || lane == LANE_INTERACTION_SECONDARY_CENTROID_Y_BITS
    {
        value = NAN_BITS;
    }

    if lane == LANE_REPOSE_X_MIN_ORDERED_BITS
        || lane == LANE_REPOSE_Y_MIN_ORDERED_BITS
        || lane == LANE_TRACKED_X_MIN_ORDERED_BITS
        || lane == LANE_TRACKED_Y_MIN_ORDERED_BITS
    {
        value = ORDERED_POS_INF;
    }
    if lane == LANE_REPOSE_X_MAX_ORDERED_BITS
        || lane == LANE_REPOSE_Y_MAX_ORDERED_BITS
        || lane == LANE_TRACKED_X_MAX_ORDERED_BITS
        || lane == LANE_TRACKED_Y_MAX_ORDERED_BITS
    {
        value = ORDERED_NEG_INF;
    }

    atomicStore(&stats_scalars[lane], value);
}
