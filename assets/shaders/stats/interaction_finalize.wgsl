// Finalizes material interaction statistics (contact ratio, centroid y values).

#import particles::mpm_types::{MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_INTERACTION_PRIMARY_COUNT: u32 = 15u;
const LANE_INTERACTION_CONTACT_COUNT: u32 = 16u;
const LANE_INTERACTION_PRIMARY_SUM_Y_FP: u32 = 17u;
const LANE_INTERACTION_SECONDARY_COUNT: u32 = 18u;
const LANE_INTERACTION_SECONDARY_SUM_Y_FP: u32 = 19u;
const LANE_INTERACTION_CONTACT_RATIO_BITS: u32 = 20u;
const LANE_INTERACTION_PRIMARY_CENTROID_Y_BITS: u32 = 21u;
const LANE_INTERACTION_SECONDARY_CENTROID_Y_BITS: u32 = 22u;
const NAN_BITS: u32 = 0x7fc00000u;

@compute @workgroup_size(1)
fn finalize_interaction(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }

    let primary_count = atomicLoad(&stats_scalars[LANE_INTERACTION_PRIMARY_COUNT]);
    let contact_count = atomicLoad(&stats_scalars[LANE_INTERACTION_CONTACT_COUNT]);
    let secondary_count = atomicLoad(&stats_scalars[LANE_INTERACTION_SECONDARY_COUNT]);

    var ratio = 0.0;
    if primary_count > 0u {
        ratio = f32(contact_count) / f32(primary_count);
    }

    let primary_sum_fp = bitcast<i32>(atomicLoad(&stats_scalars[LANE_INTERACTION_PRIMARY_SUM_Y_FP]));
    let secondary_sum_fp = bitcast<i32>(atomicLoad(&stats_scalars[LANE_INTERACTION_SECONDARY_SUM_Y_FP]));

    var primary_centroid_y = bitcast<f32>(NAN_BITS);
    if primary_count > 0u {
        primary_centroid_y = f32(primary_sum_fp) / (params.stats_position_fp_scale * f32(primary_count));
    }

    var secondary_centroid_y = bitcast<f32>(NAN_BITS);
    if secondary_count > 0u {
        secondary_centroid_y = f32(secondary_sum_fp) / (params.stats_position_fp_scale * f32(secondary_count));
    }

    atomicStore(&stats_scalars[LANE_INTERACTION_CONTACT_RATIO_BITS], bitcast<u32>(ratio));
    atomicStore(
        &stats_scalars[LANE_INTERACTION_PRIMARY_CENTROID_Y_BITS],
        bitcast<u32>(primary_centroid_y),
    );
    atomicStore(
        &stats_scalars[LANE_INTERACTION_SECONDARY_CENTROID_Y_BITS],
        bitcast<u32>(secondary_centroid_y),
    );
}
