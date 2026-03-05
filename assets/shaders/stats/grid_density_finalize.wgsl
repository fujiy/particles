// Finalizes mean/p99 fields for grid-density diagnostics.

@group(0) @binding(0) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_GRID_WATER_PHI_MAX_BITS: u32 = 24u;
const LANE_GRID_WATER_PHI_MEAN_NONZERO_BITS: u32 = 25u; // temp in, mean bits out
const LANE_GRID_WATER_PHI_P99_BITS: u32 = 26u;
const LANE_GRID_WATER_NONZERO_NODES: u32 = 27u;
const LANE_GRID_GRANULAR_PHI_MAX_BITS: u32 = 28u;
const LANE_GRID_GRANULAR_PHI_MEAN_NONZERO_BITS: u32 = 29u; // temp in, mean bits out
const LANE_GRID_GRANULAR_PHI_P99_BITS: u32 = 30u;
const LANE_GRID_GRANULAR_NONZERO_NODES: u32 = 31u;

const PHI_SUM_SCALE: f32 = 10000.0;

@compute @workgroup_size(1)
fn finalize_grid_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }

    let water_nonzero = atomicLoad(&stats_scalars[LANE_GRID_WATER_NONZERO_NODES]);
    let water_sum_scaled = atomicLoad(&stats_scalars[LANE_GRID_WATER_PHI_MEAN_NONZERO_BITS]);
    var water_mean = 0.0;
    if water_nonzero > 0u {
        water_mean = f32(water_sum_scaled) / (PHI_SUM_SCALE * f32(water_nonzero));
    }
    let water_max_bits = atomicLoad(&stats_scalars[LANE_GRID_WATER_PHI_MAX_BITS]);

    let granular_nonzero = atomicLoad(&stats_scalars[LANE_GRID_GRANULAR_NONZERO_NODES]);
    let granular_sum_scaled = atomicLoad(&stats_scalars[LANE_GRID_GRANULAR_PHI_MEAN_NONZERO_BITS]);
    var granular_mean = 0.0;
    if granular_nonzero > 0u {
        granular_mean = f32(granular_sum_scaled) / (PHI_SUM_SCALE * f32(granular_nonzero));
    }
    let granular_max_bits = atomicLoad(&stats_scalars[LANE_GRID_GRANULAR_PHI_MAX_BITS]);

    // Current implementation keeps p99 as max until dedicated histogram pass is introduced.
    atomicStore(&stats_scalars[LANE_GRID_WATER_PHI_MEAN_NONZERO_BITS], bitcast<u32>(water_mean));
    atomicStore(&stats_scalars[LANE_GRID_WATER_PHI_P99_BITS], water_max_bits);
    atomicStore(
        &stats_scalars[LANE_GRID_GRANULAR_PHI_MEAN_NONZERO_BITS],
        bitcast<u32>(granular_mean),
    );
    atomicStore(&stats_scalars[LANE_GRID_GRANULAR_PHI_P99_BITS], granular_max_bits);
}
