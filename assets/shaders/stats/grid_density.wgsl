// Collects per-phase grid density maxima and sum/nonzero counters.

#import particles::mpm_types::{GpuGridNode, MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> grid: array<GpuGridNode>;
@group(0) @binding(2) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const LANE_GRID_WATER_PHI_MAX_BITS: u32 = 24u;
const LANE_GRID_WATER_PHI_MEAN_NONZERO_BITS: u32 = 25u; // temp: sum(phi * 10000)
const LANE_GRID_WATER_NONZERO_NODES: u32 = 27u;
const LANE_GRID_GRANULAR_PHI_MAX_BITS: u32 = 28u;
const LANE_GRID_GRANULAR_PHI_MEAN_NONZERO_BITS: u32 = 29u; // temp: sum(phi * 10000)
const LANE_GRID_GRANULAR_NONZERO_NODES: u32 = 31u;

const PHI_SUM_SCALE: f32 = 10000.0;

@compute @workgroup_size(64)
fn collect_grid_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_width * params.grid_height;
    if idx >= total {
        return;
    }

    let denom = max(params.rho_ref * params.h * params.h, 1.0e-8);
    let water_phi = grid[idx].water_mass / denom;
    if water_phi > 1.0e-8 {
        atomicAdd(&stats_scalars[LANE_GRID_WATER_NONZERO_NODES], 1u);
        atomicAdd(
            &stats_scalars[LANE_GRID_WATER_PHI_MEAN_NONZERO_BITS],
            u32(round(water_phi * PHI_SUM_SCALE)),
        );
        atomicMax(
            &stats_scalars[LANE_GRID_WATER_PHI_MAX_BITS],
            bitcast<u32>(water_phi),
        );
    }

    let granular_phi = grid[idx].granular_mass / denom;
    if granular_phi > 1.0e-8 {
        atomicAdd(&stats_scalars[LANE_GRID_GRANULAR_NONZERO_NODES], 1u);
        atomicAdd(
            &stats_scalars[LANE_GRID_GRANULAR_PHI_MEAN_NONZERO_BITS],
            u32(round(granular_phi * PHI_SUM_SCALE)),
        );
        atomicMax(
            &stats_scalars[LANE_GRID_GRANULAR_PHI_MAX_BITS],
            bitcast<u32>(granular_phi),
        );
    }
}
