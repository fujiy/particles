// Grid clear pass: zero out all active grid nodes.
// One thread per grid node.

#import particles::mpm_types::{GpuGridNode, MpmParams, node_in_bounds}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<GpuGridNode>;

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_width * params.grid_height;
    if idx >= total {
        return;
    }
    grid[idx].px = 0.0;
    grid[idx].py = 0.0;
    grid[idx].mass = 0.0;
    grid[idx].pad = 0.0;
}
