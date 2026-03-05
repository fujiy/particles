// Clears temporary occupancy flags for material interaction statistics.

#import particles::mpm_types::{MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> cell_flags: array<atomic<u32>>;

@compute @workgroup_size(64)
fn clear_interaction_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_width * params.grid_height;
    if idx >= total {
        return;
    }
    atomicStore(&cell_flags[idx], 0u);
}
