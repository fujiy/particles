// Apply incremental particle-remove requests by compacting survivors into a scratch buffer.

#import particles::mpm_types::{GpuParticle}

struct GpuWorldEditRemoveParams {
    particle_count: u32,
    remove_count: u32,
    _pad0: vec2<u32>,
}

@group(0) @binding(0) var<uniform> remove_params: GpuWorldEditRemoveParams;
@group(0) @binding(1) var<storage, read> remove_particle_ids: array<u32>;
@group(0) @binding(2) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(3) var<storage, read_write> particle_scratch: array<GpuParticle>;
@group(0) @binding(4) var<storage, read_write> keep_count: atomic<u32>;

fn should_remove_particle(particle_id: u32) -> bool {
    var left = 0u;
    var right = remove_params.remove_count;
    while (left < right) {
        let mid = left + (right - left) / 2u;
        let target_id = remove_particle_ids[mid];
        if (target_id == particle_id) {
            return true;
        }
        if (target_id < particle_id) {
            left = mid + 1u;
        } else {
            right = mid;
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn apply_world_edit_remove(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle_id = gid.x;
    if particle_id >= remove_params.particle_count {
        return;
    }
    if should_remove_particle(particle_id) {
        return;
    }
    let keep_index = atomicAdd(&keep_count, 1u);
    particle_scratch[keep_index] = particles[particle_id];
}
