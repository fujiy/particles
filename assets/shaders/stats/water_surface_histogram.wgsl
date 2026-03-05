// Builds water particle y-cell histogram for p95 computation.

#import particles::mpm_types::{GpuParticle, MpmParams, PHASE_WATER}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> stats_scalars: array<atomic<u32>>;

const HIST_BASE: u32 = 64u;
const HIST_BINS: u32 = 256u;

@compute @workgroup_size(64)
fn build_water_surface_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }
    let p = particles[pid];
    if p.phase_id != PHASE_WATER {
        return;
    }
    let cell_y = i32(floor(p.x.y / max(params.h, 1.0e-6)));
    let rel = clamp(cell_y - params.grid_origin_y, 0, i32(HIST_BINS) - 1);
    let lane = HIST_BASE + u32(rel);
    atomicAdd(&stats_scalars[lane], 1u);
}
