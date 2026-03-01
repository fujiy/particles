// P2G pass: scatter particle mass, momentum, and stress to grid nodes.
// One thread per particle.
//
// Uses MLS-MPM with quadratic B-spline kernel.
// Atomic float additions via atomicAdd on bitcast u32.

#import particles::mpm_types::{GpuParticle, MpmParams, bspline_w_dw, node_index, node_in_bounds, mat2_det}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
// Grid as flat u32 for atomic float accumulation (CAS loop).
// GpuGridNode layout: [px: f32, py: f32, mass: f32, _pad: f32] = 4 x u32 per node.
@group(0) @binding(2) var<storage, read_write> grid_atomic: array<atomic<u32>>;

fn atomic_add_px(node_idx: u32, val: f32) {
    let base = node_idx * 4u; // px is offset 0
    loop {
        let old_bits = atomicLoad(&grid_atomic[base]);
        let old_f = bitcast<f32>(old_bits);
        let new_f = old_f + val;
        let new_bits = bitcast<u32>(new_f);
        let result = atomicCompareExchangeWeak(&grid_atomic[base], old_bits, new_bits);
        if result.exchanged { break; }
    }
}

fn atomic_add_py(node_idx: u32, val: f32) {
    let base = node_idx * 4u + 1u;
    loop {
        let old_bits = atomicLoad(&grid_atomic[base]);
        let old_f = bitcast<f32>(old_bits);
        let new_f = old_f + val;
        let new_bits = bitcast<u32>(new_f);
        let result = atomicCompareExchangeWeak(&grid_atomic[base], old_bits, new_bits);
        if result.exchanged { break; }
    }
}

fn atomic_add_mass(node_idx: u32, val: f32) {
    let base = node_idx * 4u + 2u;
    loop {
        let old_bits = atomicLoad(&grid_atomic[base]);
        let old_f = bitcast<f32>(old_bits);
        let new_f = old_f + val;
        let new_bits = bitcast<u32>(new_f);
        let result = atomicCompareExchangeWeak(&grid_atomic[base], old_bits, new_bits);
        if result.exchanged { break; }
    }
}

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }

    let p = particles[pid];
    let h = params.h;
    let inv_h = 1.0 / h;
    let dt = params.dt;

    // Particle state
    let xp = p.x;
    let vp = p.v;
    let mp = p.mass;
    let v0p = p.v_rest;

    // F matrix components
    let f00 = p.f_a; let f01 = p.f_b;
    let f10 = p.f_c; let f11 = p.f_d;

    // C matrix components (APIC affine)
    let c00 = p.c_a; let c01 = p.c_b;
    let c10 = p.c_c; let c11 = p.c_d;

    // J = det(F)
    let j = mat2_det(f00, f01, f10, f11);
    // Clamp J for stability
    let j_clamped = clamp(j, params.j_min, params.j_max);

    // Pressure (weak compressibility EOS)
    let pressure = params.bulk_modulus * max(j_clamped - 1.0, 0.0);

    let vj = v0p * j_clamped;

    // Quadratic B-spline stencil: base node
    let grid_pos = xp * inv_h;
    let base = vec2<i32>(
        i32(floor(grid_pos.x - 0.5)),
        i32(floor(grid_pos.y - 0.5)),
    );

    // Scatter to 3x3 stencil
    for (var oy: i32 = 0; oy < 3; oy++) {
        for (var ox: i32 = 0; ox < 3; ox++) {
            let node = base + vec2<i32>(ox, oy);
            if !node_in_bounds(node.x, node.y, params) {
                continue;
            }
            let nidx = node_index(node.x, node.y, params);

            let rel = grid_pos - vec2<f32>(f32(node.x), f32(node.y));
            let wx_dw = bspline_w_dw(rel.x);
            let wy_dw = bspline_w_dw(rel.y);
            let w = wx_dw.x * wy_dw.x;
            if w <= 0.0 { continue; }

            let grad = vec2<f32>(wx_dw.y * wy_dw.x, wx_dw.x * wy_dw.y) * inv_h;

            // Affine momentum: mp * (vp + C * (xi - xp))
            let dx = vec2<f32>(f32(node.x), f32(node.y)) * h - xp;
            let affine_v = vec2<f32>(
                vp.x + c00 * dx.x + c10 * dx.y,
                vp.y + c01 * dx.x + c11 * dx.y,
            );

            // Stress divergence: vj * pressure * grad (isotropic)
            let stress_contrib = vec2<f32>(
                vj * pressure * grad.x,
                vj * pressure * grad.y,
            );

            // Total momentum contribution
            let dp = vec2<f32>(
                w * mp * affine_v.x - dt * stress_contrib.x,
                w * mp * affine_v.y - dt * stress_contrib.y,
            );
            let dm = w * mp;

            atomic_add_px(nidx, dp.x);
            atomic_add_py(nidx, dp.y);
            atomic_add_mass(nidx, dm);
        }
    }
}
