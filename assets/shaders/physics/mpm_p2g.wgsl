// P2G pass: scatter particle mass, momentum, and stress to grid nodes.
// One thread per particle.
//
// Uses MLS-MPM with quadratic B-spline kernel.
// Atomic float additions via atomicAdd on bitcast u32.

#import particles::mpm_types::{GpuParticle, MpmParams, PHASE_WATER, PHASE_GRANULAR_SOIL, phase_is_granular, bspline_w_dw, node_index, node_in_bounds, mat2_det}

// Fill-fraction thresholds for tension suppression [Eqs.10-11, physics.md].
// Below PHI_MIN: tension fully suppressed (s=0); above PHI_MAX: full tension (s=1).
const PHI_MIN: f32 = 0.10;
const PHI_MAX: f32 = 0.80;

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> particles: array<GpuParticle>;
// Grid as flat u32 for atomic float accumulation (CAS loop).
// GpuGridNode layout:
//   [water_px, water_py, water_mass, water_pad, granular_px, granular_py, granular_mass, granular_pad]
@group(0) @binding(2) var<storage, read_write> grid_atomic: array<atomic<u32>>;

const GRID_NODE_STRIDE_U32: u32 = 8u;
const GRANULAR_STRESS_ABS_MAX: f32 = 1.0e4;

struct DpParams {
    lambda: f32,
    mu: f32,
    alpha: f32,
    k: f32,
    hardening: f32,
}

struct GranularStressResult {
    sigma_xx: f32,
    sigma_xy: f32,
    sigma_yy: f32,
    jp_new: f32,
}

fn atomic_add_lane(lane: u32, val: f32) {
    loop {
        let old_bits = atomicLoad(&grid_atomic[lane]);
        let old_f = bitcast<f32>(old_bits);
        let new_f = old_f + val;
        let new_bits = bitcast<u32>(new_f);
        let result = atomicCompareExchangeWeak(&grid_atomic[lane], old_bits, new_bits);
        if result.exchanged { break; }
    }
}

fn atomic_add_phase(node_idx: u32, phase_id: u32, dp: vec2<f32>, dm: f32) {
    let base = node_idx * GRID_NODE_STRIDE_U32;
    if phase_id == PHASE_WATER {
        atomic_add_lane(base + 0u, dp.x);
        atomic_add_lane(base + 1u, dp.y);
        atomic_add_lane(base + 2u, dm);
    } else {
        atomic_add_lane(base + 4u, dp.x);
        atomic_add_lane(base + 5u, dp.y);
        atomic_add_lane(base + 6u, dm);
    }
}

fn dp_params_for_phase(phase_id: u32, params: MpmParams) -> DpParams {
    if phase_id == PHASE_GRANULAR_SOIL {
        return DpParams(
            params.dp_lambda_soil,
            params.dp_mu_soil,
            params.dp_alpha_soil,
            params.dp_k_soil,
            params.dp_hardening_soil,
        );
    }
    return DpParams(
        params.dp_lambda_sand,
        params.dp_mu_sand,
        params.dp_alpha_sand,
        params.dp_k_sand,
        params.dp_hardening_sand,
    );
}

fn granular_stress_return_mapping(
    f00: f32,
    f01: f32,
    f10: f32,
    f11: f32,
    jp: f32,
    params: MpmParams,
    dp: DpParams,
) -> GranularStressResult {
    // Small-strain surrogate from deformation gradient.
    let eps_xx = f00 - 1.0;
    let eps_yy = f11 - 1.0;
    let eps_xy = 0.5 * (f01 + f10);

    let tr = eps_xx + eps_yy;
    let dev_xx = eps_xx - 0.5 * tr;
    let dev_yy = eps_yy - 0.5 * tr;
    let dev_xy = eps_xy;

    var sigma_xx = 2.0 * dp.mu * dev_xx + dp.lambda * tr;
    var sigma_yy = 2.0 * dp.mu * dev_yy + dp.lambda * tr;
    var sigma_xy = 2.0 * dp.mu * dev_xy;
    var jp_new = jp;

    var mean = 0.5 * (sigma_xx + sigma_yy);
    var sxx = sigma_xx - mean;
    var syy = sigma_yy - mean;
    var sxy = sigma_xy;

    let pressure_comp = max(-mean, 0.0);
    let s_norm = sqrt(max(sxx * sxx + syy * syy + 2.0 * sxy * sxy, 0.0));
    let k_eff = dp.k * (1.0 + dp.hardening * max(1.0 - jp, 0.0));
    let yield_f = s_norm + dp.alpha * pressure_comp - k_eff;
    if yield_f > 0.0 {
        let denom =
            2.0 * dp.mu + dp.lambda * dp.alpha * dp.alpha + dp.hardening + 1e-6;
        let delta_gamma = clamp(yield_f / denom, 0.0, 0.06);
        let shrink = max(0.0, 1.0 - (2.0 * dp.mu * delta_gamma) / (s_norm + 1e-6));
        sxx *= shrink;
        syy *= shrink;
        sxy *= shrink;
        let p_new = max(0.0, pressure_comp - dp.lambda * dp.alpha * delta_gamma);
        sigma_xx = sxx - p_new;
        sigma_yy = syy - p_new;
        sigma_xy = sxy;
        jp_new = clamp(jp * exp(-delta_gamma * dp.hardening), 0.7, 1.6);
    }

    // Granular phase has no tensile capacity in v1.
    mean = 0.5 * (sigma_xx + sigma_yy);
    let max_mean = -max(params.granular_tensile_clamp, 0.0);
    if mean > max_mean {
        let correction = mean - max_mean;
        sigma_xx -= correction;
        sigma_yy -= correction;
    }
    sigma_xx = clamp(sigma_xx, -GRANULAR_STRESS_ABS_MAX, GRANULAR_STRESS_ABS_MAX);
    sigma_xy = clamp(sigma_xy, -GRANULAR_STRESS_ABS_MAX, GRANULAR_STRESS_ABS_MAX);
    sigma_yy = clamp(sigma_yy, -GRANULAR_STRESS_ABS_MAX, GRANULAR_STRESS_ABS_MAX);

    return GranularStressResult(sigma_xx, sigma_xy, sigma_yy, jp_new);
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

    let phase_id = p.phase_id;
    let is_granular = phase_is_granular(phase_id);
    // EOS: p = K * (1/J - 1)  [Eq.7, physics.md]
    var pressure = params.bulk_modulus * (1.0 / j_clamped - 1.0);
    // Tension suppression via fill fraction [Eq.11, physics.md]:
    // negative pressure is smoothly damped by s(φ) = smoothstep(PHI_MIN, PHI_MAX, φ_p).
    if !is_granular && pressure < 0.0 {
        let phi_p = p.phi_p;
        let s = smoothstep(PHI_MIN, PHI_MAX, phi_p);
        pressure *= s;
    }
    var granular_sigma_xx = 0.0;
    var granular_sigma_xy = 0.0;
    var granular_sigma_yy = 0.0;
    var jp_out = p.jp;
    if is_granular {
        let dp = dp_params_for_phase(phase_id, params);
        let mapped = granular_stress_return_mapping(
            f00,
            f01,
            f10,
            f11,
            p.jp,
            params,
            dp,
        );
        granular_sigma_xx = mapped.sigma_xx;
        granular_sigma_xy = mapped.sigma_xy;
        granular_sigma_yy = mapped.sigma_yy;
        jp_out = mapped.jp_new;
    } else {
        jp_out = 1.0;
    }

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

            // Internal force: -V * sigma * grad
            var stress_force = vec2<f32>(vj * pressure * grad.x, vj * pressure * grad.y);
            if is_granular {
                let sigma_grad = vec2<f32>(
                    granular_sigma_xx * grad.x + granular_sigma_xy * grad.y,
                    granular_sigma_xy * grad.x + granular_sigma_yy * grad.y,
                );
                stress_force = vj * sigma_grad;
            }

            // Total momentum contribution
            let dp = vec2<f32>(
                w * mp * affine_v.x + dt * stress_force.x,
                w * mp * affine_v.y + dt * stress_force.y,
            );
            let dm = w * mp;

            atomic_add_phase(nidx, phase_id, dp, dm);
        }
    }

    particles[pid].jp = jp_out;
}
