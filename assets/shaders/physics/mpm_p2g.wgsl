// P2G pass: scatter particle mass, momentum, and stress to grid nodes.
// One thread per particle.
//
// Uses MLS-MPM with quadratic B-spline kernel.
// Atomic float additions via atomicAdd on bitcast u32.

#import particles::mpm_types::{GpuParticle, MpmParams, PHASE_WATER, PHASE_GRANULAR_SOIL, phase_is_granular, bspline_w_dw, mat2_det}

// Fill-fraction thresholds for tension suppression [Eqs.10-11, physics.md].
// Below PHI_MIN: tension fully suppressed (s=0); above PHI_MAX: full tension (s=1).
const PHI_MIN: f32 = 0.10;
const PHI_MAX: f32 = 0.80;
const INVALID_SLOT: u32 = 0xffffffffu;

struct GpuChunkMeta {
    chunk_coord_x: i32,
    chunk_coord_y: i32,
    neighbor_slot_id: array<u32, 8>,
    active_tile_mask: u32,
    particle_count_curr: u32,
    particle_count_next: u32,
    occupied_bit_curr: u32,
    occupied_bit_next: u32,
}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> particles: array<GpuParticle>;
// Grid as flat u32 for atomic float accumulation (CAS loop).
// GpuGridNode layout:
//   [water_px, water_py, water_mass, water_pad, granular_px, granular_py, granular_mass, granular_pad]
@group(0) @binding(2) var<storage, read_write> grid_atomic: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> chunk_meta: array<GpuChunkMeta>;

const GRID_NODE_STRIDE_U32: u32 = 8u;

struct ElasticParams {
    lambda: f32,
    mu: f32,
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

fn neighbor_index_from_delta(dx: i32, dy: i32) -> u32 {
    if dx == -1 && dy == -1 { return 0u; }
    if dx ==  0 && dy == -1 { return 1u; }
    if dx ==  1 && dy == -1 { return 2u; }
    if dx == -1 && dy ==  0 { return 3u; }
    if dx ==  1 && dy ==  0 { return 4u; }
    if dx == -1 && dy ==  1 { return 5u; }
    if dx ==  0 && dy ==  1 { return 6u; }
    if dx ==  1 && dy ==  1 { return 7u; }
    return INVALID_SLOT;
}

fn node_index_from_slot_local(slot_id: u32, local_x: u32, local_y: u32, params: MpmParams) -> u32 {
    let nodes_per_chunk = params.chunk_node_dim * params.chunk_node_dim;
    return slot_id * nodes_per_chunk + local_y * params.chunk_node_dim + local_x;
}

fn elastic_params_for_phase(phase_id: u32, params: MpmParams) -> ElasticParams {
    if phase_id == PHASE_GRANULAR_SOIL {
        return ElasticParams(params.dp_lambda_soil, params.dp_mu_soil);
    }
    return ElasticParams(params.dp_lambda_sand, params.dp_mu_sand);
}

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }

    let p = particles[pid];
    let home_slot = p.home_chunk_slot_id;
    if home_slot >= params.resident_chunk_count || home_slot == INVALID_SLOT {
        return;
    }
    let home_chunk = chunk_meta[home_slot];
    let cdim_i = i32(params.chunk_node_dim);
    if cdim_i <= 0 {
        return;
    }
    let home_node_origin = vec2<i32>(home_chunk.chunk_coord_x * cdim_i, home_chunk.chunk_coord_y * cdim_i);
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
    let j_clamped = clamp(j, params.j_min, params.j_max);

    let phase_id = p.phase_id;
    let is_granular = phase_is_granular(phase_id);

    // EOS: p = K * (1/J - 1)  [Eq.7, physics.md]
    var pressure = params.bulk_modulus * (1.0 / j_clamped - 1.0);
    // Tension suppression via fill fraction [Eq.11, physics.md]:
    // negative pressure is smoothly damped by s(φ) = smoothstep(PHI_MIN, PHI_MAX, φ_p).
    if !is_granular && pressure < 0.0 {
        let s = smoothstep(PHI_MIN, PHI_MAX, p.phi_p);
        pressure *= s;
    }

    // Water internal force uses V * p * grad where V = V0 * J.
    let vj = v0p * j_clamped;

    // Granular internal force uses Eq.24 with Neo-Hookean P [Eq.13].
    var a00 = 0.0;
    var a01 = 0.0;
    var a10 = 0.0;
    var a11 = 0.0;
    if is_granular {
        let elastic = elastic_params_for_phase(phase_id, params);
        let j_safe = max(j, 1.0e-6);
        let inv_j = 1.0 / j_safe;
        let log_j = log(j_safe);

        // F^{-T}
        let invt00 = f11 * inv_j;
        let invt01 = -f10 * inv_j;
        let invt10 = -f01 * inv_j;
        let invt11 = f00 * inv_j;

        // P = mu(F - F^{-T}) + lambda * ln(J) * F^{-T}
        let p00 = elastic.mu * (f00 - invt00) + elastic.lambda * log_j * invt00;
        let p01 = elastic.mu * (f01 - invt01) + elastic.lambda * log_j * invt01;
        let p10 = elastic.mu * (f10 - invt10) + elastic.lambda * log_j * invt10;
        let p11 = elastic.mu * (f11 - invt11) + elastic.lambda * log_j * invt11;

        // A = P * F^T (row-major A entries)
        a00 = p00 * f00 + p10 * f10;
        a01 = p00 * f01 + p10 * f11;
        a10 = p01 * f00 + p11 * f10;
        a11 = p01 * f01 + p11 * f11;
    }

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
            let local = node - home_node_origin;
            var delta_chunk_x = 0;
            var delta_chunk_y = 0;
            if local.x < 0 {
                delta_chunk_x = -1;
            } else if local.x >= cdim_i {
                delta_chunk_x = 1;
            }
            if local.y < 0 {
                delta_chunk_y = -1;
            } else if local.y >= cdim_i {
                delta_chunk_y = 1;
            }
            if abs(delta_chunk_x) > 1 || abs(delta_chunk_y) > 1 {
                continue;
            }

            var slot_id = home_slot;
            if delta_chunk_x != 0 || delta_chunk_y != 0 {
                let neighbor_idx = neighbor_index_from_delta(delta_chunk_x, delta_chunk_y);
                if neighbor_idx == INVALID_SLOT {
                    continue;
                }
                let neighbor_slot = home_chunk.neighbor_slot_id[neighbor_idx];
                if neighbor_slot == INVALID_SLOT {
                    continue;
                }
                slot_id = neighbor_slot;
            }

            let local_x = local.x - delta_chunk_x * cdim_i;
            let local_y = local.y - delta_chunk_y * cdim_i;
            if local_x < 0 || local_y < 0 || local_x >= cdim_i || local_y >= cdim_i {
                continue;
            }
            let nidx = node_index_from_slot_local(slot_id, u32(local_x), u32(local_y), params);

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

            // Internal force contribution (without dt): Eq.24.
            // Here `grad` is ∇_{x_p} w_ip (particle-position gradient), so the sign is
            // opposite to the usual ∇_{x_i} w_ip form used in the equation text.
            var stress_force = vec2<f32>(vj * pressure * grad.x, vj * pressure * grad.y);
            if is_granular {
                let a_grad = vec2<f32>(
                    a00 * grad.x + a01 * grad.y,
                    a10 * grad.x + a11 * grad.y,
                );
                stress_force = -v0p * a_grad;
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
}
