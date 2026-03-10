// G2P pass: gather grid velocity to particles, update v, x, C, F.
// One thread per particle.

#import particles::mpm_types::{GpuParticle, GpuGridNode, MpmParams, PHASE_WATER, PHASE_GRANULAR_SOIL, phase_is_granular, bspline_w_dw, mat2_det}

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
@group(0) @binding(2) var<storage, read> grid: array<GpuGridNode>;
@group(0) @binding(3) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(4) var<storage, read> terrain_node_solid: array<u32>;

const MASS_EPSILON: f32 = 1e-8;
const APIC_MOMENT_EPS: f32 = 1.0e-10;
const APIC_FULL_SUPPORT_SECOND_MOMENT: f32 = 0.25;
const SVD_SIGMA_MIN: f32 = 1.0e-5;
const DP_NORM_EPS: f32 = 1.0e-8;
const V_VOL_CLAMP: f32 = 6.0;
const INVALID_SLOT: u32 = 0xffffffffu;

struct DpParams {
    lambda: f32,
    mu: f32,
    alpha: f32,
}

struct Svd2 {
    // Row-major U and V entries.
    u00: f32,
    u01: f32,
    u10: f32,
    u11: f32,
    v00: f32,
    v01: f32,
    v10: f32,
    v11: f32,
    sigma1: f32,
    sigma2: f32,
}

struct ProjectionResult {
    f00: f32,
    f01: f32,
    f10: f32,
    f11: f32,
    v_vol: f32,
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

fn node_index_from_global(
    home_slot: u32,
    home_chunk: GpuChunkMeta,
    home_node_origin: vec2<i32>,
    node: vec2<i32>,
    params: MpmParams,
) -> u32 {
    let cdim_i = i32(params.chunk_node_dim);
    if cdim_i <= 0 {
        return INVALID_SLOT;
    }
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
        return INVALID_SLOT;
    }

    var slot_id = home_slot;
    if delta_chunk_x != 0 || delta_chunk_y != 0 {
        let neighbor_idx = neighbor_index_from_delta(delta_chunk_x, delta_chunk_y);
        if neighbor_idx == INVALID_SLOT {
            return INVALID_SLOT;
        }
        let neighbor_slot = home_chunk.neighbor_slot_id[neighbor_idx];
        if neighbor_slot == INVALID_SLOT {
            return INVALID_SLOT;
        }
        slot_id = neighbor_slot;
    }
    let local_x = local.x - delta_chunk_x * cdim_i;
    let local_y = local.y - delta_chunk_y * cdim_i;
    if local_x < 0 || local_y < 0 || local_x >= cdim_i || local_y >= cdim_i {
        return INVALID_SLOT;
    }
    return node_index_from_slot_local(slot_id, u32(local_x), u32(local_y), params);
}

fn node_is_solid(node_idx: u32) -> bool {
    return node_idx == INVALID_SLOT || terrain_node_solid[node_idx] != 0u;
}

fn finite_f32(x: f32) -> bool {
    return x == x && abs(x) <= 1.0e20;
}

fn normalize_or(v: vec2<f32>, fallback: vec2<f32>) -> vec2<f32> {
    let l = length(v);
    if l > 1.0e-8 {
        return v / l;
    }
    return fallback;
}

fn dp_params_for_phase(phase_id: u32, params: MpmParams) -> DpParams {
    if phase_id == PHASE_GRANULAR_SOIL {
        return DpParams(params.dp_lambda_soil, params.dp_mu_soil, params.dp_alpha_soil);
    }
    return DpParams(params.dp_lambda_sand, params.dp_mu_sand, params.dp_alpha_sand);
}

fn svd_2x2_from_storage(f00: f32, f01: f32, f10: f32, f11: f32) -> Svd2 {
    // Stored as column-major [f00, f01, f10, f11] where matrix is:
    // [f00 f10]
    // [f01 f11]
    let a = f00;
    let b = f10;
    let c = f01;
    let d = f11;

    // C = F^T F (row-major entries c00,c01,c11)
    let c00 = a * a + c * c;
    let c01 = a * b + c * d;
    let c11 = b * b + d * d;

    let trace = c00 + c11;
    let diff = c00 - c11;
    let disc = sqrt(max(diff * diff + 4.0 * c01 * c01, 0.0));
    let lambda1 = max(0.5 * (trace + disc), 0.0);
    let lambda2 = max(0.5 * (trace - disc), 0.0);

    let sigma1 = sqrt(max(lambda1, SVD_SIGMA_MIN * SVD_SIGMA_MIN));
    let sigma2 = sqrt(max(lambda2, SVD_SIGMA_MIN * SVD_SIGMA_MIN));

    var v1 = vec2<f32>(1.0, 0.0);
    if abs(c01) > 1.0e-8 {
        v1 = normalize_or(vec2<f32>(lambda1 - c11, c01), vec2<f32>(1.0, 0.0));
    } else if c00 < c11 {
        v1 = vec2<f32>(0.0, 1.0);
    }
    let v2 = vec2<f32>(-v1.y, v1.x);

    // U columns from F * V * Sigma^{-1}
    var u1 = vec2<f32>(
        (a * v1.x + b * v1.y) / sigma1,
        (c * v1.x + d * v1.y) / sigma1,
    );
    var u2 = vec2<f32>(
        (a * v2.x + b * v2.y) / sigma2,
        (c * v2.x + d * v2.y) / sigma2,
    );
    u1 = normalize_or(u1, vec2<f32>(1.0, 0.0));
    u2 = normalize_or(u2, vec2<f32>(-u1.y, u1.x));

    return Svd2(
        u1.x, u2.x,
        u1.y, u2.y,
        v1.x, v2.x,
        v1.y, v2.y,
        sigma1,
        sigma2,
    );
}

fn project_granular_dp(
    f00: f32,
    f01: f32,
    f10: f32,
    f11: f32,
    v_vol: f32,
    phase_id: u32,
    params: MpmParams,
) -> ProjectionResult {
    let dp = dp_params_for_phase(phase_id, params);
    let svd = svd_2x2_from_storage(f00, f01, f10, f11);

    var eps1 = log(max(svd.sigma1, SVD_SIGMA_MIN)) + 0.5 * v_vol;
    var eps2 = log(max(svd.sigma2, SVD_SIGMA_MIN)) + 0.5 * v_vol;

    // Case II: no tensile strength in volumetric expansion.
    let sum_eps0 = eps1 + eps2;
    if sum_eps0 >= 0.0 {
        eps1 = 0.0;
        eps2 = 0.0;
    } else {
        let mean = 0.5 * sum_eps0;
        let dev1 = eps1 - mean;
        let dev2 = eps2 - mean;
        let dev_norm = sqrt(dev1 * dev1 + dev2 * dev2 + DP_NORM_EPS);

        // d=2 => (d*lambda + 2*mu)/(2*mu)
        let volumetric_scale = ((2.0 * dp.lambda + 2.0 * dp.mu) / max(2.0 * dp.mu, 1.0e-8));
        let delta_gamma = dev_norm + volumetric_scale * sum_eps0 * dp.alpha;

        // Case I / III
        if delta_gamma > 0.0 {
            let scale = delta_gamma / dev_norm;
            eps1 -= scale * dev1;
            eps2 -= scale * dev2;
        }
    }

    let sigma_out1 = exp(eps1);
    let sigma_out2 = exp(eps2);

    // Recompose F = U * diag(sigma_out) * V^T (row-major)
    let us00 = svd.u00 * sigma_out1;
    let us01 = svd.u01 * sigma_out2;
    let us10 = svd.u10 * sigma_out1;
    let us11 = svd.u11 * sigma_out2;

    let a = us00 * svd.v00 + us01 * svd.v01; // row0 col0
    let b = us00 * svd.v10 + us01 * svd.v11; // row0 col1
    let c = us10 * svd.v00 + us11 * svd.v01; // row1 col0
    let d = us10 * svd.v10 + us11 * svd.v11; // row1 col1

    var out_f00 = a;
    var out_f01 = c;
    var out_f10 = b;
    var out_f11 = d;

    var j_tr = mat2_det(f00, f01, f10, f11);
    if !finite_f32(j_tr) || j_tr <= 1.0e-8 {
        j_tr = 1.0e-8;
    }

    var j_corr = mat2_det(out_f00, out_f01, out_f10, out_f11);
    if !finite_f32(j_corr) || j_corr <= 1.0e-8 {
        let iso = sqrt(params.j_min);
        let out_v_vol = clamp(v_vol, -V_VOL_CLAMP, V_VOL_CLAMP);
        return ProjectionResult(iso, 0.0, 0.0, iso, out_v_vol);
    }

    // Eq.21: v_vol^{n+1} = v_vol^n - ln det(F_corr) + ln det(F_tr)
    var out_v_vol = clamp(v_vol - log(j_corr) + log(j_tr), -V_VOL_CLAMP, V_VOL_CLAMP);

    // Keep J within runtime stability bounds after projection.
    let target_j = clamp(j_corr, params.j_min, params.j_max);
    let scale = sqrt(target_j / j_corr);
    if finite_f32(scale) && scale > 0.0 {
        out_f00 *= scale;
        out_f01 *= scale;
        out_f10 *= scale;
        out_f11 *= scale;
    }

    if !finite_f32(out_f00) || !finite_f32(out_f01) || !finite_f32(out_f10) || !finite_f32(out_f11) {
        out_f00 = 1.0;
        out_f01 = 0.0;
        out_f10 = 0.0;
        out_f11 = 1.0;
        out_v_vol = 0.0;
    }

    return ProjectionResult(out_f00, out_f01, out_f10, out_f11, out_v_vol);
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }

    let h = params.h;
    let inv_h = 1.0 / h;
    let dt = params.dt;

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
    let xp = p.x;
    let phase_id = p.phase_id;
    let granular = phase_is_granular(phase_id);
    let phase_is_water = phase_id == PHASE_WATER;
    var v_vol = p.v_vol;

    var vp = vec2<f32>(0.0, 0.0);
    // New C matrix reconstructed from the truncated fluid-side moment matrix.
    var c00 = 0.0; var c01 = 0.0;
    var c10 = 0.0; var c11 = 0.0;
    var numer00 = 0.0; var numer01 = 0.0;
    var numer10 = 0.0; var numer11 = 0.0;
    var moment00 = 0.0; var moment01 = 0.0;
    var moment11 = 0.0;
    var mean_dx = vec2<f32>(0.0, 0.0);
    // Water fill fraction φ_p [Eq.9, physics.md]: gathered from grid water_mass.
    var phi_p = 0.0;

    let grid_pos = xp * inv_h;
    let base = vec2<i32>(
        i32(floor(grid_pos.x - 0.5)),
        i32(floor(grid_pos.y - 0.5)),
    );

    var fluid_weight_sum = 0.0;
    for (var oy: i32 = 0; oy < 3; oy++) {
        for (var ox: i32 = 0; ox < 3; ox++) {
            let node = base + vec2<i32>(ox, oy);
            let nidx = node_index_from_global(home_slot, home_chunk, home_node_origin, node, params);
            if node_is_solid(nidx) {
                continue;
            }
            let rel = grid_pos - vec2<f32>(f32(node.x), f32(node.y));
            let wx_dw = bspline_w_dw(rel.x);
            let wy_dw = bspline_w_dw(rel.y);
            let w = wx_dw.x * wy_dw.x;
            if w <= 0.0 { continue; }
            fluid_weight_sum += w;
        }
    }
    if fluid_weight_sum > 1.0e-8 {
        for (var oy: i32 = 0; oy < 3; oy++) {
            for (var ox: i32 = 0; ox < 3; ox++) {
                let node = base + vec2<i32>(ox, oy);
                let nidx = node_index_from_global(home_slot, home_chunk, home_node_origin, node, params);
                if node_is_solid(nidx) {
                    continue;
                }
                let rel = grid_pos - vec2<f32>(f32(node.x), f32(node.y));
                let wx_dw = bspline_w_dw(rel.x);
                let wy_dw = bspline_w_dw(rel.y);
                let w_raw = wx_dw.x * wy_dw.x;
                if w_raw <= 0.0 { continue; }
                let w = w_raw / fluid_weight_sum;

                var mi = 0.0;
                var vi = vec2<f32>(0.0, 0.0);
                if granular {
                    mi = grid[nidx].granular_mass;
                    if mi >= MASS_EPSILON {
                        vi = vec2<f32>(
                            grid[nidx].granular_px / mi,
                            grid[nidx].granular_py / mi,
                        );
                    }
                } else {
                    mi = grid[nidx].water_mass;
                    if mi >= MASS_EPSILON {
                        vi = vec2<f32>(
                            grid[nidx].water_px / mi,
                            grid[nidx].water_py / mi,
                        );
                    }
                }

                vp += w * vi;

                // Boundary-aware APIC/MLS: track the truncated-support first and second moments.
                let xi = vec2<f32>(f32(node.x), f32(node.y)) * h;
                let dx = xi - xp;
                mean_dx += w * dx;
                numer00 += w * vi.x * dx.x;
                numer01 += w * vi.x * dx.y;
                numer10 += w * vi.y * dx.x;
                numer11 += w * vi.y * dx.y;
                moment00 += w * dx.x * dx.x;
                moment01 += w * dx.x * dx.y;
                moment11 += w * dx.y * dx.y;

                // Water fill fraction φ_p = Σ \tilde{w}_ip * (m_i^w / (ρ_0 * h^d))  [Eqs.8-9, 28, physics.md]
                if phase_is_water {
                    let node_phi = grid[nidx].water_mass / (params.rho_ref * h * h);
                    phi_p += w * node_phi;
                }
            }
        }
    }

    numer00 -= vp.x * mean_dx.x;
    numer01 -= vp.x * mean_dx.y;
    numer10 -= vp.y * mean_dx.x;
    numer11 -= vp.y * mean_dx.y;
    moment00 -= mean_dx.x * mean_dx.x;
    moment01 -= mean_dx.x * mean_dx.y;
    moment11 -= mean_dx.y * mean_dx.y;

    let moment_trace = moment00 + moment11;
    let moment_disc = sqrt(max((moment00 - moment11) * (moment00 - moment11) + 4.0 * moment01 * moment01, 0.0));
    let moment_min_eig = max(0.5 * (moment_trace - moment_disc), 0.0);
    let ideal_moment = APIC_FULL_SUPPORT_SECOND_MOMENT * h * h;
    let apic_support_ratio = clamp(moment_min_eig / max(ideal_moment, 1.0e-8), 0.0, 1.0);
    let apic_support_scale = apic_support_ratio * apic_support_ratio;
    let moment_det = moment00 * moment11 - moment01 * moment01;
    let moment_eps = max(APIC_MOMENT_EPS, APIC_MOMENT_EPS * h * h * h * h);
    if moment_det > moment_eps {
        let inv00 = moment11 / moment_det;
        let inv01 = -moment01 / moment_det;
        let inv11 = moment00 / moment_det;

        c00 = numer00 * inv00 + numer01 * inv01;
        c10 = numer00 * inv01 + numer01 * inv11;
        c01 = numer10 * inv00 + numer11 * inv01;
        c11 = numer10 * inv01 + numer11 * inv11;
    }

    // Clamp C norm
    let c_norm_sq = c00*c00 + c01*c01 + c10*c10 + c11*c11;
    let c_max_sq = params.c_max_norm * params.c_max_norm;
    if c_norm_sq > c_max_sq {
        let scale = params.c_max_norm / sqrt(c_norm_sq);
        c00 *= scale; c01 *= scale;
        c10 *= scale; c11 *= scale;
    }

    // Split C usage:
    // - c_def: deformation update (keep C_raw for granular to preserve elastic response at alpha=0)
    // - c_xfer: APIC/PIC transfer term stored to particle for next-step P2G affine
    var c_def00 = c00; var c_def01 = c01;
    var c_def10 = c10; var c_def11 = c11;
    var c_xfer00 = c00; var c_xfer01 = c01;
    var c_xfer10 = c10; var c_xfer11 = c11;
    if granular {
        let alpha_apic_granular = params.alpha_apic_granular * apic_support_scale;
        c_xfer00 *= alpha_apic_granular; c_xfer01 *= alpha_apic_granular;
        c_xfer10 *= alpha_apic_granular; c_xfer11 *= alpha_apic_granular;
    } else {
        let alpha_apic_water = params.alpha_apic_water * apic_support_scale;
        c_def00 *= alpha_apic_water; c_def01 *= alpha_apic_water;
        c_def10 *= alpha_apic_water; c_def11 *= alpha_apic_water;
        c_xfer00 = c_def00; c_xfer01 = c_def01;
        c_xfer10 = c_def10; c_xfer11 = c_def11;
    }

    // Correct the gathered mean velocity back to x_p so PIC/APIC stays consistent on truncated stencils.
    vp -= vec2<f32>(
        c_xfer00 * mean_dx.x + c_xfer10 * mean_dx.y,
        c_xfer01 * mean_dx.x + c_xfer11 * mean_dx.y,
    );

    // Update F: F_new = (I + dt * C) * F_old
    let f00 = p.f_a; let f01 = p.f_b;
    let f10 = p.f_c; let f11 = p.f_d;

    let dc00 = dt * c_def00; let dc01 = dt * c_def01;
    let dc10 = dt * c_def10; let dc11 = dt * c_def11;

    let nf00 = f00 + dc00 * f00 + dc10 * f01;
    let nf01 = f01 + dc01 * f00 + dc11 * f01;
    let nf10 = f10 + dc00 * f10 + dc10 * f11;
    let nf11 = f11 + dc01 * f10 + dc11 * f11;

    // Compute updated F.
    var out_f00 = nf00; var out_f01 = nf01;
    var out_f10 = nf10; var out_f11 = nf11;

    if phase_is_water {
        // Water isotropization: discard shear by setting F = sqrt(J) * I  [Eq.35, physics.md]
        var j_water = mat2_det(nf00, nf01, nf10, nf11);
        if !finite_f32(j_water) || j_water <= 1.0e-6 {
            j_water = 1.0e-6;
        }
        let iso = sqrt(clamp(j_water, params.j_min, params.j_max));
        out_f00 = iso; out_f01 = 0.0;
        out_f10 = 0.0; out_f11 = iso;
        v_vol = 0.0;
    } else if granular {
        // Ensure a valid trial F before DP projection.
        var j_trial = mat2_det(nf00, nf01, nf10, nf11);
        if !finite_f32(j_trial) || j_trial <= 1.0e-6 {
            let iso = sqrt(params.j_min);
            out_f00 = iso; out_f01 = 0.0;
            out_f10 = 0.0; out_f11 = iso;
            v_vol = 0.0;
        } else {
            let projected = project_granular_dp(nf00, nf01, nf10, nf11, v_vol, phase_id, params);
            out_f00 = projected.f00;
            out_f01 = projected.f01;
            out_f10 = projected.f10;
            out_f11 = projected.f11;
            v_vol = projected.v_vol;
        }
    }

    // Update position
    var new_v = vp;
    if !finite_f32(new_v.x) || !finite_f32(new_v.y) {
        new_v = vec2<f32>(0.0, 0.0);
    }
    let new_x = xp + dt * new_v;

    // Write back
    particles[pid].x = new_x;
    particles[pid].v = new_v;
    particles[pid].c_a = c_xfer00; particles[pid].c_b = c_xfer01;
    particles[pid].c_c = c_xfer10; particles[pid].c_d = c_xfer11;
    particles[pid].f_a = out_f00; particles[pid].f_b = out_f01;
    particles[pid].f_c = out_f10; particles[pid].f_d = out_f11;
    particles[pid].v_vol = v_vol;
    particles[pid].phi_p = phi_p;
}
