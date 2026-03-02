// G2P pass: gather grid velocity to particles, update v, x, C, F.
// One thread per particle.

#import particles::mpm_types::{GpuParticle, GpuGridNode, MpmParams, PHASE_WATER, phase_is_granular, bspline_w_dw, node_index, node_in_bounds, mat2_det}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read> grid: array<GpuGridNode>;

const MASS_EPSILON: f32 = 1e-8;
const GRANULAR_SPEED_MAX: f32 = 12.0;

fn finite_f32(x: f32) -> bool {
    return x == x && abs(x) <= 1.0e20;
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
    let xp = p.x;
    let phase_id = p.phase_id;
    let granular = phase_is_granular(phase_id);
    let phase_is_water = phase_id == PHASE_WATER;
    var jp = max(p.jp, 1e-6);
    var vp = vec2<f32>(0.0, 0.0);
    // New C matrix (velocity gradient * 4/h^2 in MLS-MPM)
    var c00 = 0.0; var c01 = 0.0;
    var c10 = 0.0; var c11 = 0.0;
    // Water fill fraction φ_p [Eq.9, physics.md]: gathered from grid water_mass.
    var phi_p = 0.0;

    let grid_pos = xp * inv_h;
    let base = vec2<i32>(
        i32(floor(grid_pos.x - 0.5)),
        i32(floor(grid_pos.y - 0.5)),
    );

    for (var oy: i32 = 0; oy < 3; oy++) {
        for (var ox: i32 = 0; ox < 3; ox++) {
            let node = base + vec2<i32>(ox, oy);
            if !node_in_bounds(node.x, node.y, params) {
                continue;
            }
            let nidx = node_index(node.x, node.y, params);
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
            if mi < MASS_EPSILON {
                continue;
            }

            let rel = grid_pos - vec2<f32>(f32(node.x), f32(node.y));
            let wx_dw = bspline_w_dw(rel.x);
            let wy_dw = bspline_w_dw(rel.y);
            let w = wx_dw.x * wy_dw.x;
            if w <= 0.0 { continue; }

            vp += w * vi;

            // APIC: C += w * v_i ⊗ (x_i - x_p) * (4 / h^2)
            let xi = vec2<f32>(f32(node.x), f32(node.y)) * h;
            let dx = xi - xp;
            let scale = w * 4.0 * inv_h * inv_h;
            c00 += scale * vi.x * dx.x;
            c01 += scale * vi.x * dx.y;
            c10 += scale * vi.y * dx.x;
            c11 += scale * vi.y * dx.y;

            // Water fill fraction φ_p = Σ w_ip * (m_i^w / (ρ_0 * h^d))  [Eqs.8-9, physics.md]
            if phase_is_water {
                let node_phi = grid[nidx].water_mass / (params.density_ref * h * h);
                phi_p += w * node_phi;
            }
        }
    }

    // Clamp C norm
    let c_norm_sq = c00*c00 + c01*c01 + c10*c10 + c11*c11;
    let c_max_sq = params.c_max_norm * params.c_max_norm;
    if c_norm_sq > c_max_sq {
        let scale = params.c_max_norm / sqrt(c_norm_sq);
        c00 *= scale; c01 *= scale;
        c10 *= scale; c11 *= scale;
    }
    // APIC<->PIC blend [Eq.32, physics.md]: C = alpha_apic * C_raw
    let alpha_apic = select(params.alpha_apic_water, params.alpha_apic_granular, granular);
    c00 *= alpha_apic; c01 *= alpha_apic;
    c10 *= alpha_apic; c11 *= alpha_apic;

    // Update F: F_new = (I + dt * C) * F_old
    let f00 = p.f_a; let f01 = p.f_b;
    let f10 = p.f_c; let f11 = p.f_d;

    // dF = dt * C
    let dc00 = dt * c00; let dc01 = dt * c01;
    let dc10 = dt * c10; let dc11 = dt * c11;

    // F_new = (I + dC) * F = F + dC * F
    let nf00 = f00 + dc00 * f00 + dc10 * f01;
    let nf01 = f01 + dc01 * f00 + dc11 * f01;
    let nf10 = f10 + dc00 * f10 + dc10 * f11;
    let nf11 = f11 + dc01 * f10 + dc11 * f11;

    let j_actual = mat2_det(nf00, nf01, nf10, nf11);
    var target_j = clamp(j_actual, params.j_min, params.j_max);
    if granular {
        // Clamp elastic Jacobian via J / Jp while preserving plastic volume tracker.
        let je = j_actual / jp;
        let je_clamped = clamp(je, params.j_min, params.j_max);
        target_j = je_clamped * jp;
    } else if phase_is_water {
        jp = 1.0;
    }
    if !finite_f32(target_j) || target_j <= 1e-6 {
        target_j = 1e-6;
    }

    // Compute updated F.
    var snf00 = nf00; var snf01 = nf01;
    var snf10 = nf10; var snf11 = nf11;
    if phase_is_water {
        // Water isotropization: discard shear by setting F = sqrt(J) * I  [Eq.35, physics.md]
        // Preserves volume ratio while eliminating accumulated shear ("rubber" artifact).
        let iso = sqrt(target_j);
        snf00 = iso; snf01 = 0.0;
        snf10 = 0.0; snf11 = iso;
    } else {
        // Granular / other: uniform rescale to enforce J clamp.
        if abs(j_actual) > 1e-10 {
            let ratio = target_j / j_actual;
            if finite_f32(ratio) && ratio > 0.0 {
                let rescale = sqrt(ratio);
                snf00 *= rescale; snf01 *= rescale;
                snf10 *= rescale; snf11 *= rescale;
            } else {
                snf00 = 1.0; snf01 = 0.0;
                snf10 = 0.0; snf11 = 1.0;
            }
        }
        if !finite_f32(snf00) || !finite_f32(snf01) || !finite_f32(snf10) || !finite_f32(snf11) {
            snf00 = 1.0; snf01 = 0.0;
            snf10 = 0.0; snf11 = 1.0;
        }
    }

    // Update position
    var new_v = vp;
    if !finite_f32(new_v.x) || !finite_f32(new_v.y) {
        new_v = vec2<f32>(0.0, 0.0);
    } else if granular {
        let speed = length(new_v);
        if speed > GRANULAR_SPEED_MAX {
            new_v *= GRANULAR_SPEED_MAX / speed;
        }
    }
    let new_x = xp + dt * new_v;

    // Write back
    particles[pid].x = new_x;
    particles[pid].v = new_v;
    particles[pid].c_a = c00; particles[pid].c_b = c01;
    particles[pid].c_c = c10; particles[pid].c_d = c11;
    particles[pid].f_a = snf00; particles[pid].f_b = snf01;
    particles[pid].f_c = snf10; particles[pid].f_d = snf11;
    particles[pid].jp = jp;
    particles[pid].phi_p = phi_p;
}
