// G2P pass: gather grid velocity to particles, update v, x, C, F.
// One thread per particle.

#import particles::mpm_types::{GpuParticle, GpuGridNode, MpmParams, bspline_w_dw, node_index, node_in_bounds, mat2_det}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read> grid: array<GpuGridNode>;

const MASS_EPSILON: f32 = 1e-8;

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= params.particle_count {
        return;
    }

    let h = params.h;
    let inv_h = 1.0 / h;
    let dt = params.dt;

    let xp = particles[pid].x;
    var vp = vec2<f32>(0.0, 0.0);
    // New C matrix (velocity gradient * 4/h^2 in MLS-MPM)
    var c00 = 0.0; var c01 = 0.0;
    var c10 = 0.0; var c11 = 0.0;

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
            let mi = grid[nidx].mass;
            if mi < MASS_EPSILON {
                continue;
            }
            // Velocity stored as momentum / mass
            let vi = vec2<f32>(
                grid[nidx].px / mi,
                grid[nidx].py / mi,
            );

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

    // Update F: F_new = (I + dt * C) * F_old
    let f00 = particles[pid].f_a; let f01 = particles[pid].f_b;
    let f10 = particles[pid].f_c; let f11 = particles[pid].f_d;

    // dF = dt * C
    let dc00 = dt * c00; let dc01 = dt * c01;
    let dc10 = dt * c10; let dc11 = dt * c11;

    // F_new = (I + dC) * F = F + dC * F
    let nf00 = f00 + dc00 * f00 + dc10 * f01;
    let nf01 = f01 + dc01 * f00 + dc11 * f01;
    let nf10 = f10 + dc00 * f10 + dc10 * f11;
    let nf11 = f11 + dc01 * f10 + dc11 * f11;

    // Clamp J = det(F_new)
    var j = mat2_det(nf00, nf01, nf10, nf11);
    j = clamp(j, params.j_min, params.j_max);

    // Rescale F to enforce J clamp (scale both columns uniformly)
    let j_actual = mat2_det(nf00, nf01, nf10, nf11);
    var snf00 = nf00; var snf01 = nf01;
    var snf10 = nf10; var snf11 = nf11;
    if abs(j_actual) > 1e-10 {
        let rescale = sqrt(j / j_actual);
        snf00 *= rescale; snf01 *= rescale;
        snf10 *= rescale; snf11 *= rescale;
    }

    // Update position
    let new_x = xp + dt * vp;

    // Write back
    particles[pid].x = new_x;
    particles[pid].v = vp;
    particles[pid].c_a = c00; particles[pid].c_b = c01;
    particles[pid].c_c = c10; particles[pid].c_d = c11;
    particles[pid].f_a = snf00; particles[pid].f_b = snf01;
    particles[pid].f_c = snf10; particles[pid].f_d = snf11;
}
