// Grid update pass: convert momentum to velocity, apply gravity and terrain boundary.
// One thread per grid node.

#import particles::mpm_types::{GpuGridNode, MpmParams}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read_write> grid: array<GpuGridNode>;
// Terrain SDF buffer: one f32 per grid node (sdf_m).
// Positive = outside solid, negative = inside solid.
@group(0) @binding(2) var<storage, read> terrain_sdf: array<f32>;
// Terrain normal buffer: 2 x f32 per grid node (nx, ny).
@group(0) @binding(3) var<storage, read> terrain_normal: array<vec2<f32>>;

const MASS_EPSILON: f32 = 1e-8;
const RECOVERY_SPEED_CAP: f32 = 1.2;

// Terrain boundary: non-penetration + Coulomb stick/slip friction [Eqs.28-29, physics.md].
// Triggers within a thin buffer zone (sdf < threshold) to preemptively block approach.
fn apply_terrain_boundary(v: vec2<f32>, sdf: f32, normal: vec2<f32>, mu: f32, threshold: f32, dt: f32) -> vec2<f32> {
    if sdf >= threshold {
        return v;
    }

    var cn = normal;
    let nlen = length(cn);
    if nlen > 1e-6 {
        cn /= nlen;
    } else {
        cn = vec2<f32>(0.0, 1.0);
    }

    var out_v = v;

    // Eq.28: remove inward normal component (non-penetration).
    let vn = dot(out_v, cn);
    if vn < 0.0 {
        out_v -= vn * cn;
    }

    // Eq.29: Coulomb stick/slip on tangential component.
    let vn_post = max(dot(out_v, cn), 0.0);
    let vt_vec = out_v - dot(out_v, cn) * cn;
    let vt_mag = length(vt_vec);
    let stick_limit = mu * vn_post;
    if vt_mag <= stick_limit {
        out_v -= vt_vec;
    } else if vt_mag > 1e-8 {
        out_v -= vt_vec * (1.0 - stick_limit / vt_mag);
    }

    // Numerical recovery: if a node is already inside terrain, inject a bounded
    // outward speed to reduce long-lived penetration.
    if sdf < 0.0 {
        let recovery_speed = min((-sdf) / max(dt, 1.0e-6), RECOVERY_SPEED_CAP);
        out_v += cn * recovery_speed;
    }

    return out_v;
}

fn granular_phi(idx: u32, params: MpmParams) -> f32 {
    let mg = grid[idx].granular_mass;
    return mg / max(params.rho_ref * params.h * params.h, 1.0e-8);
}

fn node_idx_from_xy(x: i32, y: i32, params: MpmParams) -> u32 {
    return u32(y) * params.grid_width + u32(x);
}

@compute @workgroup_size(64)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_width * params.grid_height;
    if idx >= total {
        return;
    }

    let mw = grid[idx].water_mass;
    let mg = grid[idx].granular_mass;

    var vw = vec2<f32>(0.0, 0.0);
    var vg = vec2<f32>(0.0, 0.0);
    if mw > MASS_EPSILON {
        vw = vec2<f32>(
            grid[idx].water_px / mw,
            grid[idx].water_py / mw,
        );
        vw += vec2<f32>(params.gx, params.gy) * params.dt;
    }
    if mg > MASS_EPSILON {
        vg = vec2<f32>(
            grid[idx].granular_px / mg,
            grid[idx].granular_py / mg,
        );
        vg += vec2<f32>(params.gx, params.gy) * params.dt;
    }

    // Terrain boundary correction for both phases (Coulomb per material).
    let sdf = terrain_sdf[idx];
    let normal = terrain_normal[idx];
    let threshold = params.sdf_velocity_threshold_m;
    if mw > MASS_EPSILON {
        vw = apply_terrain_boundary(vw, sdf, normal, params.boundary_friction_water, threshold, params.dt);
    }
    if mg > MASS_EPSILON {
        vg = apply_terrain_boundary(vg, sdf, normal, params.boundary_friction_granular, threshold, params.dt);
    }

    // Water-granular momentum exchange [Eqs.39-44, physics.md].
    if mw > MASS_EPSILON && mg > MASS_EPSILON {
        let m_red = (mw * mg) / max(mw + mg, MASS_EPSILON);

        // 1) Symmetric drag [Eq.39].
        let rel_before = vw - vg;
        let eta = min(params.coupling_drag_gamma * params.dt, 1.0);
        let j_drag = -eta * m_red * rel_before;
        vw += j_drag / mw;
        vg -= j_drag / mg;

        // 2) Interface normal from granular fill gradient [Eq.41].
        let x = i32(idx % params.grid_width);
        let y = i32(idx / params.grid_width);
        let x_l = max(x - 1, 0);
        let x_r = min(x + 1, i32(params.grid_width) - 1);
        let y_d = max(y - 1, 0);
        let y_u = min(y + 1, i32(params.grid_height) - 1);

        let phi_l = granular_phi(node_idx_from_xy(x_l, y, params), params);
        let phi_r = granular_phi(node_idx_from_xy(x_r, y, params), params);
        let phi_d = granular_phi(node_idx_from_xy(x, y_d, params), params);
        let phi_u = granular_phi(node_idx_from_xy(x, y_u, params), params);

        let grad_phi = vec2<f32>(
            (phi_r - phi_l) / max(2.0 * params.h, 1.0e-6),
            (phi_u - phi_d) / max(2.0 * params.h, 1.0e-6),
        );
        let grad_len = length(grad_phi);

        if grad_len >= params.coupling_interface_min_grad {
            let n = grad_phi / (grad_len + max(params.coupling_interface_normal_eps, 1.0e-9));
            let rel = vw - vg;

            // 3) Non-penetration in interface normal [Eq.43].
            let rel_n = dot(rel, n);
            var jn = vec2<f32>(0.0, 0.0);
            if rel_n < 0.0 {
                jn = -m_red * rel_n * n;
            }

            // 4) Tangential friction cone projection [Eq.44].
            var jt = vec2<f32>(0.0, 0.0);
            let rel_t = rel - rel_n * n;
            let rel_t_mag = length(rel_t);
            let jn_mag = length(jn);
            if rel_t_mag > 1.0e-8 && jn_mag > 0.0 {
                let stick_limit = params.coupling_friction * jn_mag / max(m_red, MASS_EPSILON);
                if rel_t_mag <= stick_limit {
                    jt = -m_red * rel_t;
                } else {
                    jt = -params.coupling_friction * jn_mag * (rel_t / rel_t_mag);
                }
            }

            let impulse = jn + jt;
            vw += impulse / mw;
            vg -= impulse / mg;
        }
    }

    // Write back as phase momentum.
    if mw > MASS_EPSILON {
        grid[idx].water_px = vw.x * mw;
        grid[idx].water_py = vw.y * mw;
    } else {
        grid[idx].water_px = 0.0;
        grid[idx].water_py = 0.0;
        grid[idx].water_mass = 0.0;
    }
    if mg > MASS_EPSILON {
        grid[idx].granular_px = vg.x * mg;
        grid[idx].granular_py = vg.y * mg;
    } else {
        grid[idx].granular_px = 0.0;
        grid[idx].granular_py = 0.0;
        grid[idx].granular_mass = 0.0;
    }
}
