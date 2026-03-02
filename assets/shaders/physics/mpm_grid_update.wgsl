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

// Terrain boundary: non-penetration + Coulomb stick/slip friction [Eqs.28-29, physics.md].
// Triggers within a thin buffer zone (sdf < threshold) to preemptively block approach.
fn apply_terrain_boundary(v: vec2<f32>, sdf: f32, normal: vec2<f32>, mu: f32, threshold: f32) -> vec2<f32> {
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
    // After normal correction, recompute normal speed (≥ 0 if we just corrected it).
    let vn_post = max(dot(out_v, cn), 0.0);
    let vt_vec = out_v - dot(out_v, cn) * cn;
    let vt_mag = length(vt_vec);
    let stick_limit = mu * vn_post;
    if vt_mag <= stick_limit {
        // Stick: zero tangential.
        out_v -= vt_vec;
    } else if vt_mag > 1e-8 {
        // Slip: scale tangential to friction cone boundary.
        out_v -= vt_vec * (1.0 - stick_limit / vt_mag);
    }

    return out_v;
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
        vw = apply_terrain_boundary(vw, sdf, normal, params.boundary_friction_water, threshold);
    }
    if mg > MASS_EPSILON {
        vg = apply_terrain_boundary(vg, sdf, normal, params.boundary_friction_granular, threshold);
    }

    // Water-granular momentum exchange (symmetric impulse update).
    if mw > MASS_EPSILON && mg > MASS_EPSILON {
        let rel = vw - vg;
        let rel_speed = length(rel);
        if rel_speed > 1e-6 {
            let n = rel / rel_speed;
            let t = vec2<f32>(-n.y, n.x);
            let rel_n = max(dot(rel, n), 0.0);
            let rel_t = dot(rel, t);
            let m_red = (mw * mg) / max(mw + mg, MASS_EPSILON);

            let max_impulse = params.coupling_max_impulse_ratio * m_red * rel_speed;
            let jn_raw = rel_n * m_red * params.coupling_normal_stiffness;
            let jn = clamp(jn_raw, 0.0, max_impulse);

            let jt_raw = -rel_t * m_red * params.coupling_tangent_drag;
            let jt_limit = params.coupling_friction * jn;
            let jt = clamp(jt_raw, -jt_limit, jt_limit);

            let impulse = n * jn + t * jt;
            vw -= impulse / mw;
            vg += impulse / mg;
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
