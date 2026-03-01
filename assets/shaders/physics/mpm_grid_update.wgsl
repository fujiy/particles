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
@compute @workgroup_size(64)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_width * params.grid_height;
    if idx >= total {
        return;
    }

    let mi = grid[idx].mass;
    if mi < MASS_EPSILON {
        // Clear velocity for empty nodes
        grid[idx].px = 0.0;
        grid[idx].py = 0.0;
        return;
    }

    // Convert momentum to velocity
    var vx = grid[idx].px / mi;
    var vy = grid[idx].py / mi;

    // Apply gravity
    vx += params.gx * params.dt;
    vy += params.gy * params.dt;

    // Terrain boundary correction
    let sdf = terrain_sdf[idx];
    let nx = terrain_normal[idx].x;
    let ny = terrain_normal[idx].y;

    let threshold = params.sdf_velocity_threshold_m;

    if sdf < threshold {
        // Match CPU boundary logic: use SDF normal (fallback to +Y).
        var cnx = nx;
        var cny = ny;
        let nlen = sqrt(cnx * cnx + cny * cny);
        if nlen > 1e-6 {
            cnx /= nlen;
            cny /= nlen;
        } else {
            cnx = 0.0;
            cny = 1.0;
        }

        var corrected_vx = vx;
        var corrected_vy = vy;
        let normal_speed = corrected_vx * cnx + corrected_vy * cny;
        if normal_speed < 0.0 {
            corrected_vx -= cnx * normal_speed;
            corrected_vy -= cny * normal_speed;
        }

        let depth = threshold - sdf;
        if depth > 0.0 {
            let push_speed = clamp(
                depth * max(params.deep_push_gain_per_s, 0.0),
                0.0,
                max(params.deep_push_speed_cap_mps, 0.0),
            );
            corrected_vx += cnx * push_speed;
            corrected_vy += cny * push_speed;
        }

        let normal_speed2 = corrected_vx * cnx + corrected_vy * cny;
        let tang_x = corrected_vx - normal_speed2 * cnx;
        let tang_y = corrected_vy - normal_speed2 * cny;
        vx = corrected_vx - params.tangential_damping * tang_x;
        vy = corrected_vy - params.tangential_damping * tang_y;
    }

    // Write back as momentum (other passes read velocity from px/py / mass)
    // Store velocity directly (G2P will read px/py as velocity when mass>0)
    grid[idx].px = vx * mi;
    grid[idx].py = vy * mi;
}
