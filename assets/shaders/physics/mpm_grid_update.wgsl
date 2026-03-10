// Grid update pass: convert momentum to velocity, apply gravity and terrain boundary.
// One workgroup per active tile, one thread per node within the 8x8 tile.

#import particles::mpm_types::{GpuGridNode, MpmParams}

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
@group(0) @binding(1) var<storage, read_write> grid: array<GpuGridNode>;
// Terrain SDF buffer: one f32 per grid node (sdf_m).
// Positive = outside solid, negative = inside solid.
@group(0) @binding(2) var<storage, read> terrain_sdf: array<f32>;
// Terrain normal buffer: 2 x f32 per grid node (nx, ny).
@group(0) @binding(3) var<storage, read> terrain_normal: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(5) var<storage, read> active_tiles: array<vec2<u32>>;

const MASS_EPSILON: f32 = 1e-8;
const INVALID_SLOT: u32 = 0xffffffffu;
const ACTIVE_TILE_NODE_DIM: u32 = 8u;

// Terrain boundary: non-penetration + Coulomb stick/slip friction [Eqs.28-29, physics.md].
// Triggers within a thin buffer zone (sdf < threshold) to preemptively block approach.
fn apply_terrain_boundary(
    v: vec2<f32>,
    sdf: f32,
    normal: vec2<f32>,
    mu: f32,
    threshold: f32,
    normal_projection_scale: f32,
) -> vec2<f32> {
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
        out_v -= vn * cn * clamp(normal_projection_scale, 0.0, 1.0);
    }

    // Eq.29: Coulomb stick/slip on tangential component.
    let vn_post = max(dot(out_v, cn), 0.0);
    let vt_vec = out_v - dot(out_v, cn) * cn;
    let vt_mag = length(vt_vec);
    let stick_limit = mu * vn_post;
    if vt_mag <= stick_limit {
        out_v -= vt_vec;
    } else if vt_mag > 1e-8 && stick_limit > 0.0 {
        out_v -= vt_vec * (stick_limit / vt_mag);
    }

    return out_v;
}

fn granular_phi(idx: u32, params: MpmParams) -> f32 {
    let mg = grid[idx].granular_mass;
    return mg / max(params.rho_ref * params.h * params.h, 1.0e-8);
}

fn node_coord_from_slot_index(idx: u32, params: MpmParams) -> vec2<i32> {
    let cdim = params.chunk_node_dim;
    let nodes_per_chunk = cdim * cdim;
    let chunk_slot = idx / nodes_per_chunk;
    let local_idx = idx - chunk_slot * nodes_per_chunk;

    let local_x = i32(local_idx % cdim);
    let local_y = i32(local_idx / cdim);
    let chunk = chunk_meta[chunk_slot];
    let cdim_i = i32(cdim);

    return vec2<i32>(
        chunk.chunk_coord_x * cdim_i + local_x,
        chunk.chunk_coord_y * cdim_i + local_y,
    );
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

fn shifted_node_index(
    chunk_slot: u32,
    local_x: i32,
    local_y: i32,
    shift_x: i32,
    shift_y: i32,
    params: MpmParams,
) -> u32 {
    let cdim_i = i32(params.chunk_node_dim);
    if cdim_i <= 0 {
        return INVALID_SLOT;
    }
    var nx = local_x + shift_x;
    var ny = local_y + shift_y;
    var delta_x = 0;
    var delta_y = 0;
    if nx < 0 {
        delta_x = -1;
        nx += cdim_i;
    } else if nx >= cdim_i {
        delta_x = 1;
        nx -= cdim_i;
    }
    if ny < 0 {
        delta_y = -1;
        ny += cdim_i;
    } else if ny >= cdim_i {
        delta_y = 1;
        ny -= cdim_i;
    }
    if abs(delta_x) > 1 || abs(delta_y) > 1 {
        return INVALID_SLOT;
    }

    var slot_id = chunk_slot;
    if delta_x != 0 || delta_y != 0 {
        let neighbor_idx = neighbor_index_from_delta(delta_x, delta_y);
        if neighbor_idx == INVALID_SLOT {
            return INVALID_SLOT;
        }
        let neighbor_slot = chunk_meta[chunk_slot].neighbor_slot_id[neighbor_idx];
        if neighbor_slot == INVALID_SLOT {
            return INVALID_SLOT;
        }
        slot_id = neighbor_slot;
    }

    if nx < 0 || ny < 0 || nx >= cdim_i || ny >= cdim_i {
        return INVALID_SLOT;
    }
    return node_index_from_slot_local(slot_id, u32(nx), u32(ny), params);
}

@compute @workgroup_size(64)
fn grid_update(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let cdim = params.chunk_node_dim;
    if cdim == 0u {
        return;
    }
    let record = active_tiles[workgroup_id.x];
    let chunk_slot = record.x;
    let tile_id = record.y;
    let tiles_per_axis = max((cdim + ACTIVE_TILE_NODE_DIM - 1u) / ACTIVE_TILE_NODE_DIM, 1u);
    let tile_x = tile_id % tiles_per_axis;
    let tile_y = tile_id / tiles_per_axis;
    let local_x_u = tile_x * ACTIVE_TILE_NODE_DIM + (local_invocation_index % ACTIVE_TILE_NODE_DIM);
    let local_y_u = tile_y * ACTIVE_TILE_NODE_DIM + (local_invocation_index / ACTIVE_TILE_NODE_DIM);
    if local_x_u >= cdim || local_y_u >= cdim {
        return;
    }
    let local_x = i32(local_x_u);
    let local_y = i32(local_y_u);
    let idx = node_index_from_slot_local(chunk_slot, local_x_u, local_y_u, params);

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
        vw = apply_terrain_boundary(
            vw,
            sdf,
            normal,
            params.boundary_friction_water,
            threshold,
            params.boundary_normal_projection_scale,
        );
    }
    if mg > MASS_EPSILON {
        vg = apply_terrain_boundary(
            vg,
            sdf,
            normal,
            params.boundary_friction_granular,
            threshold,
            params.boundary_normal_projection_scale,
        );
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
        var idx_l = shifted_node_index(chunk_slot, local_x, local_y, -1, 0, params);
        var idx_r = shifted_node_index(chunk_slot, local_x, local_y, 1, 0, params);
        var idx_d = shifted_node_index(chunk_slot, local_x, local_y, 0, -1, params);
        var idx_u = shifted_node_index(chunk_slot, local_x, local_y, 0, 1, params);
        if idx_l == INVALID_SLOT { idx_l = idx; }
        if idx_r == INVALID_SLOT { idx_r = idx; }
        if idx_d == INVALID_SLOT { idx_d = idx; }
        if idx_u == INVALID_SLOT { idx_u = idx; }

        let phi_l = granular_phi(idx_l, params);
        let phi_r = granular_phi(idx_r, params);
        let phi_d = granular_phi(idx_d, params);
        let phi_u = granular_phi(idx_u, params);

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
