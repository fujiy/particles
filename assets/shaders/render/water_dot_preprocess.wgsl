struct WaterDotParams {
    origin_x: f32,
    origin_y: f32,
    dot_size_m: f32,
    density_threshold: f32,
    atomic_scale: f32,
    splat_radius_dots: f32,
    blur_sigma_dots: f32,
    _pad0: f32,
    width: u32,
    height: u32,
    particle_count: u32,
    blur_radius_dots: u32,
    world_dot_origin_x: i32,
    world_dot_origin_y: i32,
    palette_seed: u32,
    _pad1: u32,
}

struct GpuParticle {
    x: vec2<f32>,
    v: vec2<f32>,
    mass: f32,
    v_0: f32,
    f_00: f32,
    f_01: f32,
    f_10: f32,
    f_11: f32,
    c_00: f32,
    c_01: f32,
    c_10: f32,
    c_11: f32,
    v_vol: f32,
    phase_id: u32,
    phi_p: f32,
    home_chunk_slot_id: u32,
    render_seed: u32,
    _pad_particle: u32,
}

const PHASE_WATER: u32 = 0u;
const PARTICLE_HOME_MATERIAL_SHIFT: u32 = 24u;
const WINNER_MATERIAL_MASK: u32 = 0xffu;
const WINNER_KEY_SHIFT: u32 = 8u;
const WINNER_EMPTY: u32 = 0xffffffffu;
const WINNER_KEY_MAX: u32 = 0x00ffffffu;
const RACE_KEY_SCALE: f32 = 4096.0;

@group(0) @binding(0) var<uniform> params: WaterDotParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> coverage_atomic: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> winner_packed: array<atomic<u32>>;

fn dot_index(x: i32, y: i32) -> u32 {
    return u32(y) * params.width + u32(x);
}

fn in_bounds(x: i32, y: i32) -> bool {
    return x >= 0 && y >= 0 && u32(x) < params.width && u32(y) < params.height;
}

fn gaussian_weight(distance_dots: f32, sigma_dots: f32) -> f32 {
    let sigma = max(sigma_dots, 0.01);
    let t = distance_dots / sigma;
    return exp(-0.5 * t * t);
}

fn particle_material_id(p: GpuParticle) -> u32 {
    return p.home_chunk_slot_id >> PARTICLE_HOME_MATERIAL_SHIFT;
}

fn hash_u32(seed: u32, x: i32, y: i32) -> u32 {
    var state = seed * 0x9e3779b9u;
    state = state ^ (u32(x) * 0x85ebca6bu);
    state = state ^ (u32(y) * 0xc2b2ae35u);
    state = state ^ (state >> 16u);
    state = state * 0x7feb352du;
    state = state ^ (state >> 15u);
    state = state * 0x846ca68bu;
    state = state ^ (state >> 16u);
    return state;
}

fn random_open01(seed: u32, x: i32, y: i32) -> f32 {
    let h = (hash_u32(seed, x, y) & 0x00ffffffu) + 1u;
    return f32(h) / 16777216.0;
}

fn reservoir_payload(key: f32, material_id: u32) -> u32 {
    let quantized = min(u32(key * RACE_KEY_SCALE), WINNER_KEY_MAX);
    return (quantized << WINNER_KEY_SHIFT) | (material_id & WINNER_MATERIAL_MASK);
}

@compute @workgroup_size(64)
fn clear_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }
    atomicStore(&coverage_atomic[idx], 0u);
    atomicStore(&winner_packed[idx], WINNER_EMPTY);
}

@compute @workgroup_size(64)
fn splat_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle_index = gid.x;
    if particle_index >= params.particle_count {
        return;
    }

    let p = particles[particle_index];
    let dot_pos = (p.x - vec2<f32>(params.origin_x, params.origin_y)) / params.dot_size_m;
    let dot_area = max(params.dot_size_m * params.dot_size_m, 1.0e-6);
    let equivalent_dots = max((p.v_0 / dot_area) * 14.0, 14.0);

    let sigma = max(params.blur_sigma_dots, 0.5);
    let radius = max(
        params.splat_radius_dots,
        min(f32(max(params.blur_radius_dots, 1u)), ceil(sigma * 3.0)),
    );

    let min_x = i32(floor(dot_pos.x - radius));
    let max_x = i32(ceil(dot_pos.x + radius));
    let min_y = i32(floor(dot_pos.y - radius));
    let max_y = i32(ceil(dot_pos.y + radius));

    var weight_sum = 0.0;
    for (var y = min_y; y <= max_y; y = y + 1) {
        for (var x = min_x; x <= max_x; x = x + 1) {
            if !in_bounds(x, y) {
                continue;
            }
            let center = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
            let distance_dots = length(center - dot_pos);
            if distance_dots > radius {
                continue;
            }
            weight_sum = weight_sum + gaussian_weight(distance_dots, sigma);
        }
    }
    if weight_sum <= 1.0e-6 {
        return;
    }

    let material_id = select(particle_material_id(p), 0u, p.phase_id == PHASE_WATER);
    for (var y = min_y; y <= max_y; y = y + 1) {
        for (var x = min_x; x <= max_x; x = x + 1) {
            if !in_bounds(x, y) {
                continue;
            }
            let center = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
            let distance_dots = length(center - dot_pos);
            if distance_dots > radius {
                continue;
            }

            let kernel_w = gaussian_weight(distance_dots, sigma) / weight_sum;
            let contribution = equivalent_dots * kernel_w;
            let atomic_units = u32(max(0.0, round(contribution * params.atomic_scale)));
            if atomic_units == 0u {
                continue;
            }

            let idx = dot_index(x, y);
            atomicAdd(&coverage_atomic[idx], atomic_units);

            let world_dot_x = params.world_dot_origin_x + x;
            let world_dot_y = params.world_dot_origin_y + y;
            let u = random_open01(max(p.render_seed, 1u), world_dot_x, world_dot_y);
            let race_key = -log(u) / max(contribution, 1.0e-6);
            atomicMin(&winner_packed[idx], reservoir_payload(race_key, material_id));
        }
    }
}
