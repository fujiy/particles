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
    palette_seed: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
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
}

const PHASE_WATER: u32 = 0u;
const MATERIAL_STONE_GRANULAR: u32 = 2u;
const MATERIAL_SOIL_GRANULAR: u32 = 4u;
const MATERIAL_SAND_GRANULAR: u32 = 6u;
const MATERIAL_GRASS_GRANULAR: u32 = 8u;
const PARTICLE_HOME_MATERIAL_SHIFT: u32 = 24u;

@group(0) @binding(0) var<uniform> params: WaterDotParams;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var<storage, read_write> density_atomic_water: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> density_atomic_stone: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> density_atomic_soil: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> density_atomic_sand: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> density_atomic_grass: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> blur_tmp: array<f32>;
@group(0) @binding(8) var<storage, read_write> blurred_density_water: array<f32>;
@group(0) @binding(9) var<storage, read_write> blurred_density_stone: array<f32>;
@group(0) @binding(10) var<storage, read_write> blurred_density_soil: array<f32>;
@group(0) @binding(11) var<storage, read_write> blurred_density_sand: array<f32>;
@group(0) @binding(12) var<storage, read_write> blurred_density_grass: array<f32>;

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

@compute @workgroup_size(64)
fn clear_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    atomicStore(&density_atomic_water[idx], 0u);
    atomicStore(&density_atomic_stone[idx], 0u);
    atomicStore(&density_atomic_soil[idx], 0u);
    atomicStore(&density_atomic_sand[idx], 0u);
    atomicStore(&density_atomic_grass[idx], 0u);
    blur_tmp[idx] = 0.0;
    blurred_density_water[idx] = 0.0;
    blurred_density_stone[idx] = 0.0;
    blurred_density_soil[idx] = 0.0;
    blurred_density_sand[idx] = 0.0;
    blurred_density_grass[idx] = 0.0;
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
    // Convert rest area to equivalent dot count and keep a floor so sparse regions
    // still form a continuous sheet after blur/threshold.
    let equivalent_dots = max((p.v_0 / dot_area) * 14.0, 14.0);

    let radius = max(params.splat_radius_dots, 0.5);
    let sigma = max(params.blur_sigma_dots, 0.5);

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

            let w = gaussian_weight(distance_dots, sigma) / weight_sum;
            let contribution = equivalent_dots * w;
            let atomic_units = u32(max(0.0, round(contribution * params.atomic_scale)));
            if atomic_units == 0u {
                continue;
            }
            let idx = dot_index(x, y);
            if p.phase_id == PHASE_WATER {
                atomicAdd(&density_atomic_water[idx], atomic_units);
            } else if particle_material_id(p) == MATERIAL_STONE_GRANULAR {
                atomicAdd(&density_atomic_stone[idx], atomic_units);
            } else if particle_material_id(p) == MATERIAL_SOIL_GRANULAR {
                atomicAdd(&density_atomic_soil[idx], atomic_units);
            } else if particle_material_id(p) == MATERIAL_SAND_GRANULAR {
                atomicAdd(&density_atomic_sand[idx], atomic_units);
            } else if particle_material_id(p) == MATERIAL_GRASS_GRANULAR {
                atomicAdd(&density_atomic_grass[idx], atomic_units);
            }
        }
    }
}

@compute @workgroup_size(64)
fn blur_x_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sx = clamp(x + k, 0, i32(params.width) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample =
            f32(atomicLoad(&density_atomic_water[dot_index(sx, y)])) / params.atomic_scale;
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blur_tmp[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_y_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sy = clamp(y + k, 0, i32(params.height) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample = blur_tmp[dot_index(x, sy)];
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blurred_density_water[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_x_stone(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sx = clamp(x + k, 0, i32(params.width) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample =
            f32(atomicLoad(&density_atomic_stone[dot_index(sx, y)])) / params.atomic_scale;
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blur_tmp[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_y_stone(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sy = clamp(y + k, 0, i32(params.height) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample = blur_tmp[dot_index(x, sy)];
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blurred_density_stone[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_x_soil(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sx = clamp(x + k, 0, i32(params.width) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample =
            f32(atomicLoad(&density_atomic_soil[dot_index(sx, y)])) / params.atomic_scale;
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blur_tmp[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_y_soil(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sy = clamp(y + k, 0, i32(params.height) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample = blur_tmp[dot_index(x, sy)];
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blurred_density_soil[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_x_sand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sx = clamp(x + k, 0, i32(params.width) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample =
            f32(atomicLoad(&density_atomic_sand[dot_index(sx, y)])) / params.atomic_scale;
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blur_tmp[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_y_sand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sy = clamp(y + k, 0, i32(params.height) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample = blur_tmp[dot_index(x, sy)];
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blurred_density_sand[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_x_grass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sx = clamp(x + k, 0, i32(params.width) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample =
            f32(atomicLoad(&density_atomic_grass[dot_index(sx, y)])) / params.atomic_scale;
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blur_tmp[idx] = density_sum / max(weight_sum, 1.0e-6);
}

@compute @workgroup_size(64)
fn blur_y_grass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }

    let x = i32(idx % params.width);
    let y = i32(idx / params.width);

    let radius = i32(params.blur_radius_dots);
    var density_sum = 0.0;
    var weight_sum = 0.0;

    for (var k = -radius; k <= radius; k = k + 1) {
        let sy = clamp(y + k, 0, i32(params.height) - 1);
        let w = gaussian_weight(f32(abs(k)), params.blur_sigma_dots);
        let sample = blur_tmp[dot_index(x, sy)];
        density_sum = density_sum + sample * w;
        weight_sum = weight_sum + w;
    }

    blurred_density_grass[idx] = density_sum / max(weight_sum, 1.0e-6);
}
