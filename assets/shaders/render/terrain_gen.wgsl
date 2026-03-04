#define_import_path particles::terrain_gen

const TERRAIN_GENERATOR_VERSION: u32 = 1u;
const WORLD_SEED: u32 = 13370u;
const TERRAIN_SEED_MIX: u32 = 0x9e3779b9u;

const HEIGHT_NOISE_FREQ: f32 = 0.01;
const HEIGHT_NOISE_AMP_CELLS: f32 = 50.0;
const HEIGHT_NOISE_DETAIL_FREQ_MULT: f32 = 6.0;
const HEIGHT_NOISE_DETAIL_AMP_CELLS: f32 = 2.0;
const HEIGHT_NOISE_DETAIL_OCTAVES: u32 = 3u;
const BASE_SURFACE_Y: f32 = 0.0;
const SOIL_DEPTH_CELLS: i32 = 4;

const TERRAIN_FBM_OCTAVES: u32 = 4u;
const TERRAIN_FBM_LACUNARITY: f32 = 2.0;
const TERRAIN_FBM_GAIN: f32 = 0.5;
const FBM_ATTENUATION_SCALE: f32 = 2.2;

fn generator_seed() -> u32 {
    return WORLD_SEED ^ (TERRAIN_GENERATOR_VERSION * TERRAIN_SEED_MIX);
}

fn hash_u32(x: u32) -> u32 {
    var s = x;
    s ^= (s >> 16u);
    s *= 0x7feb352du;
    s ^= (s >> 15u);
    s *= 0x846ca68bu;
    s ^= (s >> 16u);
    return s;
}

fn hash_2d(x: i32, y: i32, seed: u32) -> u32 {
    let ux = bitcast<u32>(x);
    let uy = bitcast<u32>(y);
    return hash_u32(ux * 0x45d9f3bu ^ uy * 0x27d4eb2du ^ seed);
}

fn hash_to_signed_unit(h: u32) -> f32 {
    let unit01 = f32(h & 0x00ffffffu) / f32(0x01000000u);
    return unit01 * 2.0 - 1.0;
}

fn smoothstep01(t: f32) -> f32 {
    let c = clamp(t, 0.0, 1.0);
    return c * c * (3.0 - 2.0 * c);
}

fn value_noise_2d(p: vec2<f32>, seed: u32) -> f32 {
    let x0 = i32(floor(p.x));
    let y0 = i32(floor(p.y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let tx = smoothstep01(p.x - f32(x0));
    let ty = smoothstep01(p.y - f32(y0));

    let v00 = hash_to_signed_unit(hash_2d(x0, y0, seed));
    let v10 = hash_to_signed_unit(hash_2d(x1, y0, seed));
    let v01 = hash_to_signed_unit(hash_2d(x0, y1, seed));
    let v11 = hash_to_signed_unit(hash_2d(x1, y1, seed));

    let vx0 = mix(v00, v10, tx);
    let vx1 = mix(v01, v11, tx);
    return mix(vx0, vx1, ty);
}

fn octave_low_pass_weight(freq: f32, footprint_cells: f32) -> f32 {
    let footprint_freq = max(freq * footprint_cells, 0.0);
    let t = footprint_freq * FBM_ATTENUATION_SCALE;
    return 1.0 / (1.0 + t * t);
}

fn fbm_1d_filtered(
    x: f32,
    base_freq: f32,
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    footprint_cells: f32,
    channel_offset: f32,
) -> f32 {
    var sum = 0.0;
    var amp = 1.0;
    var freq = base_freq;
    var amp_sum = 0.0;
    let seed = generator_seed();

    for (var octave = 0u; octave < 16u; octave = octave + 1u) {
        if octave >= octaves {
            break;
        }
        let low_pass = octave_low_pass_weight(freq, footprint_cells);
        if low_pass < 1.0e-3 && octave > 1u {
            break;
        }
        let value = value_noise_2d(
            vec2<f32>(x * freq, channel_offset + f32(octave) * 13.37),
            seed,
        );
        let weighted_amp = amp * low_pass;
        sum = sum + value * weighted_amp;
        amp_sum = amp_sum + weighted_amp;
        amp = amp * gain;
        freq = freq * lacunarity;
    }

    if amp_sum <= 1.0e-6 {
        return 0.0;
    }
    return sum / amp_sum;
}

fn sample_surface_height_cells(x_cells: f32, footprint_cells: f32) -> f32 {
    let macro_n = fbm_1d_filtered(
        x_cells,
        HEIGHT_NOISE_FREQ,
        TERRAIN_FBM_OCTAVES,
        TERRAIN_FBM_LACUNARITY,
        TERRAIN_FBM_GAIN,
        footprint_cells,
        f32(TERRAIN_GENERATOR_VERSION),
    );
    let detail_n = fbm_1d_filtered(
        x_cells,
        HEIGHT_NOISE_FREQ * HEIGHT_NOISE_DETAIL_FREQ_MULT,
        HEIGHT_NOISE_DETAIL_OCTAVES,
        TERRAIN_FBM_LACUNARITY,
        0.5,
        footprint_cells,
        f32(TERRAIN_GENERATOR_VERSION) + 101.0,
    );
    return BASE_SURFACE_Y
        + macro_n * HEIGHT_NOISE_AMP_CELLS
        + detail_n * HEIGHT_NOISE_DETAIL_AMP_CELLS;
}

fn surface_height_for_x(world_x: i32) -> i32 {
    return i32(round(sample_surface_height_cells(f32(world_x), 0.0)));
}

fn material_for_cell(global_cell: vec2<i32>, generation_enabled: bool) -> u32 {
    if !generation_enabled {
        return 0u;
    }
    let surface_y = surface_height_for_x(global_cell.x);
    if global_cell.y > surface_y {
        return 0u;
    }
    if global_cell.y > (surface_y - SOIL_DEPTH_CELLS) {
        return 2u;
    }
    return 1u;
}
