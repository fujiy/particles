/// TerrainNearUpdate compute pass.
///
/// Reads material IDs from `override_cells` (CPU-uploaded, one u32 per world cell)
/// and writes them into the `near_tex` StorageTexture (R16Uint).
///
/// Texture layout: one texel == one world cell. `texel (tx, ty)` corresponds to
/// world cell `(cache_origin_x + tx, cache_origin_y + ty)`.
///
/// Material encoding: 0 = Empty, 1 = Stone, 2 = Soil, 3 = Sand.

struct NearParams {
    override_origin_x: i32,
    override_origin_y: i32,
    override_width:    u32,
    override_height:   u32,
    cache_origin_x:    i32,
    cache_origin_y:    i32,
    _pad0:             u32,
    _pad1:             u32,
}

@group(0) @binding(0) var<uniform>  params:         NearParams;
@group(0) @binding(1) var<storage, read> override_cells: array<u32>;
@group(0) @binding(2) var           near_tex:       texture_storage_2d<r16uint, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(near_tex);
    if gid.x >= tex_size.x || gid.y >= tex_size.y {
        return;
    }

    // Local texture coordinate → world cell coordinate.
    let world_x = params.cache_origin_x + i32(gid.x);
    let world_y = params.cache_origin_y + i32(gid.y);

    // Look up in the override buffer (covers the same world extent as the texture).
    let ov_x = world_x - params.override_origin_x;
    let ov_y = world_y - params.override_origin_y;

    var material: u32 = 0u;
    if ov_x >= 0 && ov_y >= 0
       && u32(ov_x) < params.override_width
       && u32(ov_y) < params.override_height {
        let idx = u32(ov_y) * params.override_width + u32(ov_x);
        material = override_cells[idx];
    }

    textureStore(near_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<u32>(material, 0u, 0u, 0u));
}
