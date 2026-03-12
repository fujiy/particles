#import bevy_render::view::View

@group(0) @binding(0) var back_tex: texture_2d<f32>;
@group(0) @binding(1) var main_tex: texture_2d<f32>;
@group(0) @binding(2) var nearest_sampler: sampler;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

fn quad_corner(v: u32) -> vec2<f32> {
    if v == 0u { return vec2<f32>(-1.0, -1.0); }
    if v == 1u { return vec2<f32>( 1.0, -1.0); }
    if v == 2u { return vec2<f32>( 1.0,  1.0); }
    if v == 3u { return vec2<f32>(-1.0, -1.0); }
    if v == 4u { return vec2<f32>( 1.0,  1.0); }
    return vec2<f32>(-1.0,  1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let corner = quad_corner(vertex_index);
    var out: VertexOut;
    out.clip_position = vec4<f32>(corner, 0.0, 1.0);
    out.uv = corner * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

fn layer_over(bg: vec4<f32>, fg: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(fg.a, 0.0, 1.0);
    return vec4<f32>(bg.rgb * (1.0 - alpha) + fg.rgb * alpha, 1.0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let back = textureSampleLevel(back_tex, nearest_sampler, in.uv, 0.0);
    let main = textureSampleLevel(main_tex, nearest_sampler, in.uv, 0.0);
    return layer_over(back, main);
}
