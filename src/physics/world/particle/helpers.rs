use bevy::prelude::{IVec2, Vec2};

use super::{
    ChunkEdge, FAR_FIELD_EDGE_INSET_M, GRANULAR_GRANULAR_CONTACT_FRICTION_BOOST,
    GRANULAR_SOLID_CONTACT_FRICTION_BOOST,
};
use crate::physics::material::{
    DEFAULT_MATERIAL_PARAMS, ParticleMaterial, particle_properties, water_kernel_radius_m,
};
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, TerrainCell, TerrainWorld, cell_to_world_center, world_to_cell,
};

pub(super) fn particle_grid_axis(particles_per_cell: u32) -> u32 {
    if particles_per_cell <= 1 {
        return 1;
    }
    let mut axis = 1u32;
    while axis.saturating_mul(axis) < particles_per_cell {
        axis += 1;
    }
    if axis.saturating_mul(axis) == particles_per_cell {
        axis
    } else {
        1
    }
}

pub(super) fn resolve_terrain_contact_cell(
    position: Vec2,
    signed_distance: f32,
    normal: Vec2,
    terrain: &TerrainWorld,
) -> Option<IVec2> {
    let boundary_point = position - normal * signed_distance.max(0.0);
    let probe = boundary_point - normal * (CELL_SIZE_M * 0.25);
    let center_cell = world_to_cell(probe);

    let mut best: Option<(f32, IVec2)> = None;
    for y in (center_cell.y - 1)..=(center_cell.y + 1) {
        for x in (center_cell.x - 1)..=(center_cell.x + 1) {
            let cell = IVec2::new(x, y);
            if !matches!(
                terrain.get_loaded_cell_or_empty(cell),
                TerrainCell::Solid { .. }
            ) {
                continue;
            }
            let d2 = distance_sq_to_cell_aabb(boundary_point, cell);
            let replace = match best {
                Some((best_d2, _)) => d2 < best_d2,
                None => true,
            };
            if replace {
                best = Some((d2, cell));
            }
        }
    }
    best.map(|(_, cell)| cell)
}

pub(super) fn distance_sq_to_cell_aabb(point: Vec2, cell: IVec2) -> f32 {
    let center = cell_to_world_center(cell);
    let half = CELL_SIZE_M * 0.5;
    let min = center - Vec2::splat(half);
    let max = center + Vec2::splat(half);
    let closest = Vec2::new(point.x.clamp(min.x, max.x), point.y.clamp(min.y, max.y));
    point.distance_squared(closest)
}

pub(super) fn default_particle_mass() -> f32 {
    particle_properties(ParticleMaterial::WaterLiquid).mass
}

pub(super) fn particle_grid_cell(position: Vec2) -> IVec2 {
    let kernel_radius = water_kernel_radius_m(DEFAULT_MATERIAL_PARAMS);
    IVec2::new(
        (position.x / kernel_radius).floor() as i32,
        (position.y / kernel_radius).floor() as i32,
    )
}

pub(super) fn kernel_poly6(r2: f32, support_radius: f32) -> f32 {
    let h2 = support_radius * support_radius;
    if !(0.0..h2).contains(&r2) {
        return 0.0;
    }
    let x = h2 - r2;
    let coeff = 4.0 / (std::f32::consts::PI * h2.powi(4));
    coeff * x.powi(3)
}

pub(super) fn kernel_spiky_grad(r: Vec2, support_radius: f32) -> Vec2 {
    let len = r.length();
    if !(1e-6..support_radius).contains(&len) {
        return Vec2::ZERO;
    }
    let coeff = -30.0 / (std::f32::consts::PI * support_radius.powi(5));
    r * (coeff * (support_radius - len).powi(2) / len)
}

pub(crate) fn is_water_particle(material: ParticleMaterial) -> bool {
    matches!(material, ParticleMaterial::WaterLiquid)
}

pub(crate) fn is_granular_particle(material: ParticleMaterial) -> bool {
    matches!(
        material,
        ParticleMaterial::StoneGranular
            | ParticleMaterial::SoilGranular
            | ParticleMaterial::SandGranular
    )
}

pub(super) fn is_solid_particle(material: ParticleMaterial) -> bool {
    matches!(
        material,
        ParticleMaterial::StoneSolid | ParticleMaterial::SoilSolid | ParticleMaterial::SandSolid
    )
}

pub(crate) fn granular_contact_friction_pair_scale(
    material_a: ParticleMaterial,
    material_b: ParticleMaterial,
) -> Option<f32> {
    if is_water_particle(material_a) || is_water_particle(material_b) {
        return None;
    }
    if !(is_granular_particle(material_a) || is_granular_particle(material_b)) {
        return None;
    }
    let pair_scale = if is_granular_particle(material_a) && is_granular_particle(material_b) {
        GRANULAR_GRANULAR_CONTACT_FRICTION_BOOST
    } else if is_solid_particle(material_a) || is_solid_particle(material_b) {
        GRANULAR_SOLID_CONTACT_FRICTION_BOOST
    } else {
        1.0
    };
    Some(pair_scale)
}

pub(super) fn granular_tangential_friction_push(
    contact_normal: Vec2,
    normal_push: Vec2,
    relative_velocity: Vec2,
    static_friction: f32,
    dynamic_friction: f32,
    dt_sub: f32,
) -> Vec2 {
    let normal_push_mag = normal_push.length();
    if normal_push_mag <= 1e-8 {
        return Vec2::ZERO;
    }
    let tangential_velocity =
        relative_velocity - contact_normal * relative_velocity.dot(contact_normal);
    let tangential_speed = tangential_velocity.length();
    if tangential_speed <= 1e-6 {
        return Vec2::ZERO;
    }

    let tangential_displacement = tangential_velocity * dt_sub.max(0.0);
    let tangential_displacement_mag = tangential_displacement.length();
    if tangential_displacement_mag <= 1e-8 {
        return Vec2::ZERO;
    }

    let static_limit = static_friction.max(0.0) * normal_push_mag;
    if tangential_displacement_mag <= static_limit {
        return -tangential_displacement;
    }

    let dynamic_limit = dynamic_friction.max(0.0) * normal_push_mag;
    if dynamic_limit <= 1e-8 {
        return Vec2::ZERO;
    }
    let friction_mag = dynamic_limit.min(tangential_displacement_mag);
    -tangential_displacement.normalize_or_zero() * friction_mag
}

pub(super) fn spawn_jitter_offset(
    cell: IVec2,
    material: ParticleMaterial,
    slot_x: u32,
    slot_y: u32,
    max_offset: f32,
) -> Vec2 {
    if max_offset <= 0.0 {
        return Vec2::ZERO;
    }
    let base = mix_u32(
        (cell.x as u32).wrapping_mul(0x45d9_f3b)
            ^ (cell.y as u32).wrapping_mul(0x27d4_eb2d)
            ^ material_seed(material)
            ^ slot_x.wrapping_mul(0x1656_67b1)
            ^ slot_y.wrapping_mul(0x9e37_79b9),
    );
    let hx = mix_u32(base ^ 0x68bc_21eb);
    let hy = mix_u32(base ^ 0x02e5_be93);
    let fx = (hx as f32 / u32::MAX as f32) * 2.0 - 1.0;
    let fy = (hy as f32 / u32::MAX as f32) * 2.0 - 1.0;
    Vec2::new(fx, fy) * max_offset
}

pub(super) fn material_seed(material: ParticleMaterial) -> u32 {
    match material {
        ParticleMaterial::WaterLiquid => 0x1111_1111,
        ParticleMaterial::StoneSolid => 0x2222_2222,
        ParticleMaterial::StoneGranular => 0x3333_3333,
        ParticleMaterial::SoilSolid => 0x4444_4444,
        ParticleMaterial::SoilGranular => 0x5555_5555,
        ParticleMaterial::SandSolid => 0x6666_6666,
        ParticleMaterial::SandGranular => 0x7777_7777,
    }
}

pub(super) fn mix_u32(mut state: u32) -> u32 {
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    state
}

pub(super) fn terrain_ghost_neighbor_vector(
    position: Vec2,
    terrain: &TerrainWorld,
    support_radius: f32,
) -> Option<Vec2> {
    let (signed_distance, normal) = terrain.sample_signed_distance_and_normal(position)?;
    let d = signed_distance.max(0.0);
    if d >= support_radius {
        return None;
    }
    Some(normal * (2.0 * d))
}

pub(super) fn world_pos_to_chunk(position: Vec2) -> IVec2 {
    let cell = world_to_cell(position);
    IVec2::new(
        cell.x.div_euclid(CHUNK_SIZE_I32),
        cell.y.div_euclid(CHUNK_SIZE_I32),
    )
}

pub(super) fn is_chunk_outside_radius(chunk: IVec2, center_chunk: IVec2, radius: i32) -> bool {
    let radius = radius.max(0);
    (chunk.x - center_chunk.x).abs() > radius || (chunk.y - center_chunk.y).abs() > radius
}

pub(super) fn edge_sort_key(edge: ChunkEdge) -> i32 {
    match edge {
        ChunkEdge::North => 0,
        ChunkEdge::East => 1,
        ChunkEdge::South => 2,
        ChunkEdge::West => 3,
    }
}

pub(super) fn edge_sample_for_active_boundary(
    chunk: IVec2,
    position: Vec2,
    center_chunk: IVec2,
    active_radius: i32,
) -> (ChunkEdge, Vec2, i32) {
    let active_radius = active_radius.max(0);
    let chunk_world_size = CHUNK_SIZE_I32 as f32 * CELL_SIZE_M;
    let boundary_min_chunk = center_chunk - IVec2::splat(active_radius);
    let boundary_max_chunk = center_chunk + IVec2::splat(active_radius);

    let side_min_x = boundary_min_chunk.x as f32 * chunk_world_size;
    let side_max_x = (boundary_max_chunk.x + 1) as f32 * chunk_world_size;
    let side_min_y = boundary_min_chunk.y as f32 * chunk_world_size;
    let side_max_y = (boundary_max_chunk.y + 1) as f32 * chunk_world_size;

    let inset = FAR_FIELD_EDGE_INSET_M.max(0.0);
    let min_x = side_min_x + inset;
    let max_x = side_max_x - inset;
    let min_y = side_min_y + inset;
    let max_y = side_max_y - inset;

    let dx = chunk.x - center_chunk.x;
    let dy = chunk.y - center_chunk.y;
    let (edge, sample_pos) = if dx.abs() >= dy.abs() {
        if dx >= 0 {
            let boundary_x = side_max_x + inset;
            (
                ChunkEdge::West,
                Vec2::new(boundary_x, position.y.clamp(min_y, max_y)),
            )
        } else {
            let boundary_x = side_min_x - inset;
            (
                ChunkEdge::East,
                Vec2::new(boundary_x, position.y.clamp(min_y, max_y)),
            )
        }
    } else if dy >= 0 {
        let boundary_y = side_max_y + inset;
        (
            ChunkEdge::South,
            Vec2::new(position.x.clamp(min_x, max_x), boundary_y),
        )
    } else {
        let boundary_y = side_min_y - inset;
        (
            ChunkEdge::North,
            Vec2::new(position.x.clamp(min_x, max_x), boundary_y),
        )
    };
    let grid_cell = particle_grid_cell(sample_pos);
    let grid_segment = match edge {
        ChunkEdge::North | ChunkEdge::South => grid_cell.x,
        ChunkEdge::East | ChunkEdge::West => grid_cell.y,
    };
    (edge, sample_pos, grid_segment)
}
