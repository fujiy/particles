use bevy::prelude::*;

use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, world_to_cell,
};

pub const RATE_LEVEL_MIN: u8 = 0;
pub const RATE_LEVEL_MAX_DEFAULT: u8 = 8;
pub const RATE_DIVISOR_MIN: u16 = 1;
// `rate_divisor` is stored as `u16`, so `2^level` is representable up to level 15.
// Inputs above this are clamped to keep scheduling math valid and predictable.
pub const RATE_LEVEL_MAX_SUPPORTED: u8 = (u16::BITS - 1) as u8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubBlockCoord(pub IVec2);

#[derive(Clone, Copy, Debug)]
pub struct SubBlockOverlaySample {
    pub coord: IVec2,
    pub rate_divisor: u16,
    pub debt_ratio: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SubBlockFrameCounters {
    pub promote_counter: u16,
    pub demote_counter: u16,
    pub active_hold_frames: u16,
    pub contact_frames: u16,
    pub idle_frames: u16,
}

impl Default for SubBlockFrameCounters {
    fn default() -> Self {
        Self {
            promote_counter: 0,
            demote_counter: 0,
            active_hold_frames: 0,
            contact_frames: 0,
            idle_frames: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SubBlockState {
    pub rate_divisor: u16,
    pub scheduled_this_substep: bool,
    pub boundary_debt_impulse: Vec2,
    pub boundary_debt_peak: f32,
    pub latest_mass_flux: f32,
    pub latest_penetration_m: f32,
    pub latest_intrusion: bool,
    pub counters: SubBlockFrameCounters,
}

impl Default for SubBlockState {
    fn default() -> Self {
        Self {
            rate_divisor: RATE_DIVISOR_MIN,
            scheduled_this_substep: true,
            boundary_debt_impulse: Vec2::ZERO,
            boundary_debt_peak: 0.0,
            latest_mass_flux: 0.0,
            latest_penetration_m: 0.0,
            latest_intrusion: false,
            counters: SubBlockFrameCounters::default(),
        }
    }
}

pub fn clamp_rate_level(level: u8, max_level: u8) -> u8 {
    level.min(max_level.min(RATE_LEVEL_MAX_SUPPORTED))
}

pub fn rate_divisor_from_level(level: u8) -> u16 {
    let clamped = clamp_rate_level(level, RATE_LEVEL_MAX_SUPPORTED);
    RATE_DIVISOR_MIN << clamped
}

pub fn normalize_rate_class_with_max(rate_divisor: u16, max_level: u8) -> u16 {
    let max_divisor = rate_divisor_from_level(max_level);
    if rate_divisor <= RATE_DIVISOR_MIN {
        return RATE_DIVISOR_MIN;
    }
    let mut normalized = RATE_DIVISOR_MIN;
    while normalized < rate_divisor && normalized < max_divisor {
        normalized <<= 1;
    }
    normalized.min(max_divisor).max(RATE_DIVISOR_MIN)
}

pub fn normalize_rate_class(rate_divisor: u16) -> u16 {
    normalize_rate_class_with_max(rate_divisor, RATE_LEVEL_MAX_DEFAULT)
}

pub fn rate_level_from_divisor(rate_divisor: u16, max_level: u8) -> u8 {
    let mut divisor = normalize_rate_class_with_max(rate_divisor, max_level);
    let mut level = RATE_LEVEL_MIN;
    while divisor > RATE_DIVISOR_MIN {
        divisor >>= 1;
        level = level.saturating_add(1);
    }
    clamp_rate_level(level, max_level)
}

pub fn world_pos_to_sub_block(position: Vec2, sub_block_size_cells: i32) -> IVec2 {
    let size = sub_block_size_cells.max(1);
    let cell = world_to_cell(position);
    IVec2::new(cell.x.div_euclid(size), cell.y.div_euclid(size))
}

pub fn sub_block_world_min(coord: IVec2, sub_block_size_cells: i32) -> Vec2 {
    coord.as_vec2() * sub_block_world_size_m(sub_block_size_cells)
}

pub fn sub_block_world_size_m(sub_block_size_cells: i32) -> f32 {
    sub_block_size_cells.max(1) as f32 * CELL_SIZE_M
}

pub fn sub_block_world_bounds(coord: IVec2, sub_block_size_cells: i32) -> (Vec2, Vec2) {
    let min = sub_block_world_min(coord, sub_block_size_cells);
    let max = min + Vec2::splat(sub_block_world_size_m(sub_block_size_cells));
    (min, max)
}

pub fn sub_block_to_chunk(coord: IVec2, sub_block_size_cells: i32) -> IVec2 {
    let size = sub_block_size_cells.max(1);
    let origin_cell = coord * size;
    IVec2::new(
        origin_cell.x.div_euclid(CHUNK_SIZE_I32),
        origin_cell.y.div_euclid(CHUNK_SIZE_I32),
    )
}

pub fn chunk_to_sub_block_span(sub_block_size_cells: i32) -> i32 {
    let size = sub_block_size_cells.max(1);
    (CHUNK_SIZE_I32 / size).max(1)
}

pub fn chunk_world_span_m(span_chunks: i32) -> f32 {
    span_chunks.max(1) as f32 * CHUNK_WORLD_SIZE_M
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_class_normalization_keeps_power_of_two_up_to_max_level() {
        assert_eq!(normalize_rate_class_with_max(1, 8), 1);
        assert_eq!(normalize_rate_class_with_max(2, 8), 2);
        assert_eq!(normalize_rate_class_with_max(8, 8), 8);
        assert_eq!(normalize_rate_class_with_max(3, 8), 4);
        assert_eq!(normalize_rate_class_with_max(9, 8), 16);
        assert_eq!(normalize_rate_class_with_max(250, 8), 256);
    }

    #[test]
    fn rate_level_conversion_round_trips() {
        for level in 0..=8 {
            let divisor = rate_divisor_from_level(level);
            assert_eq!(rate_level_from_divisor(divisor, 8), level);
        }
    }

    #[test]
    fn sub_block_coord_conversion_handles_negative_cells() {
        let size = 16;
        let inside = world_pos_to_sub_block(Vec2::new(-0.1, -0.1), size);
        let lower = world_pos_to_sub_block(Vec2::new(-4.2, -4.2), size);
        assert_eq!(inside, IVec2::new(-1, -1));
        assert_eq!(lower, IVec2::new(-2, -2));
    }

    #[test]
    fn sub_block_maps_back_to_expected_chunk() {
        let size = 16;
        assert_eq!(sub_block_to_chunk(IVec2::new(0, 0), size), IVec2::ZERO);
        assert_eq!(sub_block_to_chunk(IVec2::new(1, 1), size), IVec2::new(1, 1));
        assert_eq!(sub_block_to_chunk(IVec2::new(2, 0), size), IVec2::new(2, 0));
        assert_eq!(sub_block_to_chunk(IVec2::new(-1, -1), size), IVec2::new(-1, -1));
    }
}
