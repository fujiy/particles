use std::collections::{HashMap, HashSet, hash_map::Entry};

use bevy::prelude::*;

pub const CELL_SIZE_M: f32 = 0.25;
pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CELL_PIXEL_SIZE: u32 = 4;
pub const CHUNK_PIXEL_SIZE: u32 = (CHUNK_SIZE as u32) * CELL_PIXEL_SIZE;
pub const CHUNK_WORLD_SIZE_M: f32 = (CHUNK_SIZE as f32) * CELL_SIZE_M;
pub const DEFAULT_SOLID_HP: u16 = 1_000;
pub const WORLD_MIN_CHUNK_X: i32 = -2;
pub const WORLD_MAX_CHUNK_X: i32 = 1;
pub const WORLD_MIN_CHUNK_Y: i32 = -2;
pub const WORLD_MAX_CHUNK_Y: i32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainMaterial {
    Rock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TerrainCell {
    #[default]
    Empty,
    Solid {
        material: TerrainMaterial,
        hp: u16,
    },
}

impl TerrainCell {
    pub fn rock() -> Self {
        Self::Solid {
            material: TerrainMaterial::Rock,
            hp: DEFAULT_SOLID_HP,
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, TerrainCell::Empty)
    }
}

#[derive(Debug, Clone)]
pub struct TerrainChunk {
    cells: [TerrainCell; CHUNK_SIZE * CHUNK_SIZE],
    dirty: bool,
}

impl TerrainChunk {
    pub fn generated(chunk_coord: IVec2) -> Self {
        let mut chunk = Self {
            cells: [TerrainCell::Empty; CHUNK_SIZE * CHUNK_SIZE],
            dirty: true,
        };

        for local_y in 0..CHUNK_SIZE_I32 {
            let global_y = chunk_coord.y * CHUNK_SIZE_I32 + local_y;
            if global_y > 0 {
                continue;
            }
            for local_x in 0..CHUNK_SIZE_I32 {
                let local = IVec2::new(local_x, local_y);
                let index = local_cell_to_index(local);
                chunk.cells[index] = TerrainCell::rock();
            }
        }

        chunk
    }

    pub fn get(&self, local_cell: IVec2) -> TerrainCell {
        self.cells[local_cell_to_index(local_cell)]
    }

    pub fn set(&mut self, local_cell: IVec2, next: TerrainCell) -> bool {
        let index = local_cell_to_index(local_cell);
        let prev = self.cells[index];
        if prev == next {
            return false;
        }
        self.cells[index] = next;
        self.dirty = true;
        true
    }

    pub fn is_empty(&self) -> bool {
        self.cells.iter().all(TerrainCell::is_empty)
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

#[derive(Resource, Debug, Default)]
pub struct TerrainWorld {
    chunks: HashMap<IVec2, TerrainChunk>,
    dirty_chunks: HashSet<IVec2>,
}

impl TerrainWorld {
    pub fn reset_fixed_world(&mut self) {
        self.chunks.clear();
        self.dirty_chunks.clear();

        for chunk_y in WORLD_MIN_CHUNK_Y..=WORLD_MAX_CHUNK_Y {
            for chunk_x in WORLD_MIN_CHUNK_X..=WORLD_MAX_CHUNK_X {
                self.ensure_chunk_loaded(IVec2::new(chunk_x, chunk_y));
            }
        }

        let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32;
        let max_cell_x = (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1;
        let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32;
        let max_cell_y = (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1;

        self.fill_rect(
            IVec2::new(min_cell_x, min_cell_y),
            IVec2::new(min_cell_x, max_cell_y),
            TerrainCell::rock(),
        );
        self.fill_rect(
            IVec2::new(max_cell_x, min_cell_y),
            IVec2::new(max_cell_x, max_cell_y),
            TerrainCell::rock(),
        );
    }

    pub fn ensure_chunk_loaded(&mut self, chunk_coord: IVec2) {
        self.ensure_chunk_mut(chunk_coord);
    }

    #[allow(dead_code)]
    pub fn get_cell(&mut self, global_cell: IVec2) -> TerrainCell {
        let (chunk_coord, local_cell) = global_to_chunk_local(global_cell);
        self.ensure_chunk_mut(chunk_coord).get(local_cell)
    }

    pub fn set_cell(&mut self, global_cell: IVec2, next: TerrainCell) -> bool {
        let (chunk_coord, local_cell) = global_to_chunk_local(global_cell);
        let changed = self.ensure_chunk_mut(chunk_coord).set(local_cell, next);
        if changed {
            self.dirty_chunks.insert(chunk_coord);
        }
        changed
    }

    pub fn fill_rect(&mut self, min_cell: IVec2, max_cell: IVec2, value: TerrainCell) {
        let min_x = min_cell.x.min(max_cell.x);
        let max_x = min_cell.x.max(max_cell.x);
        let min_y = min_cell.y.min(max_cell.y);
        let max_y = min_cell.y.max(max_cell.y);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                self.set_cell(IVec2::new(x, y), value);
            }
        }
    }

    #[allow(dead_code)]
    pub fn for_each_in_chunk<F>(&mut self, chunk_coord: IVec2, mut f: F)
    where
        F: FnMut(IVec2, TerrainCell),
    {
        self.ensure_chunk_loaded(chunk_coord);

        let Some(chunk) = self.chunks.get(&chunk_coord) else {
            return;
        };

        let base_cell = chunk_coord * CHUNK_SIZE_I32;
        for local_y in 0..CHUNK_SIZE_I32 {
            for local_x in 0..CHUNK_SIZE_I32 {
                let local_cell = IVec2::new(local_x, local_y);
                let global_cell = base_cell + local_cell;
                f(global_cell, chunk.get(local_cell));
            }
        }
    }

    pub fn chunk(&self, chunk_coord: IVec2) -> Option<&TerrainChunk> {
        self.chunks.get(&chunk_coord)
    }

    pub fn get_loaded_cell_or_empty(&self, global_cell: IVec2) -> TerrainCell {
        let (chunk_coord, local_cell) = global_to_chunk_local(global_cell);
        self.chunks
            .get(&chunk_coord)
            .map(|chunk| chunk.get(local_cell))
            .unwrap_or_default()
    }

    pub fn take_dirty_chunks(&mut self) -> Vec<IVec2> {
        let mut dirty: Vec<_> = self.dirty_chunks.drain().collect();
        dirty.sort_by_key(|coord| (coord.y, coord.x));

        for chunk_coord in &dirty {
            if let Some(chunk) = self.chunks.get_mut(chunk_coord) {
                chunk.clear_dirty();
            }
        }

        dirty
    }

    fn ensure_chunk_mut(&mut self, chunk_coord: IVec2) -> &mut TerrainChunk {
        match self.chunks.entry(chunk_coord) {
            Entry::Vacant(vacant) => {
                let chunk = TerrainChunk::generated(chunk_coord);
                if chunk.is_dirty() {
                    self.dirty_chunks.insert(chunk_coord);
                }
                vacant.insert(chunk)
            }
            Entry::Occupied(occupied) => occupied.into_mut(),
        }
    }
}

#[allow(dead_code)]
pub fn world_to_cell(world_position: Vec2) -> IVec2 {
    IVec2::new(
        (world_position.x / CELL_SIZE_M).floor() as i32,
        (world_position.y / CELL_SIZE_M).floor() as i32,
    )
}

pub fn cell_to_world_center(global_cell: IVec2) -> Vec2 {
    (global_cell.as_vec2() + Vec2::splat(0.5)) * CELL_SIZE_M
}

pub fn global_to_chunk_local(global_cell: IVec2) -> (IVec2, IVec2) {
    let chunk_coord = IVec2::new(
        global_cell.x.div_euclid(CHUNK_SIZE_I32),
        global_cell.y.div_euclid(CHUNK_SIZE_I32),
    );
    let local_cell = IVec2::new(
        global_cell.x.rem_euclid(CHUNK_SIZE_I32),
        global_cell.y.rem_euclid(CHUNK_SIZE_I32),
    );
    (chunk_coord, local_cell)
}

fn local_cell_to_index(local_cell: IVec2) -> usize {
    debug_assert!(local_cell.x >= 0 && local_cell.x < CHUNK_SIZE_I32);
    debug_assert!(local_cell.y >= 0 && local_cell.y < CHUNK_SIZE_I32);
    (local_cell.y as usize) * CHUNK_SIZE + (local_cell.x as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_to_chunk_local_handles_negative_cells() {
        let (chunk, local) = global_to_chunk_local(IVec2::new(-1, -1));
        assert_eq!(chunk, IVec2::new(-1, -1));
        assert_eq!(local, IVec2::new(31, 31));
    }

    #[test]
    fn generated_chunk_fills_cells_below_surface() {
        let chunk_above = TerrainChunk::generated(IVec2::new(0, 1));
        assert!(chunk_above.is_empty());

        let chunk_surface = TerrainChunk::generated(IVec2::new(0, 0));
        assert!(matches!(
            chunk_surface.get(IVec2::new(0, 0)),
            TerrainCell::Solid { .. }
        ));

        let chunk_below = TerrainChunk::generated(IVec2::new(0, -1));
        assert!(matches!(
            chunk_below.get(IVec2::new(0, 31)),
            TerrainCell::Solid { .. }
        ));
    }

    #[test]
    fn world_cell_round_trip_uses_cell_center() {
        let cell = IVec2::new(-3, 5);
        let world = cell_to_world_center(cell);
        let round_trip = world_to_cell(world);
        assert_eq!(round_trip, cell);
    }
}
