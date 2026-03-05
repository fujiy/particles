use std::collections::{HashMap, HashSet, hash_map::Entry};

use bevy::prelude::*;

pub use super::constants::{
    CELL_SIZE_M, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_WORLD_SIZE_M, DEFAULT_SOLID_HP,
    WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y,
};
pub use crate::physics::material::TerrainMaterial;
const SDF_INF: f32 = 1.0e9;
const SDF_QUERY_RADIUS_CELLS: i32 = 10;
pub const TERRAIN_GENERATOR_VERSION: u32 = 2;

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
    pub fn solid(material: TerrainMaterial) -> Self {
        Self::Solid {
            material,
            hp: DEFAULT_SOLID_HP,
        }
    }

    pub fn stone() -> Self {
        Self::solid(TerrainMaterial::Stone)
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

    pub fn is_pristine_with_base(
        &self,
        base_cells: &[TerrainCell; CHUNK_SIZE * CHUNK_SIZE],
    ) -> bool {
        self.cells
            .iter()
            .zip(base_cells.iter())
            .all(|(current, base)| current == base)
    }

    fn from_cells(cells: [TerrainCell; CHUNK_SIZE * CHUNK_SIZE]) -> Self {
        Self {
            cells,
            dirty: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrainOverrideChunk {
    cells: [Option<TerrainCell>; CHUNK_SIZE * CHUNK_SIZE],
    override_count: usize,
}

impl Default for TerrainOverrideChunk {
    fn default() -> Self {
        Self {
            cells: [None; CHUNK_SIZE * CHUNK_SIZE],
            override_count: 0,
        }
    }
}

impl TerrainOverrideChunk {
    pub fn get(&self, local_cell: IVec2) -> Option<TerrainCell> {
        self.cells[local_cell_to_index(local_cell)]
    }

    pub fn set(&mut self, local_cell: IVec2, next: Option<TerrainCell>) -> bool {
        let index = local_cell_to_index(local_cell);
        let prev = self.cells[index];
        if prev == next {
            return false;
        }
        self.cells[index] = next;
        if prev.is_none() && next.is_some() {
            self.override_count = self.override_count.saturating_add(1);
        } else if prev.is_some() && next.is_none() {
            self.override_count = self.override_count.saturating_sub(1);
        }
        true
    }

    pub fn is_empty(&self) -> bool {
        self.override_count == 0
    }

    pub fn iter_overrides(&self) -> impl Iterator<Item = (IVec2, TerrainCell)> + '_ {
        self.cells.iter().enumerate().filter_map(|(index, cell)| {
            let cell = (*cell)?;
            let local_x = (index % CHUNK_SIZE) as i32;
            let local_y = (index / CHUNK_SIZE) as i32;
            Some((IVec2::new(local_x, local_y), cell))
        })
    }
}

#[derive(Resource, Debug)]
pub struct TerrainWorld {
    generation_enabled: bool,
    terrain_version: u64,
    terrain_render_version: u64,
    chunks: HashMap<IVec2, TerrainChunk>,
    generated_base_chunks: HashMap<IVec2, [TerrainCell; CHUNK_SIZE * CHUNK_SIZE]>,
    chunk_overrides: HashMap<IVec2, TerrainOverrideChunk>,
    dirty_chunks: HashSet<IVec2>,
    dirty_override_chunks: HashSet<IVec2>,
    static_particle_pos: Vec<Vec2>,
    static_particle_grid: HashMap<IVec2, Vec<usize>>,
    static_particles_dirty: bool,
    sdf_samples: Vec<f32>,
    sdf_width: usize,
    sdf_height: usize,
    sdf_sample_spacing_m: f32,
    sdf_origin_m: Vec2,
    sdf_min_world_m: Vec2,
    sdf_max_world_m: Vec2,
}

impl Default for TerrainWorld {
    fn default() -> Self {
        Self {
            generation_enabled: true,
            terrain_version: 0,
            terrain_render_version: 0,
            chunks: HashMap::default(),
            generated_base_chunks: HashMap::default(),
            chunk_overrides: HashMap::default(),
            dirty_chunks: HashSet::default(),
            dirty_override_chunks: HashSet::default(),
            static_particle_pos: Vec::default(),
            static_particle_grid: HashMap::default(),
            static_particles_dirty: true,
            sdf_samples: Vec::default(),
            sdf_width: 0,
            sdf_height: 0,
            sdf_sample_spacing_m: 0.0,
            sdf_origin_m: Vec2::ZERO,
            sdf_min_world_m: Vec2::ZERO,
            sdf_max_world_m: Vec2::ZERO,
        }
    }
}

impl TerrainWorld {
    pub fn set_generation_enabled(&mut self, enabled: bool) {
        if self.generation_enabled != enabled {
            self.terrain_version = self.terrain_version.wrapping_add(1);
            self.terrain_render_version = self.terrain_render_version.wrapping_add(1);
        }
        self.generation_enabled = enabled;
    }

    pub fn generation_enabled(&self) -> bool {
        self.generation_enabled
    }

    pub fn terrain_version(&self) -> u64 {
        self.terrain_version
    }

    pub fn terrain_render_version(&self) -> u64 {
        self.terrain_render_version
    }

    pub fn clear(&mut self) {
        self.terrain_version = self.terrain_version.wrapping_add(1);
        self.terrain_render_version = self.terrain_render_version.wrapping_add(1);
        self.chunks.clear();
        self.generated_base_chunks.clear();
        self.chunk_overrides.clear();
        self.dirty_chunks.clear();
        self.dirty_override_chunks.clear();
        self.static_particle_pos.clear();
        self.static_particle_grid.clear();
        self.static_particles_dirty = true;
        self.sdf_samples.clear();
        self.sdf_width = 0;
        self.sdf_height = 0;
        self.sdf_sample_spacing_m = 0.0;
        self.sdf_origin_m = Vec2::ZERO;
        self.sdf_min_world_m = Vec2::ZERO;
        self.sdf_max_world_m = Vec2::ZERO;
    }

    pub fn reset_fixed_world(&mut self) {
        self.generation_enabled = true;
        self.clear();

        for chunk_y in WORLD_MIN_CHUNK_Y..=WORLD_MAX_CHUNK_Y {
            for chunk_x in WORLD_MIN_CHUNK_X..=WORLD_MAX_CHUNK_X {
                self.ensure_chunk_loaded(IVec2::new(chunk_x, chunk_y));
            }
        }
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
            self.terrain_version = self.terrain_version.wrapping_add(1);
            self.terrain_render_version = self.terrain_render_version.wrapping_add(1);
            self.update_chunk_override(global_cell, next);
            self.dirty_chunks.insert(chunk_coord);
            self.static_particles_dirty = true;
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

    pub fn load_generated_chunk_from_material_ids(
        &mut self,
        chunk_coord: IVec2,
        material_ids: &[u16; CHUNK_SIZE * CHUNK_SIZE],
    ) {
        if self.chunks.contains_key(&chunk_coord) {
            return;
        }
        let mut cells = [TerrainCell::Empty; CHUNK_SIZE * CHUNK_SIZE];
        for (index, material) in material_ids.iter().copied().enumerate() {
            cells[index] = material_id_to_terrain_cell(material);
        }

        self.generated_base_chunks.insert(chunk_coord, cells);
        let mut chunk = TerrainChunk::from_cells(cells);
        if let Some(overrides) = self.chunk_overrides.get(&chunk_coord) {
            for (local_cell, cell) in overrides.iter_overrides() {
                let _ = chunk.set(local_cell, cell);
            }
        }
        if chunk.is_dirty() {
            self.dirty_chunks.insert(chunk_coord);
        }
        self.chunks.insert(chunk_coord, chunk);
        self.terrain_version = self.terrain_version.wrapping_add(1);
        self.static_particles_dirty = true;
    }

    pub fn loaded_chunk_coords(&self) -> Vec<IVec2> {
        let mut coords: Vec<_> = self.chunks.keys().copied().collect();
        coords.sort_by_key(|coord| (coord.y, coord.x));
        coords
    }

    pub fn loaded_cell_bounds(&self) -> Option<(IVec2, IVec2)> {
        let mut min_chunk = IVec2::new(i32::MAX, i32::MAX);
        let mut max_chunk = IVec2::new(i32::MIN, i32::MIN);
        for &chunk in self.chunks.keys() {
            min_chunk.x = min_chunk.x.min(chunk.x);
            min_chunk.y = min_chunk.y.min(chunk.y);
            max_chunk.x = max_chunk.x.max(chunk.x);
            max_chunk.y = max_chunk.y.max(chunk.y);
        }
        if min_chunk.x > max_chunk.x || min_chunk.y > max_chunk.y {
            return None;
        }
        let min_cell = min_chunk * CHUNK_SIZE_I32;
        let max_cell = (max_chunk + IVec2::ONE) * CHUNK_SIZE_I32 - IVec2::ONE;
        Some((min_cell, max_cell))
    }

    pub fn clear_loaded_cells(&mut self) {
        if self.chunks.is_empty() {
            return;
        }
        let mut any_changed = false;
        let loaded_chunks: Vec<IVec2> = self.chunks.keys().copied().collect();
        for chunk_coord in loaded_chunks {
            let mut changed = false;
            let mut changed_cells = Vec::new();
            {
                let Some(chunk) = self.chunks.get_mut(&chunk_coord) else {
                    continue;
                };
                for local_y in 0..CHUNK_SIZE_I32 {
                    for local_x in 0..CHUNK_SIZE_I32 {
                        let local_cell = IVec2::new(local_x, local_y);
                        if chunk.set(local_cell, TerrainCell::Empty) {
                            changed_cells.push(chunk_coord * CHUNK_SIZE_I32 + local_cell);
                            changed = true;
                        }
                    }
                }
            }
            for global_cell in changed_cells {
                self.update_chunk_override(global_cell, TerrainCell::Empty);
            }
            if changed {
                any_changed = true;
                self.dirty_chunks.insert(chunk_coord);
            }
        }
        if any_changed {
            self.terrain_version = self.terrain_version.wrapping_add(1);
            self.terrain_render_version = self.terrain_render_version.wrapping_add(1);
        }
        self.static_particles_dirty = true;
    }

    #[allow(dead_code)]
    pub fn get_loaded_cell_or_empty(&self, global_cell: IVec2) -> TerrainCell {
        let (chunk_coord, local_cell) = global_to_chunk_local(global_cell);
        self.chunks
            .get(&chunk_coord)
            .map(|chunk| chunk.get(local_cell))
            .unwrap_or_default()
    }

    pub fn get_loaded_or_overridden_cell(&self, global_cell: IVec2) -> Option<TerrainCell> {
        let (chunk_coord, local_cell) = global_to_chunk_local(global_cell);
        if let Some(chunk) = self.chunks.get(&chunk_coord) {
            return Some(chunk.get(local_cell));
        }
        self.chunk_overrides
            .get(&chunk_coord)
            .and_then(|overrides| overrides.get(local_cell))
    }

    pub fn get_cell_or_generated(&self, global_cell: IVec2) -> TerrainCell {
        if let Some(cell) = self.get_loaded_or_overridden_cell(global_cell) {
            return cell;
        }
        TerrainCell::Empty
    }

    pub fn get_cell_or_generated_with_surface_y(
        &self,
        global_cell: IVec2,
        _surface_y: i32,
    ) -> TerrainCell {
        self.get_cell_or_generated(global_cell)
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

    pub fn dirty_chunk_coords(&self) -> Vec<IVec2> {
        let mut dirty: Vec<_> = self.dirty_chunks.iter().copied().collect();
        dirty.sort_by_key(|coord| (coord.y, coord.x));
        dirty
    }

    pub fn override_chunk_coords(&self) -> Vec<IVec2> {
        let mut coords: Vec<_> = self.chunk_overrides.keys().copied().collect();
        coords.sort_by_key(|coord| (coord.y, coord.x));
        coords
    }

    pub fn override_chunk_cells(&self, chunk_coord: IVec2) -> Option<&TerrainOverrideChunk> {
        self.chunk_overrides.get(&chunk_coord)
    }

    pub fn take_dirty_override_chunks(&mut self) -> Vec<IVec2> {
        let mut dirty: Vec<_> = self.dirty_override_chunks.drain().collect();
        dirty.sort_by_key(|coord| (coord.y, coord.x));
        dirty
    }

    pub fn dirty_override_chunk_coords(&self) -> Vec<IVec2> {
        let mut dirty: Vec<_> = self.dirty_override_chunks.iter().copied().collect();
        dirty.sort_by_key(|coord| (coord.y, coord.x));
        dirty
    }

    pub fn rebuild_static_particles_if_dirty(&mut self, grid_cell_size: f32) {
        if !self.static_particles_dirty {
            return;
        }

        self.static_particle_pos.clear();
        self.static_particle_grid.clear();

        for (chunk_coord, chunk) in &self.chunks {
            let base_cell = *chunk_coord * CHUNK_SIZE_I32;
            for local_y in 0..CHUNK_SIZE_I32 {
                for local_x in 0..CHUNK_SIZE_I32 {
                    let local_cell = IVec2::new(local_x, local_y);
                    let global_cell = base_cell + local_cell;
                    if !matches!(chunk.get(local_cell), TerrainCell::Solid { .. }) {
                        continue;
                    }
                    let pos = cell_to_world_center(global_cell);
                    let index = self.static_particle_pos.len();
                    self.static_particle_pos.push(pos);

                    let grid = IVec2::new(
                        (pos.x / grid_cell_size).floor() as i32,
                        (pos.y / grid_cell_size).floor() as i32,
                    );
                    self.static_particle_grid
                        .entry(grid)
                        .or_default()
                        .push(index);
                }
            }
        }

        self.sdf_samples.clear();
        self.sdf_width = 0;
        self.sdf_height = 0;
        self.sdf_sample_spacing_m = 0.0;
        self.sdf_origin_m = Vec2::ZERO;
        self.sdf_min_world_m = Vec2::ZERO;
        self.sdf_max_world_m = Vec2::ZERO;
        self.static_particles_dirty = false;
    }

    pub fn sample_signed_distance_and_normal(&self, world_pos: Vec2) -> Option<(f32, Vec2)> {
        let d = self.sample_signed_distance(world_pos);
        let eps = (0.5 * CELL_SIZE_M).max(1e-4);
        let dx = self.sample_signed_distance(world_pos + Vec2::new(eps, 0.0))
            - self.sample_signed_distance(world_pos - Vec2::new(eps, 0.0));
        let dy = self.sample_signed_distance(world_pos + Vec2::new(0.0, eps))
            - self.sample_signed_distance(world_pos - Vec2::new(0.0, eps));
        let normal = Vec2::new(dx, dy).normalize_or_zero();
        if normal == Vec2::ZERO {
            return Some((d, Vec2::Y));
        }
        Some((d, normal))
    }

    #[allow(dead_code)]
    pub fn gather_static_neighbors(
        &self,
        position: Vec2,
        grid_cell_size: f32,
        out_neighbors: &mut Vec<usize>,
    ) {
        out_neighbors.clear();
        let center = IVec2::new(
            (position.x / grid_cell_size).floor() as i32,
            (position.y / grid_cell_size).floor() as i32,
        );

        for y in (center.y - 1)..=(center.y + 1) {
            for x in (center.x - 1)..=(center.x + 1) {
                let Some(indices) = self.static_particle_grid.get(&IVec2::new(x, y)) else {
                    continue;
                };
                out_neighbors.extend(indices.iter().copied());
            }
        }
    }

    pub fn static_particle_positions(&self) -> &[Vec2] {
        &self.static_particle_pos
    }

    fn sample_signed_distance(&self, world_pos: Vec2) -> f32 {
        let center_cell = world_to_cell(world_pos);
        let inside = matches!(
            self.get_loaded_cell_or_empty(center_cell),
            TerrainCell::Solid { .. }
        );
        let mut best = SDF_INF;
        for dy in -SDF_QUERY_RADIUS_CELLS..=SDF_QUERY_RADIUS_CELLS {
            for dx in -SDF_QUERY_RADIUS_CELLS..=SDF_QUERY_RADIUS_CELLS {
                let cell = center_cell + IVec2::new(dx, dy);
                let cell_is_solid = matches!(
                    self.get_loaded_cell_or_empty(cell),
                    TerrainCell::Solid { .. }
                );
                if cell_is_solid == inside {
                    continue;
                }
                let min = cell.as_vec2() * CELL_SIZE_M;
                let max = min + Vec2::splat(CELL_SIZE_M);
                let closest = world_pos.clamp(min, max);
                best = best.min(world_pos.distance(closest));
            }
        }
        if best >= SDF_INF * 0.5 {
            best = (SDF_QUERY_RADIUS_CELLS as f32 + 1.0) * CELL_SIZE_M;
        }
        if inside { -best } else { best }
    }

    fn ensure_chunk_mut(&mut self, chunk_coord: IVec2) -> &mut TerrainChunk {
        match self.chunks.entry(chunk_coord) {
            Entry::Vacant(vacant) => {
                let mut chunk = TerrainChunk {
                    cells: [TerrainCell::Empty; CHUNK_SIZE * CHUNK_SIZE],
                    dirty: true,
                };
                if let Some(overrides) = self.chunk_overrides.get(&chunk_coord) {
                    for (local_cell, cell) in overrides.iter_overrides() {
                        let _ = chunk.set(local_cell, cell);
                    }
                }
                if chunk.is_dirty() {
                    self.dirty_chunks.insert(chunk_coord);
                }
                self.terrain_version = self.terrain_version.wrapping_add(1);
                self.static_particles_dirty = true;
                vacant.insert(chunk)
            }
            Entry::Occupied(occupied) => occupied.into_mut(),
        }
    }

    fn update_chunk_override(&mut self, global_cell: IVec2, next: TerrainCell) {
        let (chunk_coord, local_cell) = global_to_chunk_local(global_cell);
        let generated = if self.generation_enabled {
            self.generated_base_chunks
                .get(&chunk_coord)
                .map(|cells| cells[local_cell_to_index(local_cell)])
                .unwrap_or(TerrainCell::Empty)
        } else {
            TerrainCell::Empty
        };
        let override_cell = (next != generated).then_some(next);
        let changed = match self.chunk_overrides.entry(chunk_coord) {
            Entry::Vacant(vacant) => {
                let Some(cell) = override_cell else {
                    return;
                };
                let mut chunk = TerrainOverrideChunk::default();
                let _ = chunk.set(local_cell, Some(cell));
                vacant.insert(chunk);
                true
            }
            Entry::Occupied(mut occupied) => {
                let changed = occupied.get_mut().set(local_cell, override_cell);
                if occupied.get().is_empty() {
                    occupied.remove();
                }
                changed
            }
        };
        if changed {
            self.dirty_override_chunks.insert(chunk_coord);
        }
    }
}

fn material_id_to_terrain_cell(material_id: u16) -> TerrainCell {
    match material_id {
        1 => TerrainCell::solid(TerrainMaterial::Stone),
        2 => TerrainCell::solid(TerrainMaterial::Soil),
        3 => TerrainCell::solid(TerrainMaterial::Sand),
        _ => TerrainCell::Empty,
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
        assert_eq!(local, IVec2::new(CHUNK_SIZE_I32 - 1, CHUNK_SIZE_I32 - 1));
    }

    #[test]
    fn world_cell_round_trip_uses_cell_center() {
        let cell = IVec2::new(-3, 5);
        let world = cell_to_world_center(cell);
        let round_trip = world_to_cell(world);
        assert_eq!(round_trip, cell);
    }

    #[test]
    fn static_particles_marked_dirty_on_cell_changes() {
        let mut terrain = TerrainWorld::default();
        terrain.set_generation_enabled(false);
        terrain.ensure_chunk_loaded(IVec2::ZERO);
        assert!(terrain.set_cell(IVec2::new(0, 0), TerrainCell::stone()));
        terrain.rebuild_static_particles_if_dirty(0.25);
        assert!(!terrain.static_particles_dirty);

        let target = IVec2::new(0, 4);
        assert!(terrain.set_cell(target, TerrainCell::stone()));
        assert!(terrain.static_particles_dirty);

        terrain.rebuild_static_particles_if_dirty(0.25);
        assert!(!terrain.static_particles_dirty);
        assert!(!terrain.static_particle_positions().is_empty());
    }

    #[test]
    fn sdf_sign_matches_loaded_solid_and_empty_space() {
        let mut terrain = TerrainWorld::default();
        terrain.set_generation_enabled(false);
        terrain.ensure_chunk_loaded(IVec2::ZERO);
        assert!(terrain.set_cell(IVec2::new(0, 0), TerrainCell::stone()));
        terrain.rebuild_static_particles_if_dirty(0.25);
        let inside = cell_to_world_center(IVec2::new(0, 0));
        let above = cell_to_world_center(IVec2::new(0, 3));
        let d_inside = terrain
            .sample_signed_distance_and_normal(inside)
            .map(|(d, _)| d)
            .unwrap_or(f32::INFINITY);
        let d_above = terrain
            .sample_signed_distance_and_normal(above)
            .map(|(d, _)| d)
            .unwrap_or(f32::NEG_INFINITY);
        assert!(d_inside < 0.0);
        assert!(d_above > 0.0);
    }

    #[test]
    fn clear_loaded_cells_only_clears_existing_loaded_chunks() {
        let mut terrain = TerrainWorld::default();
        let chunk_a = IVec2::new(0, 0);
        let chunk_b = IVec2::new(3, 0);
        terrain.ensure_chunk_loaded(chunk_a);
        terrain.ensure_chunk_loaded(chunk_b);

        let before = terrain.loaded_chunk_coords();
        assert_eq!(before.len(), 2);

        terrain.clear_loaded_cells();

        let after = terrain.loaded_chunk_coords();
        assert_eq!(after.len(), 2);
        assert!(after.contains(&chunk_a));
        assert!(after.contains(&chunk_b));
        assert!(terrain.chunk(IVec2::new(1, 0)).is_none());
        assert!(terrain.chunk(IVec2::new(2, 0)).is_none());
    }

    #[test]
    fn get_cell_or_generated_uses_overrides_when_chunk_unloaded() {
        let mut terrain = TerrainWorld::default();
        let chunk = IVec2::new(9, -4);
        let cell = chunk * CHUNK_SIZE_I32 + IVec2::new(3, 5);
        let empty_ids = [0u16; CHUNK_SIZE * CHUNK_SIZE];
        terrain.load_generated_chunk_from_material_ids(chunk, &empty_ids);
        let modified = TerrainCell::stone();
        assert!(terrain.set_cell(cell, modified));
        assert_eq!(terrain.get_cell_or_generated(cell), TerrainCell::stone());

        terrain.chunks.remove(&chunk);
        assert_eq!(terrain.get_cell_or_generated(cell), TerrainCell::stone());

        terrain.load_generated_chunk_from_material_ids(chunk, &empty_ids);
        assert!(terrain.set_cell(cell, TerrainCell::Empty));
        terrain.chunks.remove(&chunk);
        assert_eq!(terrain.get_cell_or_generated(cell), TerrainCell::Empty);
    }

    #[test]
    fn get_cell_or_generated_with_surface_y_matches_default_path() {
        let mut terrain = TerrainWorld::default();
        terrain.set_generation_enabled(false);
        for x in -8..=8 {
            for y in -8..=8 {
                let cell = IVec2::new(x, y);
                assert_eq!(
                    terrain.get_cell_or_generated(cell),
                    terrain.get_cell_or_generated_with_surface_y(cell, 0),
                );
            }
        }
    }

    #[test]
    fn get_cell_or_generated_returns_empty_when_generation_disabled() {
        let mut terrain = TerrainWorld::default();
        terrain.set_generation_enabled(false);
        let cell = IVec2::new(0, -32);
        assert_eq!(terrain.get_cell_or_generated(cell), TerrainCell::Empty);
    }

    #[test]
    fn override_chunk_uses_fixed_array_and_dirty_tracking() {
        let mut terrain = TerrainWorld::default();
        let chunk = IVec2::new(7, -3);
        let cell = chunk * CHUNK_SIZE_I32 + IVec2::new(11, 9);
        let empty_ids = [0u16; CHUNK_SIZE * CHUNK_SIZE];
        terrain.load_generated_chunk_from_material_ids(chunk, &empty_ids);
        assert!(terrain.set_cell(cell, TerrainCell::stone()));

        let dirty_chunks = terrain.take_dirty_override_chunks();
        assert_eq!(dirty_chunks, vec![chunk]);

        let override_chunk = terrain.override_chunk_cells(chunk).unwrap();
        assert_eq!(
            override_chunk.get(IVec2::new(11, 9)),
            Some(TerrainCell::stone())
        );

        assert!(terrain.set_cell(cell, TerrainCell::Empty));
        let dirty_chunks = terrain.take_dirty_override_chunks();
        assert_eq!(dirty_chunks, vec![chunk]);
        assert!(terrain.override_chunk_cells(chunk).is_none());
    }
}
