use std::collections::{HashMap, HashSet, hash_map::Entry};

use bevy::prelude::*;
use super::generation;

pub use super::material::TerrainMaterial;
pub use super::generation::{
    BASE_SURFACE_Y, HEIGHT_NOISE_AMP_CELLS, HEIGHT_NOISE_FREQ, SOIL_DEPTH_CELLS,
    TERRAIN_GENERATOR_VERSION, WORLD_SEED,
};

pub const CELL_SIZE_M: f32 = 0.25;
pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CELL_PIXEL_SIZE: u32 = 8;
pub const CHUNK_PIXEL_SIZE: u32 = (CHUNK_SIZE as u32) * CELL_PIXEL_SIZE;
pub const CHUNK_WORLD_SIZE_M: f32 = (CHUNK_SIZE as f32) * CELL_SIZE_M;
pub const DEFAULT_SOLID_HP: u16 = 1_000;
pub const WORLD_MIN_CHUNK_X: i32 = -2;
pub const WORLD_MAX_CHUNK_X: i32 = 1;
pub const WORLD_MIN_CHUNK_Y: i32 = -2;
pub const WORLD_MAX_CHUNK_Y: i32 = 1;
pub const TERRAIN_SDF_SAMPLES_PER_CELL: i32 = 2;
const SDF_INF: f32 = 1.0e9;
const SDF_DIAGONAL_COST: f32 = std::f32::consts::SQRT_2;

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
    pub fn generated(chunk_coord: IVec2) -> Self {
        let mut chunk = Self {
            cells: [TerrainCell::Empty; CHUNK_SIZE * CHUNK_SIZE],
            dirty: true,
        };

        for local_x in 0..CHUNK_SIZE_I32 {
            let global_x = chunk_coord.x * CHUNK_SIZE_I32 + local_x;
            for local_y in 0..CHUNK_SIZE_I32 {
                let global_y = chunk_coord.y * CHUNK_SIZE_I32 + local_y;
                let local = IVec2::new(local_x, local_y);
                let global_cell = IVec2::new(global_x, global_y);
                chunk.cells[local_cell_to_index(local)] = generated_cell_for_world(global_cell);
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

    pub fn is_pristine_generated(&self, chunk_coord: IVec2) -> bool {
        for local_y in 0..CHUNK_SIZE_I32 {
            for local_x in 0..CHUNK_SIZE_I32 {
                let local = IVec2::new(local_x, local_y);
                let global_cell = chunk_coord * CHUNK_SIZE_I32 + local;
                if self.get(local) != generated_cell_for_world(global_cell) {
                    return false;
                }
            }
        }
        true
    }
}

pub fn surface_y_for_world_x(world_x: i32) -> i32 {
    generation::surface_y_for_world_x(world_x)
}

pub fn generated_cell_for_world(global_cell: IVec2) -> TerrainCell {
    generation::generated_material_for_world_cell(global_cell)
        .map(TerrainCell::solid)
        .unwrap_or(TerrainCell::Empty)
}

#[derive(Resource, Debug, Default)]
pub struct TerrainWorld {
    chunks: HashMap<IVec2, TerrainChunk>,
    dirty_chunks: HashSet<IVec2>,
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

impl TerrainWorld {
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.dirty_chunks.clear();
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

    pub fn unload_pristine_chunks_outside_radius(&mut self, center_chunk: IVec2, keep_radius: i32) {
        let keep_radius = keep_radius.max(0);
        let unload_targets: Vec<IVec2> = self
            .chunks
            .iter()
            .filter_map(|(&chunk_coord, chunk)| {
                let dx = (chunk_coord.x - center_chunk.x).abs();
                let dy = (chunk_coord.y - center_chunk.y).abs();
                if dx <= keep_radius && dy <= keep_radius {
                    return None;
                }
                if chunk.is_pristine_generated(chunk_coord) {
                    Some(chunk_coord)
                } else {
                    None
                }
            })
            .collect();

        if unload_targets.is_empty() {
            return;
        }
        for chunk_coord in unload_targets {
            self.chunks.remove(&chunk_coord);
            self.dirty_chunks.remove(&chunk_coord);
        }
        self.static_particles_dirty = true;
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
        let Some((min_cell, max_cell)) = self.loaded_cell_bounds() else {
            return;
        };
        self.fill_rect(min_cell, max_cell, TerrainCell::Empty);
    }

    #[allow(dead_code)]
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

        self.rebuild_signed_distance_field();
        self.static_particles_dirty = false;
    }

    pub fn sample_signed_distance_and_normal(&self, world_pos: Vec2) -> Option<(f32, Vec2)> {
        let d = self.sample_signed_distance(world_pos)?;
        let eps = self.sdf_sample_spacing_m.max(1e-4);
        let dx = self
            .sample_signed_distance(world_pos + Vec2::new(eps, 0.0))
            .unwrap_or(d)
            - self
                .sample_signed_distance(world_pos - Vec2::new(eps, 0.0))
                .unwrap_or(d);
        let dy = self
            .sample_signed_distance(world_pos + Vec2::new(0.0, eps))
            .unwrap_or(d)
            - self
                .sample_signed_distance(world_pos - Vec2::new(0.0, eps))
                .unwrap_or(d);
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

    fn rebuild_signed_distance_field(&mut self) {
        let samples_per_cell = TERRAIN_SDF_SAMPLES_PER_CELL.max(1) as usize;
        let Some((min_cell, max_cell)) = self.loaded_cell_bounds() else {
            self.sdf_samples.clear();
            self.sdf_width = 0;
            self.sdf_height = 0;
            self.sdf_sample_spacing_m = 0.0;
            self.sdf_origin_m = Vec2::ZERO;
            self.sdf_min_world_m = Vec2::ZERO;
            self.sdf_max_world_m = Vec2::ZERO;
            return;
        };
        let world_cells_w = (max_cell.x - min_cell.x + 1) as usize;
        let world_cells_h = (max_cell.y - min_cell.y + 1) as usize;

        self.sdf_width = world_cells_w * samples_per_cell;
        self.sdf_height = world_cells_h * samples_per_cell;
        self.sdf_sample_spacing_m = CELL_SIZE_M / samples_per_cell as f32;
        self.sdf_origin_m = Vec2::new(
            min_cell.x as f32 * CELL_SIZE_M + 0.5 * self.sdf_sample_spacing_m,
            min_cell.y as f32 * CELL_SIZE_M + 0.5 * self.sdf_sample_spacing_m,
        );
        self.sdf_min_world_m = Vec2::new(min_cell.x as f32 * CELL_SIZE_M, min_cell.y as f32 * CELL_SIZE_M);
        self.sdf_max_world_m = Vec2::new(
            (max_cell.x + 1) as f32 * CELL_SIZE_M,
            (max_cell.y + 1) as f32 * CELL_SIZE_M,
        );

        let sample_count = self.sdf_width * self.sdf_height;
        if sample_count == 0 {
            self.sdf_samples.clear();
            return;
        }

        let mut solid_mask = vec![false; sample_count];
        for y in 0..self.sdf_height {
            for x in 0..self.sdf_width {
                let idx = sdf_index(self.sdf_width, x, y);
                let sample_pos = self.sdf_origin_m
                    + Vec2::new(
                        x as f32 * self.sdf_sample_spacing_m,
                        y as f32 * self.sdf_sample_spacing_m,
                    );
                let cell = world_to_cell(sample_pos);
                solid_mask[idx] = matches!(
                    self.get_loaded_cell_or_empty(cell),
                    TerrainCell::Solid { .. }
                );
            }
        }
        let non_solid_mask = solid_mask.iter().map(|&v| !v).collect::<Vec<_>>();
        let distance_to_solid = distance_transform(
            &solid_mask,
            self.sdf_width,
            self.sdf_height,
            self.sdf_sample_spacing_m,
        );
        let distance_to_non_solid = distance_transform(
            &non_solid_mask,
            self.sdf_width,
            self.sdf_height,
            self.sdf_sample_spacing_m,
        );
        let half_sample = 0.5 * self.sdf_sample_spacing_m;
        self.sdf_samples.resize(sample_count, 0.0);
        for i in 0..sample_count {
            self.sdf_samples[i] = if solid_mask[i] {
                -distance_to_non_solid[i] + half_sample
            } else {
                distance_to_solid[i] - half_sample
            };
        }
    }

    fn sample_signed_distance(&self, world_pos: Vec2) -> Option<f32> {
        if self.sdf_samples.is_empty() || self.sdf_width == 0 || self.sdf_height == 0 {
            return None;
        }
        if world_pos.x < self.sdf_min_world_m.x
            || world_pos.x >= self.sdf_max_world_m.x
            || world_pos.y < self.sdf_min_world_m.y
            || world_pos.y >= self.sdf_max_world_m.y
        {
            return None;
        }

        let width_max = (self.sdf_width.saturating_sub(1)) as f32;
        let height_max = (self.sdf_height.saturating_sub(1)) as f32;
        let fx =
            ((world_pos.x - self.sdf_origin_m.x) / self.sdf_sample_spacing_m).clamp(0.0, width_max);
        let fy = ((world_pos.y - self.sdf_origin_m.y) / self.sdf_sample_spacing_m)
            .clamp(0.0, height_max);

        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.sdf_width - 1);
        let y1 = (y0 + 1).min(self.sdf_height - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let d00 = self.sdf_samples[sdf_index(self.sdf_width, x0, y0)];
        let d10 = self.sdf_samples[sdf_index(self.sdf_width, x1, y0)];
        let d01 = self.sdf_samples[sdf_index(self.sdf_width, x0, y1)];
        let d11 = self.sdf_samples[sdf_index(self.sdf_width, x1, y1)];

        let dx0 = d00 + (d10 - d00) * tx;
        let dx1 = d01 + (d11 - d01) * tx;
        Some(dx0 + (dx1 - dx0) * ty)
    }

    fn ensure_chunk_mut(&mut self, chunk_coord: IVec2) -> &mut TerrainChunk {
        match self.chunks.entry(chunk_coord) {
            Entry::Vacant(vacant) => {
                let chunk = TerrainChunk::generated(chunk_coord);
                if chunk.is_dirty() {
                    self.dirty_chunks.insert(chunk_coord);
                }
                self.static_particles_dirty = true;
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

fn sdf_index(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn distance_transform(mask: &[bool], width: usize, height: usize, spacing: f32) -> Vec<f32> {
    let mut dist = vec![SDF_INF; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = sdf_index(width, x, y);
            if mask[idx] {
                dist[idx] = 0.0;
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            let idx = sdf_index(width, x, y);
            let mut d = dist[idx];
            if x > 0 {
                d = d.min(dist[sdf_index(width, x - 1, y)] + 1.0);
            }
            if y > 0 {
                d = d.min(dist[sdf_index(width, x, y - 1)] + 1.0);
            }
            if x > 0 && y > 0 {
                d = d.min(dist[sdf_index(width, x - 1, y - 1)] + SDF_DIAGONAL_COST);
            }
            if x + 1 < width && y > 0 {
                d = d.min(dist[sdf_index(width, x + 1, y - 1)] + SDF_DIAGONAL_COST);
            }
            dist[idx] = d;
        }
    }

    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let idx = sdf_index(width, x, y);
            let mut d = dist[idx];
            if x + 1 < width {
                d = d.min(dist[sdf_index(width, x + 1, y)] + 1.0);
            }
            if y + 1 < height {
                d = d.min(dist[sdf_index(width, x, y + 1)] + 1.0);
            }
            if x + 1 < width && y + 1 < height {
                d = d.min(dist[sdf_index(width, x + 1, y + 1)] + SDF_DIAGONAL_COST);
            }
            if x > 0 && y + 1 < height {
                d = d.min(dist[sdf_index(width, x - 1, y + 1)] + SDF_DIAGONAL_COST);
            }
            dist[idx] = d;
        }
    }

    for d in &mut dist {
        if *d < SDF_INF {
            *d *= spacing;
        }
    }

    dist
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
    fn generated_chunk_fills_cells_below_surface() {
        let chunk = TerrainChunk::generated(IVec2::new(0, 0));
        for local_y in 0..CHUNK_SIZE_I32 {
            for local_x in 0..CHUNK_SIZE_I32 {
                let world_x = local_x;
                let world_y = local_y;
                let expected = generated_cell_for_world(IVec2::new(world_x, world_y));
                assert_eq!(chunk.get(IVec2::new(local_x, local_y)), expected);
            }
        }
    }

    #[test]
    fn generated_cell_uses_soil_layer_near_surface() {
        let x = 10;
        let surface_y = surface_y_for_world_x(x);
        assert!(matches!(
            generated_cell_for_world(IVec2::new(x, surface_y)),
            TerrainCell::Solid {
                material: TerrainMaterial::Soil,
                ..
            }
        ));
        assert!(matches!(
            generated_cell_for_world(IVec2::new(x, surface_y - SOIL_DEPTH_CELLS)),
            TerrainCell::Solid {
                material: TerrainMaterial::Stone,
                ..
            }
        ));
    }

    #[test]
    fn generation_rule_is_continuous_across_chunk_boundaries() {
        let left_chunk = TerrainChunk::generated(IVec2::new(-1, 0));
        let right_chunk = TerrainChunk::generated(IVec2::new(0, 0));
        for local_y in 0..CHUNK_SIZE_I32 {
            let left_world = IVec2::new(-1, 0) * CHUNK_SIZE_I32 + IVec2::new(CHUNK_SIZE_I32 - 1, local_y);
            let right_world = IVec2::new(0, 0) * CHUNK_SIZE_I32 + IVec2::new(0, local_y);
            assert_eq!(
                left_chunk.get(IVec2::new(CHUNK_SIZE_I32 - 1, local_y)),
                generated_cell_for_world(left_world)
            );
            assert_eq!(
                right_chunk.get(IVec2::new(0, local_y)),
                generated_cell_for_world(right_world)
            );
        }
    }

    #[test]
    fn surface_function_is_deterministic() {
        for x in -256..=256 {
            assert_eq!(surface_y_for_world_x(x), surface_y_for_world_x(x));
        }
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
        terrain.reset_fixed_world();
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
    fn sdf_sign_matches_solid_and_empty_space() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(0.25);

        let surface_y = surface_y_for_world_x(0);
        let inside_ground = cell_to_world_center(IVec2::new(0, surface_y - 1));
        let above_ground = cell_to_world_center(IVec2::new(0, surface_y + 2));
        let d_inside = terrain
            .sample_signed_distance_and_normal(inside_ground)
            .map(|(d, _)| d)
            .unwrap();
        let d_above = terrain
            .sample_signed_distance_and_normal(above_ground)
            .map(|(d, _)| d)
            .unwrap();

        assert!(d_inside < 0.0);
        assert!(d_above > 0.0);
    }

    #[test]
    fn unload_pristine_chunks_outside_radius_keeps_modified_chunks() {
        let mut terrain = TerrainWorld::default();
        terrain.ensure_chunk_loaded(IVec2::new(0, 0));
        terrain.ensure_chunk_loaded(IVec2::new(5, 0));

        terrain.unload_pristine_chunks_outside_radius(IVec2::new(0, 0), 1);
        assert!(terrain.chunk(IVec2::new(5, 0)).is_none());

        terrain.ensure_chunk_loaded(IVec2::new(5, 0));
        let modified_cell = IVec2::new(5 * CHUNK_SIZE_I32, 0);
        let generated = generated_cell_for_world(modified_cell);
        let modified = if matches!(generated, TerrainCell::Empty) {
            TerrainCell::stone()
        } else {
            TerrainCell::Empty
        };
        assert!(terrain.set_cell(modified_cell, modified));

        terrain.unload_pristine_chunks_outside_radius(IVec2::new(0, 0), 1);
        assert!(terrain.chunk(IVec2::new(5, 0)).is_some());
    }
}
