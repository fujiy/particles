use std::collections::HashMap;

use bevy::prelude::*;

use crate::physics::material::{DEFAULT_MATERIAL_PARAMS, water_kernel_radius_m};
use crate::physics::solver::params_defaults::DEFAULT_SOLVER_PARAMS;

#[derive(Debug, Default)]
pub(crate) struct NeighborGrid {
    grid_min: IVec2,
    grid_size: UVec2,
    cell_starts: Vec<u32>,
    sorted_indices: Vec<usize>,
    sparse_cells: HashMap<IVec2, Vec<usize>>,
}

impl NeighborGrid {
    pub(crate) fn rebuild(&mut self, positions: &[Vec2]) {
        if positions.is_empty() {
            self.grid_min = IVec2::ZERO;
            self.grid_size = UVec2::ZERO;
            self.cell_starts.clear();
            self.sorted_indices.clear();
            self.sparse_cells.clear();
            return;
        }

        let mut min_cell = IVec2::new(i32::MAX, i32::MAX);
        let mut max_cell = IVec2::new(i32::MIN, i32::MIN);
        for &position in positions {
            let cell = particle_grid_cell(position);
            min_cell.x = min_cell.x.min(cell.x);
            min_cell.y = min_cell.y.min(cell.y);
            max_cell.x = max_cell.x.max(cell.x);
            max_cell.y = max_cell.y.max(cell.y);
        }

        let grid_w_i64 = (i64::from(max_cell.x) - i64::from(min_cell.x) + 1).max(1);
        let grid_h_i64 = (i64::from(max_cell.y) - i64::from(min_cell.y) + 1).max(1);
        let dense_cell_count_i64 = grid_w_i64.saturating_mul(grid_h_i64);
        let should_use_sparse = grid_w_i64 > DEFAULT_SOLVER_PARAMS.neighbor_grid_max_axis_cells
            || grid_h_i64 > DEFAULT_SOLVER_PARAMS.neighbor_grid_max_axis_cells
            || dense_cell_count_i64 > DEFAULT_SOLVER_PARAMS.neighbor_grid_max_dense_cells;
        if should_use_sparse {
            self.grid_min = IVec2::ZERO;
            self.grid_size = UVec2::ZERO;
            self.cell_starts.clear();
            self.sorted_indices.clear();
            self.sparse_cells.clear();
            for (index, &position) in positions.iter().enumerate() {
                let cell = particle_grid_cell(position);
                self.sparse_cells.entry(cell).or_default().push(index);
            }
            return;
        }

        let grid_w = grid_w_i64 as u32;
        let grid_h = grid_h_i64 as u32;
        let cell_count = dense_cell_count_i64 as usize;
        self.grid_min = min_cell;
        self.grid_size = UVec2::new(grid_w, grid_h);
        self.sparse_cells.clear();

        let mut counts = vec![0u32; cell_count];
        for position in positions.iter().copied() {
            let Some(cell_index) = self.cell_index_of_world(position) else {
                continue;
            };
            counts[cell_index] += 1;
        }

        self.cell_starts.clear();
        self.cell_starts.resize(cell_count + 1, 0);
        for i in 0..cell_count {
            self.cell_starts[i + 1] = self.cell_starts[i] + counts[i];
        }

        self.sorted_indices.clear();
        self.sorted_indices.resize(positions.len(), 0);
        let mut write_heads = self.cell_starts[..cell_count].to_vec();
        for (idx, &position) in positions.iter().enumerate() {
            let Some(cell_index) = self.cell_index_of_world(position) else {
                continue;
            };
            let head = write_heads[cell_index] as usize;
            self.sorted_indices[head] = idx;
            write_heads[cell_index] += 1;
        }
    }

    pub(crate) fn gather(&self, position: Vec2, out_neighbors: &mut Vec<usize>) {
        out_neighbors.clear();
        let center = particle_grid_cell(position);
        if !self.sparse_cells.is_empty() {
            for y in (center.y - 1)..=(center.y + 1) {
                for x in (center.x - 1)..=(center.x + 1) {
                    let Some(indices) = self.sparse_cells.get(&IVec2::new(x, y)) else {
                        continue;
                    };
                    out_neighbors.extend_from_slice(indices);
                }
            }
            return;
        }
        for y in (center.y - 1)..=(center.y + 1) {
            for x in (center.x - 1)..=(center.x + 1) {
                let Some(cell_index) = self.cell_index_of_cell(IVec2::new(x, y)) else {
                    continue;
                };
                let start = self.cell_starts[cell_index] as usize;
                let end = self.cell_starts[cell_index + 1] as usize;
                out_neighbors.extend_from_slice(&self.sorted_indices[start..end]);
            }
        }
    }

    fn cell_index_of_world(&self, world: Vec2) -> Option<usize> {
        self.cell_index_of_cell(particle_grid_cell(world))
    }

    fn cell_index_of_cell(&self, cell: IVec2) -> Option<usize> {
        if self.grid_size == UVec2::ZERO {
            return None;
        }
        let local = cell - self.grid_min;
        if local.x < 0 || local.y < 0 {
            return None;
        }
        if local.x >= self.grid_size.x as i32 || local.y >= self.grid_size.y as i32 {
            return None;
        }
        Some(local.y as usize * self.grid_size.x as usize + local.x as usize)
    }
}

fn particle_grid_cell(position: Vec2) -> IVec2 {
    let cell_size = water_kernel_radius_m(DEFAULT_MATERIAL_PARAMS);
    IVec2::new(
        (position.x / cell_size).floor() as i32,
        (position.y / cell_size).floor() as i32,
    )
}
