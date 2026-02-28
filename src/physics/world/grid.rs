use std::cmp::Reverse;
use std::collections::HashMap;

use bevy::prelude::*;

use crate::physics::world::constants::CELL_SIZE_M;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridNode {
    pub m: f32,
    pub p: Vec2,
    pub v: Vec2,
    pub render_mass_sum: f32,
    pub render_mass_pos_sum: Vec2,
}

#[derive(Clone, Debug)]
pub struct GridBlock {
    pub level: u8,
    pub h_b: f32,
    pub dt_b: f32,
    pub origin_node: IVec2,
    pub node_dims: UVec2,
    color_class: u16,
    owned_nodes: Vec<bool>,
    nodes: Vec<GridNode>,
    active_nodes: Vec<IVec2>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridNodeLocation {
    pub block_index: usize,
    pub local_index: usize,
}

impl GridBlock {
    pub fn new(level: u8, h_b: f32, dt_b: f32, origin_node: IVec2, node_dims: UVec2) -> Self {
        let width = node_dims.x as usize;
        let height = node_dims.y as usize;
        let node_count = width.saturating_mul(height);
        Self {
            level,
            h_b,
            dt_b,
            origin_node,
            node_dims,
            color_class: 0,
            owned_nodes: vec![true; node_count],
            nodes: vec![GridNode::default(); node_count],
            active_nodes: Vec::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn active_nodes(&self) -> &[IVec2] {
        &self.active_nodes
    }

    pub fn active_node_count(&self) -> usize {
        self.active_nodes.len()
    }

    pub fn color_class(&self) -> u16 {
        self.color_class
    }

    fn set_color_class(&mut self, color_class: u16) {
        self.color_class = color_class;
    }

    pub fn cell_dims(&self) -> UVec2 {
        self.node_dims.saturating_sub(UVec2::ONE)
    }

    pub fn world_cell_bounds(&self) -> (IVec2, IVec2) {
        let scale = self.node_scale();
        let min = self.origin_node * scale;
        let max = (self.origin_node + self.cell_dims().as_ivec2()) * scale;
        (min, max)
    }

    pub fn nodes(&self) -> &[GridNode] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut [GridNode] {
        &mut self.nodes
    }

    pub fn clear_nodes(&mut self) {
        self.nodes.fill(GridNode::default());
        self.active_nodes.clear();
    }

    pub fn node_scale(&self) -> i32 {
        1_i32 << self.level.min(30)
    }

    pub fn world_key_from_node(&self, node: IVec2) -> IVec2 {
        node * self.node_scale()
    }

    pub fn world_key_for_local_node(&self, local_node: IVec2) -> IVec2 {
        self.world_key_from_node(self.origin_node + local_node)
    }

    pub fn world_key_to_local(&self, world_key: IVec2) -> Option<UVec2> {
        let scale = self.node_scale();
        if world_key.x.rem_euclid(scale) != 0 || world_key.y.rem_euclid(scale) != 0 {
            return None;
        }
        let node = IVec2::new(world_key.x / scale, world_key.y / scale);
        self.world_node_to_local(node)
    }

    pub fn node_mut_by_world_key(&mut self, world_key: IVec2) -> Option<&mut GridNode> {
        let local = self.world_key_to_local(world_key)?;
        let index = self.local_node_index(local)?;
        self.nodes.get_mut(index)
    }

    pub fn node_by_world_key(&self, world_key: IVec2) -> Option<&GridNode> {
        let local = self.world_key_to_local(world_key)?;
        let index = self.local_node_index(local)?;
        self.nodes.get(index)
    }

    pub fn is_world_key_owned(&self, world_key: IVec2) -> bool {
        let Some(local) = self.world_key_to_local(world_key) else {
            return false;
        };
        let Some(index) = self.local_node_index(local) else {
            return false;
        };
        self.owned_nodes.get(index).copied().unwrap_or(false)
    }

    pub fn is_world_node_owned(&self, world_node: IVec2) -> bool {
        let Some(local) = self.world_node_to_local(world_node) else {
            return false;
        };
        let Some(index) = self.local_node_index(local) else {
            return false;
        };
        self.owned_nodes.get(index).copied().unwrap_or(false)
    }

    pub fn is_local_node_owned(&self, local_index: usize) -> bool {
        self.owned_nodes.get(local_index).copied().unwrap_or(false)
    }

    fn reset_owned_nodes(&mut self) {
        if self.owned_nodes.len() != self.nodes.len() {
            self.owned_nodes.resize(self.nodes.len(), true);
        }
        self.owned_nodes.fill(true);
    }

    fn set_local_node_owned(&mut self, local_index: usize, owned: bool) {
        if let Some(slot) = self.owned_nodes.get_mut(local_index) {
            *slot = owned;
        }
    }

    pub fn local_node_index(&self, local_node: UVec2) -> Option<usize> {
        if local_node.x >= self.node_dims.x || local_node.y >= self.node_dims.y {
            return None;
        }
        let width = self.node_dims.x as usize;
        Some((local_node.y as usize) * width + local_node.x as usize)
    }

    pub fn world_node_to_local(&self, world_node: IVec2) -> Option<UVec2> {
        let local = world_node - self.origin_node;
        if local.x < 0 || local.y < 0 {
            return None;
        }
        let local_u = UVec2::new(local.x as u32, local.y as u32);
        if local_u.x >= self.node_dims.x || local_u.y >= self.node_dims.y {
            return None;
        }
        Some(local_u)
    }

    pub fn node_mut_by_world(&mut self, world_node: IVec2) -> Option<&mut GridNode> {
        let local = self.world_node_to_local(world_node)?;
        let index = self.local_node_index(local)?;
        self.nodes.get_mut(index)
    }

    pub fn node_by_world(&self, world_node: IVec2) -> Option<&GridNode> {
        let local = self.world_node_to_local(world_node)?;
        let index = self.local_node_index(local)?;
        self.nodes.get(index)
    }

    pub fn rebuild_active_nodes(&mut self, mass_threshold: f32) {
        self.active_nodes.clear();
        let width = self.node_dims.x as usize;
        let height = self.node_dims.y as usize;
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                if self.is_local_node_owned(index) && self.nodes[index].m >= mass_threshold {
                    self.active_nodes
                        .push(self.origin_node + IVec2::new(x as i32, y as i32));
                }
            }
        }
    }

    pub fn world_node_min(&self) -> Vec2 {
        self.origin_node.as_vec2() * self.h_b
    }

    pub fn world_node_max(&self) -> Vec2 {
        let max_node = self.origin_node
            + IVec2::new(
                self.node_dims.x.saturating_sub(1) as i32,
                self.node_dims.y.saturating_sub(1) as i32,
            );
        max_node.as_vec2() * self.h_b
    }
}

#[derive(Resource, Debug, Default)]
pub struct GridHierarchy {
    blocks: Vec<GridBlock>,
    node_lookup: HashMap<IVec2, GridNodeLocation>,
    quadtree_index: Option<GridQuadtreeIndex>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct QuadtreeBlockKey {
    level: u8,
    qx: i32,
    qy: i32,
}

#[derive(Clone, Debug)]
struct GridQuadtreeIndex {
    block_span_cells: IVec2,
    min_level: u8,
    max_level: u8,
    block_index_by_key: HashMap<QuadtreeBlockKey, usize>,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct MpmBlockIndexTable {
    owner_indices: Vec<Vec<usize>>,
    ghost_indices: Vec<Vec<usize>>,
    /// 各blockに接触（辺/頂点共有）する近傍block index。
    /// blockレイアウト変化時に `rebuild_neighbor_map` で再計算。
    block_neighbors: Vec<Vec<usize>>,
    moved_particle_count: usize,
    rebinned_this_step: bool,
}

impl GridHierarchy {
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.node_lookup.clear();
        self.quadtree_index = None;
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks(&self) -> &[GridBlock] {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut [GridBlock] {
        &mut self.blocks
    }

    pub fn block_color_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.color_class() as usize)
            .max()
            .map(|max_color| max_color + 1)
            .unwrap_or(0)
    }

    pub fn uses_quadtree_index(&self) -> bool {
        self.quadtree_index.is_some()
    }

    pub fn node_location(&self, world_node: IVec2) -> Option<GridNodeLocation> {
        self.node_lookup.get(&world_node).copied()
    }

    pub fn node_by_world(&self, world_node: IVec2) -> Option<&GridNode> {
        let location = self.node_location(world_node)?;
        self.blocks
            .get(location.block_index)?
            .nodes()
            .get(location.local_index)
    }

    pub fn node_mut_by_world(&mut self, world_node: IVec2) -> Option<&mut GridNode> {
        let location = self.node_location(world_node)?;
        self.blocks
            .get_mut(location.block_index)?
            .nodes_mut()
            .get_mut(location.local_index)
    }

    pub fn block_index_for_position(&self, world_pos: Vec2) -> Option<usize> {
        if let Some(block_index) = self.quadtree_block_index_for_position(world_pos) {
            return Some(block_index);
        }
        // 可変h_b対応: floor(x/h_b)でノード座標を計算し、まず直接lookupを試みる。
        // 見つかればそのblockを返す。見つからない場合は各blockのAABBで判定する。
        // AABBは [origin * h, (origin + dims) * h) = floor(x/h) が [origin, origin+dims-1] に収まる範囲。
        //
        // 均一h_bの場合は元の実装と同じ動作。可変h_bでは全blockのh_bで試みる。
        for block in &self.blocks {
            let h = block.h_b.max(1e-6);
            let world_node = IVec2::new(
                (world_pos.x / h).floor() as i32,
                (world_pos.y / h).floor() as i32,
            );
            let origin = block.origin_node;
            let end = origin + block.node_dims.as_ivec2() - IVec2::ONE;
            if world_node.x >= origin.x
                && world_node.x <= end.x
                && world_node.y >= origin.y
                && world_node.y <= end.y
            {
                let world_key = block.world_key_from_node(world_node);
                if let Some(location) = self.node_location(world_key) {
                    return Some(location.block_index);
                }
            }
        }
        // 範囲外の場合はAABBクランプして最近傍ノードでlookup
        let mut best_index = None;
        let mut best_dist_sq = f32::MAX;
        for (block_index, block) in self.blocks.iter().enumerate() {
            let h = block.h_b.max(1e-6);
            let min_world = block.origin_node.as_vec2() * h;
            let max_world = (block.origin_node + block.node_dims.as_ivec2()).as_vec2() * h;
            let center = (min_world + max_world) * 0.5;
            let dx = (world_pos.x - center.x).abs() - (max_world.x - min_world.x) * 0.5;
            let dy = (world_pos.y - center.y).abs() - (max_world.y - min_world.y) * 0.5;
            let dist_sq = dx.max(0.0) * dx.max(0.0) + dy.max(0.0) * dy.max(0.0);
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_index = Some(block_index);
            }
        }
        best_index
    }

    pub fn touching_block_indices(&self, block_index: usize) -> Vec<usize> {
        let Some(block) = self.blocks.get(block_index) else {
            return Vec::new();
        };
        if let Some(index) = &self.quadtree_index {
            let (min, max) = block.world_cell_bounds();
            let mut neighbors = Vec::<usize>::new();
            for level in index.min_level..=index.max_level {
                let scale = 1_i32 << level.min(30);
                let step_x = (index.block_span_cells.x * scale).max(1);
                let step_y = (index.block_span_cells.y * scale).max(1);
                let qx_start = min.x.div_euclid(step_x) - 1;
                let qx_end = max.x.div_euclid(step_x);
                let qy_start = min.y.div_euclid(step_y) - 1;
                let qy_end = max.y.div_euclid(step_y);
                for qy in qy_start..=qy_end {
                    for qx in qx_start..=qx_end {
                        let key = QuadtreeBlockKey { level, qx, qy };
                        let Some(&candidate) = index.block_index_by_key.get(&key) else {
                            continue;
                        };
                        if candidate != block_index {
                            neighbors.push(candidate);
                        }
                    }
                }
            }
            neighbors.sort_unstable();
            neighbors.dedup();
            neighbors.retain(|&candidate| {
                self.blocks
                    .get(candidate)
                    .map(|other| blocks_touch_by_edge_or_vertex(block, other))
                    .unwrap_or(false)
            });
            return neighbors;
        }
        let mut neighbors = Vec::new();
        for (candidate, other) in self.blocks.iter().enumerate() {
            if candidate == block_index {
                continue;
            }
            if blocks_touch_by_edge_or_vertex(block, other) {
                neighbors.push(candidate);
            }
        }
        neighbors
    }

    pub fn add_block(&mut self, block: GridBlock) {
        self.blocks.push(block);
        self.blocks
            .sort_by_key(|b| (b.level, b.origin_node.y, b.origin_node.x));
        self.rebuild_node_lookup();
    }

    pub fn replace_blocks(&mut self, mut blocks: Vec<GridBlock>) {
        blocks.sort_by_key(|b| (b.level, b.origin_node.y, b.origin_node.x));
        self.blocks = blocks;
        self.rebuild_node_lookup();
    }

    fn prefer_block_owner(&self, lhs_block_index: usize, rhs_block_index: usize) -> bool {
        let lhs = &self.blocks[lhs_block_index];
        let rhs = &self.blocks[rhs_block_index];
        if lhs.level != rhs.level {
            return lhs.level < rhs.level;
        }
        if lhs.origin_node.x != rhs.origin_node.x {
            return lhs.origin_node.x < rhs.origin_node.x;
        }
        if lhs.origin_node.y != rhs.origin_node.y {
            return lhs.origin_node.y < rhs.origin_node.y;
        }
        lhs_block_index < rhs_block_index
    }

    fn set_node_owner_flag(&mut self, location: GridNodeLocation, owned: bool) {
        if let Some(block) = self.blocks.get_mut(location.block_index) {
            block.set_local_node_owned(location.local_index, owned);
        }
    }

    fn rebuild_node_lookup(&mut self) {
        self.node_lookup.clear();
        for block in &mut self.blocks {
            block.reset_owned_nodes();
        }
        for block_index in 0..self.blocks.len() {
            let node_dims = {
                let block = &self.blocks[block_index];
                block.node_dims
            };
            let width = node_dims.x as usize;
            for y in 0..node_dims.y as i32 {
                for x in 0..node_dims.x as i32 {
                    let local_index = y as usize * width + x as usize;
                    let world_node = {
                        let block = &self.blocks[block_index];
                        block.world_key_for_local_node(IVec2::new(x, y))
                    };
                    let candidate = GridNodeLocation {
                        block_index,
                        local_index,
                    };
                    let Some(previous) = self.node_lookup.get(&world_node).copied() else {
                        self.node_lookup.insert(world_node, candidate);
                        continue;
                    };
                    if self.prefer_block_owner(candidate.block_index, previous.block_index) {
                        self.set_node_owner_flag(previous, false);
                        self.node_lookup.insert(world_node, candidate);
                    } else {
                        self.set_node_owner_flag(candidate, false);
                    }
                }
            }
        }
        self.rebuild_quadtree_index();
        self.recompute_block_colors();
    }

    fn rebuild_quadtree_index(&mut self) {
        let Some(first) = self.blocks.first() else {
            self.quadtree_index = None;
            return;
        };
        let span = first.cell_dims().as_ivec2();
        if span.x <= 0 || span.y <= 0 {
            self.quadtree_index = None;
            return;
        }
        let mut min_level = u8::MAX;
        let mut max_level = 0_u8;
        let mut block_index_by_key = HashMap::<QuadtreeBlockKey, usize>::new();
        for (block_index, block) in self.blocks.iter().enumerate() {
            if block.cell_dims().as_ivec2() != span {
                self.quadtree_index = None;
                return;
            }
            if block.origin_node.x.rem_euclid(span.x) != 0
                || block.origin_node.y.rem_euclid(span.y) != 0
            {
                self.quadtree_index = None;
                return;
            }
            let key = QuadtreeBlockKey {
                level: block.level,
                qx: block.origin_node.x.div_euclid(span.x),
                qy: block.origin_node.y.div_euclid(span.y),
            };
            if block_index_by_key.insert(key, block_index).is_some() {
                self.quadtree_index = None;
                return;
            }
            min_level = min_level.min(block.level);
            max_level = max_level.max(block.level);
        }
        self.quadtree_index = Some(GridQuadtreeIndex {
            block_span_cells: span,
            min_level: if min_level == u8::MAX { 0 } else { min_level },
            max_level,
            block_index_by_key,
        });
    }

    fn quadtree_block_index_for_position(&self, world_pos: Vec2) -> Option<usize> {
        let index = self.quadtree_index.as_ref()?;
        let base_cell = IVec2::new(
            (world_pos.x / CELL_SIZE_M).floor() as i32,
            (world_pos.y / CELL_SIZE_M).floor() as i32,
        );
        for level in index.min_level..=index.max_level {
            let scale = 1_i32 << level.min(30);
            let step_x = (index.block_span_cells.x * scale).max(1);
            let step_y = (index.block_span_cells.y * scale).max(1);
            let key = QuadtreeBlockKey {
                level,
                qx: base_cell.x.div_euclid(step_x),
                qy: base_cell.y.div_euclid(step_y),
            };
            if let Some(&block_index) = index.block_index_by_key.get(&key) {
                return Some(block_index);
            }
        }
        None
    }

    fn recompute_block_colors(&mut self) {
        let block_count = self.blocks.len();
        if block_count == 0 {
            return;
        }

        let mut conflict_neighbors = vec![Vec::<usize>::new(); block_count];
        if self.quadtree_index.is_some() {
            for i in 0..block_count {
                for neighbor in self.touching_block_indices(i) {
                    if neighbor == i {
                        continue;
                    }
                    if !conflict_neighbors[i].contains(&neighbor) {
                        conflict_neighbors[i].push(neighbor);
                    }
                }
            }
        } else {
            for i in 0..block_count {
                for j in (i + 1)..block_count {
                    if !blocks_touch_by_edge_or_vertex(&self.blocks[i], &self.blocks[j]) {
                        continue;
                    }
                    conflict_neighbors[i].push(j);
                    conflict_neighbors[j].push(i);
                }
            }
        }

        let mut order: Vec<usize> = (0..block_count).collect();
        order.sort_by_key(|&block_index| {
            let block = &self.blocks[block_index];
            (
                Reverse(conflict_neighbors[block_index].len()),
                block.level,
                block.origin_node.y,
                block.origin_node.x,
            )
        });

        let mut colors = vec![u16::MAX; block_count];
        let mut used = vec![false; block_count.saturating_add(1)];
        for block_index in order {
            used.fill(false);
            for &neighbor_index in &conflict_neighbors[block_index] {
                let neighbor_color = colors[neighbor_index];
                if neighbor_color == u16::MAX {
                    continue;
                }
                let color_index = neighbor_color as usize;
                if color_index >= used.len() {
                    used.resize(color_index + 1, false);
                }
                used[color_index] = true;
            }
            let mut color = 0usize;
            while color < used.len() && used[color] {
                color += 1;
            }
            debug_assert!(color <= u16::MAX as usize);
            colors[block_index] = color.min(u16::MAX as usize) as u16;
        }

        for (block_index, &color) in colors.iter().enumerate() {
            if let Some(block) = self.blocks.get_mut(block_index) {
                block.set_color_class(color);
            }
        }
    }
}

fn intervals_touch_or_overlap(min_a: i32, max_a: i32, min_b: i32, max_b: i32) -> bool {
    let gap = (min_a - max_b).max(min_b - max_a).max(0);
    gap == 0
}

fn blocks_touch_by_edge_or_vertex(lhs: &GridBlock, rhs: &GridBlock) -> bool {
    let (lhs_min, lhs_max) = lhs.world_cell_bounds();
    let (rhs_min, rhs_max) = rhs.world_cell_bounds();
    intervals_touch_or_overlap(lhs_min.x, lhs_max.x, rhs_min.x, rhs_max.x)
        && intervals_touch_or_overlap(lhs_min.y, lhs_max.y, rhs_min.y, rhs_max.y)
}

impl MpmBlockIndexTable {
    pub fn clear(&mut self) {
        self.owner_indices.clear();
        self.ghost_indices.clear();
        self.block_neighbors.clear();
        self.moved_particle_count = 0;
        self.rebinned_this_step = false;
    }

    pub fn ensure_block_count(&mut self, block_count: usize) {
        self.owner_indices.resize_with(block_count, Vec::new);
        self.ghost_indices.resize_with(block_count, Vec::new);
        self.block_neighbors.resize_with(block_count, Vec::new);
        if self.owner_indices.len() > block_count {
            self.owner_indices.truncate(block_count);
        }
        if self.ghost_indices.len() > block_count {
            self.ghost_indices.truncate(block_count);
        }
        if self.block_neighbors.len() > block_count {
            self.block_neighbors.truncate(block_count);
        }
    }

    /// blockレイアウトから近傍マップを構築する。
    ///
    /// 2ブロックが「近傍」とみなされる条件: 辺または頂点を共有して接触していること。
    /// blockが追加・削除されたタイミングで呼び出すこと。
    pub fn rebuild_neighbor_map(&mut self, grid: &GridHierarchy) {
        let block_count = grid.block_count();
        self.block_neighbors.clear();
        self.block_neighbors.resize_with(block_count, Vec::new);
        for i in 0..block_count {
            if let Some(slot) = self.block_neighbors.get_mut(i) {
                *slot = grid.touching_block_indices(i);
            }
        }
    }

    /// block_index に接触する近傍blockのindex一覧を返す。
    pub fn neighbor_block_indices(&self, block_index: usize) -> &[usize] {
        self.block_neighbors
            .get(block_index)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn owner_indices(&self, block_index: usize) -> &[usize] {
        self.owner_indices
            .get(block_index)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn ghost_indices(&self, block_index: usize) -> &[usize] {
        self.ghost_indices
            .get(block_index)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn owner_indices_mut(&mut self, block_index: usize) -> Option<&mut Vec<usize>> {
        self.owner_indices.get_mut(block_index)
    }

    pub fn ghost_indices_mut(&mut self, block_index: usize) -> Option<&mut Vec<usize>> {
        self.ghost_indices.get_mut(block_index)
    }

    pub fn moved_particle_count(&self) -> usize {
        self.moved_particle_count
    }

    pub fn block_count(&self) -> usize {
        self.owner_indices.len().min(self.ghost_indices.len())
    }

    pub fn set_moved_particle_count(&mut self, moved_particle_count: usize) {
        self.moved_particle_count = moved_particle_count;
    }

    pub fn rebinned_this_step(&self) -> bool {
        self.rebinned_this_step
    }

    pub fn set_rebinned_this_step(&mut self, rebinned_this_step: bool) {
        self.rebinned_this_step = rebinned_this_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepare_block_with_same_mass_different_write_order(reverse: bool) -> GridBlock {
        let mut block = GridBlock::new(0, 0.25, 1.0 / 60.0, IVec2::new(-2, -2), UVec2::new(4, 4));
        let mut targets = vec![
            IVec2::new(-2, -2),
            IVec2::new(-1, -2),
            IVec2::new(0, 0),
            IVec2::new(1, 1),
        ];
        if reverse {
            targets.reverse();
        }
        for node in targets {
            block.node_mut_by_world(node).unwrap().m = 0.125;
        }
        block
    }

    #[test]
    fn active_nodes_are_deterministic_for_fixed_threshold_and_contents() {
        let mut a = prepare_block_with_same_mass_different_write_order(false);
        let mut b = prepare_block_with_same_mass_different_write_order(true);
        a.rebuild_active_nodes(0.1);
        b.rebuild_active_nodes(0.1);
        assert_eq!(a.active_nodes(), b.active_nodes());
        assert_eq!(
            a.active_nodes(),
            &[
                IVec2::new(-2, -2),
                IVec2::new(-1, -2),
                IVec2::new(0, 0),
                IVec2::new(1, 1),
            ]
        );
    }

    #[test]
    fn hierarchy_node_lookup_resolves_to_expected_block() {
        let mut hierarchy = GridHierarchy::default();
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(0, 0),
            UVec2::new(4, 4),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(4, 0),
            UVec2::new(4, 4),
        ));
        let left = hierarchy
            .node_location(IVec2::new(1, 1))
            .expect("left node must be indexed");
        let right = hierarchy
            .node_location(IVec2::new(5, 1))
            .expect("right node must be indexed");
        assert_ne!(left.block_index, right.block_index);
        assert_eq!(
            hierarchy.block_index_for_position(Vec2::new(0.30, 0.30)),
            Some(left.block_index)
        );
        assert_eq!(
            hierarchy.block_index_for_position(Vec2::new(1.30, 0.30)),
            Some(right.block_index)
        );
    }

    #[test]
    fn shared_boundary_node_prefers_lower_level_block_owner() {
        let mut hierarchy = GridHierarchy::default();
        hierarchy.add_block(GridBlock::new(
            1,
            0.5,
            1.0 / 60.0,
            IVec2::new(0, 0),
            UVec2::new(17, 17),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(16, 0),
            UVec2::new(17, 17),
        ));
        let shared = IVec2::new(16, 8);
        let owner = hierarchy
            .node_location(shared)
            .expect("shared node must have an owner");
        assert_eq!(hierarchy.blocks()[owner.block_index].level, 0);
    }

    #[test]
    fn shared_boundary_node_prefers_smaller_origin_when_same_level() {
        let mut hierarchy = GridHierarchy::default();
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(0, 0),
            UVec2::new(17, 17),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(16, 0),
            UVec2::new(17, 17),
        ));
        let shared = IVec2::new(16, 8);
        let owner = hierarchy
            .node_location(shared)
            .expect("shared node must have an owner");
        assert_eq!(
            hierarchy.blocks()[owner.block_index].origin_node,
            IVec2::new(0, 0)
        );
    }

    #[test]
    fn greedy_coloring_assigns_distinct_colors_to_2x2_blocks() {
        let mut hierarchy = GridHierarchy::default();
        for y in 0..2 {
            for x in 0..2 {
                hierarchy.add_block(GridBlock::new(
                    0,
                    0.25,
                    1.0 / 60.0,
                    IVec2::new(x * 16, y * 16),
                    UVec2::new(17, 17),
                ));
            }
        }

        assert_eq!(hierarchy.block_color_count(), 4);
        let blocks = hierarchy.blocks();
        for i in 0..blocks.len() {
            for j in (i + 1)..blocks.len() {
                if blocks_touch_by_edge_or_vertex(&blocks[i], &blocks[j]) {
                    assert_ne!(
                        blocks[i].color_class(),
                        blocks[j].color_class(),
                        "touching blocks must have different colors"
                    );
                }
            }
        }
    }

    #[test]
    fn greedy_coloring_distinguishes_vertex_touching_blocks() {
        let mut hierarchy = GridHierarchy::default();
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(0, 0),
            UVec2::new(17, 17),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(16, 0),
            UVec2::new(17, 17),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(16, 16),
            UVec2::new(17, 17),
        ));

        let blocks = hierarchy.blocks();
        let color_origin_00 = blocks
            .iter()
            .find(|block| block.origin_node == IVec2::new(0, 0))
            .map(GridBlock::color_class)
            .expect("origin (0,0) block must exist");
        let color_origin_16_16 = blocks
            .iter()
            .find(|block| block.origin_node == IVec2::new(16, 16))
            .map(GridBlock::color_class)
            .expect("origin (16,16) block must exist");
        assert_ne!(color_origin_00, color_origin_16_16);
    }

    fn find_block_index(hierarchy: &GridHierarchy, level: u8, origin: IVec2) -> usize {
        hierarchy
            .blocks()
            .iter()
            .enumerate()
            .find_map(|(index, block)| {
                if block.level == level && block.origin_node == origin {
                    Some(index)
                } else {
                    None
                }
            })
            .expect("target block must exist")
    }

    #[test]
    fn quadtree_index_enabled_for_aligned_layout_and_position_query() {
        let mut hierarchy = GridHierarchy::default();
        hierarchy.add_block(GridBlock::new(
            1,
            0.5,
            1.0 / 60.0,
            IVec2::new(-16, 0),
            UVec2::new(17, 17),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(0, 0),
            UVec2::new(17, 17),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(16, 0),
            UVec2::new(17, 17),
        ));
        assert!(hierarchy.uses_quadtree_index());

        let coarse_index = find_block_index(&hierarchy, 1, IVec2::new(-16, 0));
        let fine_index = find_block_index(&hierarchy, 0, IVec2::new(0, 0));
        assert_eq!(
            hierarchy.block_index_for_position(Vec2::new(-7.5, 1.0)),
            Some(coarse_index)
        );
        assert_eq!(
            hierarchy.block_index_for_position(Vec2::new(1.0, 1.0)),
            Some(fine_index)
        );
    }

    #[test]
    fn touching_block_indices_use_quadtree_candidates() {
        let mut hierarchy = GridHierarchy::default();
        hierarchy.add_block(GridBlock::new(
            1,
            0.5,
            1.0 / 60.0,
            IVec2::new(-16, 0),
            UVec2::new(17, 17),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(0, 0),
            UVec2::new(17, 17),
        ));
        hierarchy.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(0, 16),
            UVec2::new(17, 17),
        ));
        assert!(hierarchy.uses_quadtree_index());

        let coarse_index = find_block_index(&hierarchy, 1, IVec2::new(-16, 0));
        let mut neighbors = hierarchy.touching_block_indices(coarse_index);
        neighbors.sort_unstable();
        assert_eq!(neighbors.len(), 2);
        assert_eq!(
            neighbors,
            vec![
                find_block_index(&hierarchy, 0, IVec2::new(0, 0)),
                find_block_index(&hierarchy, 0, IVec2::new(0, 16)),
            ]
        );
    }
}
