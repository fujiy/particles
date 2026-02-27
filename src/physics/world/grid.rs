use std::collections::HashMap;

use bevy::prelude::*;

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
                if self.nodes[index].m >= mass_threshold {
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
}

/// 8方向の隣接オフセット (x, y)。インデックスは `NEIGHBOR_DIR_*` 定数で参照可能。
pub const NEIGHBOR_OFFSETS: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

#[derive(Resource, Clone, Debug, Default)]
pub struct MpmBlockIndexTable {
    owner_indices: Vec<Vec<usize>>,
    ghost_indices: Vec<Vec<usize>>,
    /// 各blockの8方向隣接block index。`NEIGHBOR_OFFSETS` と同順。
    /// blockレイアウト変化時に `rebuild_neighbor_map` で再計算。
    block_neighbors: Vec<[Option<usize>; 8]>,
    moved_particle_count: usize,
    rebinned_this_step: bool,
}

impl GridHierarchy {
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.node_lookup.clear();
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
        let h = self.blocks.first()?.h_b.max(1e-6);
        let world_node = IVec2::new(
            (world_pos.x / h).floor() as i32,
            (world_pos.y / h).floor() as i32,
        );
        if let Some(location) = self.node_location(world_node) {
            return Some(location.block_index);
        }
        let (min_node, max_node) = self.global_node_bounds()?;
        let clamped = IVec2::new(
            world_node.x.clamp(min_node.x, max_node.x),
            world_node.y.clamp(min_node.y, max_node.y),
        );
        self.node_location(clamped)
            .map(|location| location.block_index)
    }

    pub fn add_block(&mut self, block: GridBlock) {
        self.blocks.push(block);
        self.blocks
            .sort_by_key(|b| (b.level, b.origin_node.y, b.origin_node.x));
        self.rebuild_node_lookup();
    }

    fn rebuild_node_lookup(&mut self) {
        self.node_lookup.clear();
        for (block_index, block) in self.blocks.iter().enumerate() {
            let width = block.node_dims.x as usize;
            for y in 0..block.node_dims.y as i32 {
                for x in 0..block.node_dims.x as i32 {
                    let local_index = y as usize * width + x as usize;
                    let world_node = block.origin_node + IVec2::new(x, y);
                    let previous = self.node_lookup.insert(
                        world_node,
                        GridNodeLocation {
                            block_index,
                            local_index,
                        },
                    );
                    debug_assert!(
                        previous.is_none(),
                        "grid node overlap detected at {:?}: {:?} and block {}",
                        world_node,
                        previous,
                        block_index
                    );
                }
            }
        }
    }

    fn global_node_bounds(&self) -> Option<(IVec2, IVec2)> {
        let mut min_node = IVec2::new(i32::MAX, i32::MAX);
        let mut max_node = IVec2::new(i32::MIN, i32::MIN);
        for block in &self.blocks {
            let block_min = block.origin_node;
            let block_max = block.origin_node
                + IVec2::new(
                    block.node_dims.x.saturating_sub(1) as i32,
                    block.node_dims.y.saturating_sub(1) as i32,
                );
            min_node.x = min_node.x.min(block_min.x);
            min_node.y = min_node.y.min(block_min.y);
            max_node.x = max_node.x.max(block_max.x);
            max_node.y = max_node.y.max(block_max.y);
        }
        if min_node.x > max_node.x || min_node.y > max_node.y {
            None
        } else {
            Some((min_node, max_node))
        }
    }
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
        if self.owner_indices.len() > block_count {
            self.owner_indices.truncate(block_count);
        }
        if self.ghost_indices.len() > block_count {
            self.ghost_indices.truncate(block_count);
        }
        // block_neighbors は rebuild_neighbor_map で明示的に更新する
        if self.block_neighbors.len() != block_count {
            self.block_neighbors.clear();
        }
    }

    /// blockレイアウトから8方向隣接マップを構築する。
    ///
    /// 2ブロックが「隣接」とみなされる条件: それぞれのノード範囲のgapが両軸ともに0
    /// (= 直接接触しており、quadratic B-splineカーネルが境界ノードに届く)。
    /// blockが追加・削除されたタイミングで呼び出すこと。
    pub fn rebuild_neighbor_map(&mut self, blocks: &[GridBlock]) {
        let block_count = blocks.len();
        self.block_neighbors.clear();
        self.block_neighbors
            .resize_with(block_count, || [None; 8]);
        for (i, bi) in blocks.iter().enumerate() {
            let i_origin = bi.origin_node;
            let i_end = i_origin + bi.node_dims.as_ivec2(); // exclusive
            let mut slot = 0usize;
            'outer: for (j, bj) in blocks.iter().enumerate() {
                if j == i {
                    continue;
                }
                let j_origin = bj.origin_node;
                let j_end = j_origin + bj.node_dims.as_ivec2();
                let gap_x = (i_origin.x - j_end.x).max(j_origin.x - i_end.x).max(0);
                let gap_y = (i_origin.y - j_end.y).max(j_origin.y - i_end.y).max(0);
                if gap_x == 0 && gap_y == 0 {
                    if slot < 8 {
                        self.block_neighbors[i][slot] = Some(j);
                        slot += 1;
                    } else {
                        // 8超の隣接は理論上ない（2Dグリッドでは最大8）
                        debug_assert!(false, "block {} has more than 8 neighbors", i);
                        break 'outer;
                    }
                }
            }
        }
    }

    /// block_index の8連結隣接blockのindexを返す。
    /// `None` スロットは隣接blockが存在しない方向。
    pub fn neighbor_block_indices(&self, block_index: usize) -> [Option<usize>; 8] {
        self.block_neighbors
            .get(block_index)
            .copied()
            .unwrap_or([None; 8])
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
}
