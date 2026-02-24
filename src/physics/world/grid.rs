use bevy::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridNode {
    pub m: f32,
    pub p: Vec2,
    pub v: Vec2,
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
}

impl GridHierarchy {
    pub fn clear(&mut self) {
        self.blocks.clear();
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

    pub fn add_block(&mut self, block: GridBlock) {
        self.blocks.push(block);
        self.blocks
            .sort_by_key(|b| (b.level, b.origin_node.y, b.origin_node.x));
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
}
