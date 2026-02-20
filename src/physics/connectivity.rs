use std::collections::{HashSet, VecDeque};

use bevy::prelude::*;

pub const FOUR_NEIGHBOR_OFFSETS: [IVec2; 4] = [
    IVec2::new(1, 0),
    IVec2::new(-1, 0),
    IVec2::new(0, 1),
    IVec2::new(0, -1),
];

#[derive(Debug, Clone)]
pub struct FloodFillLimitedResult {
    pub cells: HashSet<IVec2>,
    pub reached_limit: bool,
}

pub fn connected_components_4(cells: &HashSet<IVec2>) -> Vec<Vec<IVec2>> {
    let mut visited = HashSet::with_capacity(cells.len());
    let mut components = Vec::new();

    for &start in cells {
        if visited.contains(&start) {
            continue;
        }
        let mut queue = VecDeque::new();
        let mut component = Vec::new();
        visited.insert(start);
        queue.push_back(start);

        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            for offset in FOUR_NEIGHBOR_OFFSETS {
                let next = cell + offset;
                if !cells.contains(&next) || visited.contains(&next) {
                    continue;
                }
                visited.insert(next);
                queue.push_back(next);
            }
        }

        components.push(component);
    }

    components
}

pub fn flood_fill_4_limited(
    start: IVec2,
    max_cells: usize,
    mut contains: impl FnMut(IVec2) -> bool,
) -> FloodFillLimitedResult {
    if max_cells == 0 || !contains(start) {
        return FloodFillLimitedResult {
            cells: HashSet::new(),
            reached_limit: max_cells == 0,
        };
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(cell) = queue.pop_front() {
        if visited.len() > max_cells {
            return FloodFillLimitedResult {
                cells: visited,
                reached_limit: true,
            };
        }

        for offset in FOUR_NEIGHBOR_OFFSETS {
            let next = cell + offset;
            if visited.contains(&next) || !contains(next) {
                continue;
            }
            visited.insert(next);
            queue.push_back(next);
        }
    }

    FloodFillLimitedResult {
        cells: visited,
        reached_limit: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connected_components_4_splits_diagonal_islands() {
        let cells = HashSet::from([
            IVec2::new(0, 0),
            IVec2::new(1, 0),
            IVec2::new(3, 3),
            IVec2::new(4, 4),
        ]);
        let components = connected_components_4(&cells);
        assert_eq!(components.len(), 3);
    }

    #[test]
    fn flood_fill_4_limited_reports_limit() {
        let cells = HashSet::from([
            IVec2::new(0, 0),
            IVec2::new(1, 0),
            IVec2::new(2, 0),
            IVec2::new(3, 0),
        ]);
        let result = flood_fill_4_limited(IVec2::ZERO, 2, |cell| cells.contains(&cell));
        assert!(result.reached_limit);
        assert!(result.cells.len() >= 3);
    }
}
