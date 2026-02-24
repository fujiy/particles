use std::collections::HashMap;
use std::time::Instant;

use bevy::log::tracing;
use bevy::prelude::*;

use crate::physics::world::grid::GridBlock;
use crate::physics::world::terrain::{
    CELL_SIZE_M, TerrainCell, TerrainWorld, world_to_cell,
};

const CACHE_LOD_MAX: u8 = 3;
const CACHE_LOD_HYSTERESIS_FRAMES: u8 = 6;
const SDF_QUERY_RADIUS_CELLS: i32 = 10;
const SDF_INF: f32 = 1.0e9;

#[derive(Clone, Copy, Debug, Default)]
pub struct TerrainBoundarySample {
    pub sdf_m: f32,
    pub normal: Vec2,
    pub solid: bool,
    pub lod_level: u8,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TerrainBoundaryQueryStats {
    pub sample_query_count: u64,
    pub cache_hit_count: u64,
    pub cache_miss_count: u64,
    pub query_wall_secs: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BlockCacheKey {
    level: u8,
    h_bits: u32,
    origin_node: IVec2,
    node_dims: UVec2,
}

#[derive(Clone, Debug, Default)]
struct BlockSdfCache {
    lod_level: u8,
    pending_lod_level: u8,
    pending_lod_frames: u8,
    samples: Vec<TerrainBoundarySample>,
}

#[derive(Resource, Debug, Default)]
pub struct TerrainBoundarySampler {
    block_caches: HashMap<BlockCacheKey, BlockSdfCache>,
    step_stats: TerrainBoundaryQueryStats,
    last_step_stats: TerrainBoundaryQueryStats,
}

impl TerrainBoundarySampler {
    pub fn clear(&mut self) {
        self.block_caches.clear();
        self.step_stats = TerrainBoundaryQueryStats::default();
        self.last_step_stats = TerrainBoundaryQueryStats::default();
    }

    pub fn begin_step(&mut self) {
        self.step_stats = TerrainBoundaryQueryStats::default();
    }

    pub fn end_step(&mut self) {
        self.last_step_stats = self.step_stats;
    }

    pub fn last_step_stats(&self) -> TerrainBoundaryQueryStats {
        self.last_step_stats
    }

    pub fn step_stats(&self) -> TerrainBoundaryQueryStats {
        self.step_stats
    }

    pub fn sample_solid(&mut self, terrain: &TerrainWorld, world_pos: Vec2) -> bool {
        self.step_stats.sample_query_count = self.step_stats.sample_query_count.saturating_add(1);
        terrain_solid_at_world(terrain, world_pos)
    }

    pub fn sample_sdf(&mut self, terrain: &TerrainWorld, world_pos: Vec2) -> f32 {
        let _span = tracing::info_span!("physics::terrain_boundary::sample_sdf").entered();
        let start = Instant::now();
        self.step_stats.sample_query_count = self.step_stats.sample_query_count.saturating_add(1);
        let sdf = sample_terrain_sdf(terrain, world_pos);
        self.step_stats.query_wall_secs += start.elapsed().as_secs_f64();
        sdf
    }

    pub fn sample_normal(&mut self, terrain: &TerrainWorld, world_pos: Vec2) -> Vec2 {
        let eps = (0.5 * CELL_SIZE_M).max(1e-4);
        let dx = sample_terrain_sdf(terrain, world_pos + Vec2::new(eps, 0.0))
            - sample_terrain_sdf(terrain, world_pos - Vec2::new(eps, 0.0));
        let dy = sample_terrain_sdf(terrain, world_pos + Vec2::new(0.0, eps))
            - sample_terrain_sdf(terrain, world_pos - Vec2::new(0.0, eps));
        Vec2::new(dx, dy).normalize_or_zero()
    }

    pub fn sample_block_nodes(
        &mut self,
        block: &GridBlock,
        terrain: &TerrainWorld,
    ) -> Vec<TerrainBoundarySample> {
        let _span = tracing::info_span!("physics::terrain_boundary::sample_block_nodes").entered();
        let start = Instant::now();
        let key = BlockCacheKey {
            level: block.level,
            h_bits: block.h_b.to_bits(),
            origin_node: block.origin_node,
            node_dims: block.node_dims,
        };
        let node_count = (block.node_dims.x as usize).saturating_mul(block.node_dims.y as usize);
        let desired_lod = desired_lod_level(block);
        let mut needs_rebuild = false;

        let cache = self.block_caches.entry(key).or_insert_with(|| {
            needs_rebuild = true;
            BlockSdfCache {
                lod_level: desired_lod,
                pending_lod_level: desired_lod,
                pending_lod_frames: 0,
                samples: Vec::new(),
            }
        });

        if cache.lod_level != desired_lod {
            if cache.pending_lod_level != desired_lod {
                cache.pending_lod_level = desired_lod;
                cache.pending_lod_frames = 1;
            } else {
                cache.pending_lod_frames = cache.pending_lod_frames.saturating_add(1);
            }
            if cache.pending_lod_frames >= CACHE_LOD_HYSTERESIS_FRAMES {
                cache.lod_level = desired_lod;
                cache.pending_lod_frames = 0;
                needs_rebuild = true;
            }
        } else {
            cache.pending_lod_frames = 0;
        }

        if cache.samples.len() != node_count {
            needs_rebuild = true;
        }

        if needs_rebuild {
            cache.samples = build_block_samples(block, cache.lod_level, terrain);
            self.step_stats.cache_miss_count = self
                .step_stats
                .cache_miss_count
                .saturating_add(node_count as u64);
        } else {
            self.step_stats.cache_hit_count = self
                .step_stats
                .cache_hit_count
                .saturating_add(node_count as u64);
        }

        self.step_stats.sample_query_count = self
            .step_stats
            .sample_query_count
            .saturating_add(node_count as u64);
        self.step_stats.query_wall_secs += start.elapsed().as_secs_f64();
        cache.samples.clone()
    }
}

fn desired_lod_level(block: &GridBlock) -> u8 {
    let base = block.level.min(CACHE_LOD_MAX);
    if block.active_node_count() == 0 {
        (base + 1).min(CACHE_LOD_MAX)
    } else {
        base
    }
}

fn terrain_solid_at_world(terrain: &TerrainWorld, world_pos: Vec2) -> bool {
    matches!(
        terrain.get_cell_or_generated(world_to_cell(world_pos)),
        TerrainCell::Solid { .. }
    )
}

fn sample_terrain_sdf(terrain: &TerrainWorld, world_pos: Vec2) -> f32 {
    let center_cell = world_to_cell(world_pos);
    let inside = matches!(terrain.get_cell_or_generated(center_cell), TerrainCell::Solid { .. });
    let mut best = SDF_INF;
    for dy in -SDF_QUERY_RADIUS_CELLS..=SDF_QUERY_RADIUS_CELLS {
        for dx in -SDF_QUERY_RADIUS_CELLS..=SDF_QUERY_RADIUS_CELLS {
            let cell = center_cell + IVec2::new(dx, dy);
            let cell_is_solid = matches!(
                terrain.get_cell_or_generated(cell),
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

fn build_block_samples(
    block: &GridBlock,
    lod_level: u8,
    terrain: &TerrainWorld,
) -> Vec<TerrainBoundarySample> {
    let width = block.node_dims.x as usize;
    let height = block.node_dims.y as usize;
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let dilation_cells = if lod_level == 0 {
        0
    } else {
        1i32 << lod_level.saturating_sub(1)
    };
    let eps = (0.5 * block.h_b).max(1e-4);
    let mut samples = vec![TerrainBoundarySample::default(); width * height];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let world_node = block.origin_node + IVec2::new(x as i32, y as i32);
            let world_pos = world_node.as_vec2() * block.h_b;
            let (mut sdf_m, mut normal) =
                if let Some((signed_distance, sample_normal)) =
                    terrain.sample_signed_distance_and_normal(world_pos)
                {
                    (signed_distance, sample_normal)
                } else {
                    let signed_distance = sample_terrain_sdf(terrain, world_pos);
                    let dx = sample_terrain_sdf(terrain, world_pos + Vec2::new(eps, 0.0))
                        - sample_terrain_sdf(terrain, world_pos - Vec2::new(eps, 0.0));
                    let dy = sample_terrain_sdf(terrain, world_pos + Vec2::new(0.0, eps))
                        - sample_terrain_sdf(terrain, world_pos - Vec2::new(0.0, eps));
                    (signed_distance, Vec2::new(dx, dy).normalize_or_zero())
                };

            let mut solid = sdf_m < 0.0;
            if !solid && dilation_cells > 0 {
                let cell = world_to_cell(world_pos);
                'scan: for dy in -dilation_cells..=dilation_cells {
                    for dx in -dilation_cells..=dilation_cells {
                        let c = cell + IVec2::new(dx, dy);
                        if matches!(terrain.get_cell_or_generated(c), TerrainCell::Solid { .. }) {
                            solid = true;
                            sdf_m = sdf_m.min(0.0);
                            break 'scan;
                        }
                    }
                }
            }
            if normal == Vec2::ZERO {
                normal = Vec2::Y;
            }
            samples[i] = TerrainBoundarySample {
                sdf_m,
                normal,
                solid,
                lod_level,
            };
        }
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::world::grid::GridBlock;

    #[test]
    fn generated_sdf_sign_is_consistent_with_generated_occupancy() {
        let terrain = TerrainWorld::default();
        let inside = Vec2::new(0.0, -8.0);
        let outside = Vec2::new(0.0, 8.0);
        assert!(sample_terrain_sdf(&terrain, inside) < 0.0);
        assert!(sample_terrain_sdf(&terrain, outside) > 0.0);
    }

    #[test]
    fn disabled_generation_uses_loaded_terrain_only() {
        let mut terrain = TerrainWorld::default();
        terrain.set_generation_enabled(false);
        terrain.ensure_chunk_loaded(IVec2::ZERO);
        terrain.clear_loaded_cells();
        terrain.set_cell(IVec2::new(0, 0), TerrainCell::stone());

        let generated_ground = Vec2::new(0.0, -8.0);
        assert!(sample_terrain_sdf(&terrain, generated_ground) > 0.0);
    }

    #[test]
    fn block_cache_returns_stable_sample_count() {
        let mut sampler = TerrainBoundarySampler::default();
        let block = GridBlock::new(0, 0.25, 1.0 / 60.0, IVec2::new(-8, -8), UVec2::new(16, 16));
        let terrain = TerrainWorld::default();
        sampler.begin_step();
        let a = sampler.sample_block_nodes(&block, &terrain);
        let b = sampler.sample_block_nodes(&block, &terrain);
        sampler.end_step();
        assert_eq!(a.len(), 16 * 16);
        assert_eq!(a.len(), b.len());
        assert!(sampler.last_step_stats().sample_query_count >= (16 * 16 * 2) as u64);
    }
}
