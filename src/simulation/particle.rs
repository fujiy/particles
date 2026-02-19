use std::collections::HashMap;

use bevy::prelude::*;

use super::terrain::{CELL_SIZE_M, TerrainCell, TerrainWorld, world_to_cell};

pub const GRAVITY_MPS2: Vec2 = Vec2::new(0.0, -9.81);
pub const FIXED_DT: f32 = 1.0 / 60.0;
pub const SUBSTEPS: usize = 2;
pub const SOLVER_ITERS: usize = 6;
pub const PARTICLE_SPACING_M: f32 = 0.125;
pub const PARTICLE_RADIUS_M: f32 = 0.06;
pub const SMOOTHING_RADIUS_M: f32 = 0.25;
pub const REST_DENSITY: f32 = 1000.0;
pub const EPSILON_LAMBDA: f32 = 1e-6;
pub const XSPH_VISCOSITY: f32 = 0.02;
pub const TERRAIN_RESTITUTION: f32 = 0.0;

const INITIAL_WATER_ORIGIN_M: Vec2 = Vec2::new(-1.6, 2.0);
const INITIAL_WATER_COLS: usize = 50;
const INITIAL_WATER_ROWS: usize = 30;
const PARTICLE_SPEED_LIMIT_MPS: f32 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleMaterial {
    Water,
}

#[derive(Debug, Default)]
struct NeighborGrid {
    buckets: HashMap<IVec2, Vec<usize>>,
}

impl NeighborGrid {
    fn rebuild(&mut self, positions: &[Vec2]) {
        self.buckets.clear();
        for (idx, position) in positions.iter().enumerate() {
            let cell = particle_grid_cell(*position);
            self.buckets.entry(cell).or_default().push(idx);
        }
    }

    fn gather(&self, position: Vec2, out_neighbors: &mut Vec<usize>) {
        out_neighbors.clear();
        let center = particle_grid_cell(position);
        for y in (center.y - 1)..=(center.y + 1) {
            for x in (center.x - 1)..=(center.x + 1) {
                let Some(indices) = self.buckets.get(&IVec2::new(x, y)) else {
                    continue;
                };
                out_neighbors.extend(indices.iter().copied());
            }
        }
    }
}

#[derive(Resource, Debug)]
pub struct ParticleWorld {
    pub pos: Vec<Vec2>,
    pub prev_pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    pub mass: Vec<f32>,
    pub material: Vec<ParticleMaterial>,
    pub density: Vec<f32>,
    pub lambda: Vec<f32>,
    pub delta_pos: Vec<Vec2>,
    initial_pos: Vec<Vec2>,
    initial_vel: Vec<Vec2>,
    neighbor_grid: NeighborGrid,
    viscosity_work: Vec<Vec2>,
}

impl Default for ParticleWorld {
    fn default() -> Self {
        let initial_pos = generate_initial_water_positions();
        let initial_vel = vec![Vec2::ZERO; initial_pos.len()];
        let mut world = Self {
            pos: initial_pos.clone(),
            prev_pos: initial_pos.clone(),
            vel: initial_vel.clone(),
            mass: vec![default_particle_mass(); initial_pos.len()],
            material: vec![ParticleMaterial::Water; initial_pos.len()],
            density: Vec::new(),
            lambda: Vec::new(),
            delta_pos: Vec::new(),
            initial_pos,
            initial_vel,
            neighbor_grid: NeighborGrid::default(),
            viscosity_work: Vec::new(),
        };
        world.resize_work_buffers();
        world
    }
}

impl ParticleWorld {
    pub fn reset_to_initial(&mut self) {
        self.pos.clone_from(&self.initial_pos);
        self.prev_pos.clone_from(&self.initial_pos);
        self.vel.clone_from(&self.initial_vel);
        self.mass.fill(default_particle_mass());
        self.material.fill(ParticleMaterial::Water);
        self.resize_work_buffers();
    }

    pub fn particle_count(&self) -> usize {
        self.pos.len()
    }

    pub fn positions(&self) -> &[Vec2] {
        &self.pos
    }

    pub fn step_if_running(&mut self, terrain: &TerrainWorld, running: bool) {
        if running {
            self.step_substeps(terrain);
        }
    }

    pub fn step_substeps(&mut self, terrain: &TerrainWorld) {
        let dt_sub = FIXED_DT / SUBSTEPS as f32;
        for _ in 0..SUBSTEPS {
            self.step_single_substep(terrain, dt_sub);
        }
    }

    fn step_single_substep(&mut self, terrain: &TerrainWorld, dt_sub: f32) {
        for i in 0..self.particle_count() {
            self.prev_pos[i] = self.pos[i];
            self.vel[i] += GRAVITY_MPS2 * dt_sub;
            self.pos[i] += self.vel[i] * dt_sub;
        }

        self.neighbor_grid.rebuild(&self.pos);
        for _ in 0..SOLVER_ITERS {
            self.solve_density_constraints(terrain);
        }

        for i in 0..self.particle_count() {
            self.vel[i] = (self.pos[i] - self.prev_pos[i]) / dt_sub;
            self.project_terrain_collision(i, terrain, true);
            self.vel[i] = self.vel[i].clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
        }

        self.apply_xsph_viscosity();
    }

    fn solve_density_constraints(&mut self, terrain: &TerrainWorld) {
        let mut neighbors = Vec::new();

        for i in 0..self.particle_count() {
            self.neighbor_grid.gather(self.pos[i], &mut neighbors);

            let mut rho = 0.0;
            let mut grad_i = Vec2::ZERO;
            let mut grad_sum_sq = 0.0;

            for &j in &neighbors {
                let r = self.pos[i] - self.pos[j];
                let r2 = r.length_squared();
                if r2 >= SMOOTHING_RADIUS_M * SMOOTHING_RADIUS_M {
                    continue;
                }

                rho += self.mass[j] * kernel_poly6(r2);

                if i == j {
                    continue;
                }

                let grad = (self.mass[j] / REST_DENSITY) * kernel_spiky_grad(r);
                grad_sum_sq += grad.length_squared();
                grad_i += grad;
            }

            grad_sum_sq += grad_i.length_squared();
            let c_i = rho / REST_DENSITY - 1.0;

            self.density[i] = rho;
            self.lambda[i] = -c_i / (grad_sum_sq + EPSILON_LAMBDA);
        }

        for i in 0..self.particle_count() {
            self.neighbor_grid.gather(self.pos[i], &mut neighbors);

            let mut delta = Vec2::ZERO;
            for &j in &neighbors {
                if i == j {
                    continue;
                }
                let r = self.pos[i] - self.pos[j];
                if r.length_squared() >= SMOOTHING_RADIUS_M * SMOOTHING_RADIUS_M {
                    continue;
                }
                delta += (self.lambda[i] + self.lambda[j]) * kernel_spiky_grad(r);
            }
            self.delta_pos[i] = delta / REST_DENSITY;
        }

        for i in 0..self.particle_count() {
            self.pos[i] += self.delta_pos[i];
            self.project_terrain_collision(i, terrain, false);
        }
    }

    fn project_terrain_collision(
        &mut self,
        particle_index: usize,
        terrain: &TerrainWorld,
        damp: bool,
    ) {
        let mut position = self.pos[particle_index];
        let radius = PARTICLE_RADIUS_M;
        for _ in 0..4 {
            let min_cell = world_to_cell(position - Vec2::splat(radius));
            let max_cell = world_to_cell(position + Vec2::splat(radius));
            let mut penetrated = false;

            for y in min_cell.y..=max_cell.y {
                for x in min_cell.x..=max_cell.x {
                    let cell = terrain.get_loaded_cell_or_empty(IVec2::new(x, y));
                    if !matches!(cell, TerrainCell::Solid { .. }) {
                        continue;
                    }

                    if let Some(normal) =
                        push_circle_out_of_cell(&mut position, radius, IVec2::new(x, y))
                    {
                        penetrated = true;
                        if damp {
                            let vn = self.vel[particle_index].dot(normal);
                            if vn < 0.0 {
                                self.vel[particle_index] -=
                                    (1.0 + TERRAIN_RESTITUTION) * vn * normal;
                            }
                        }
                    }
                }
            }

            if !penetrated {
                break;
            }
        }

        self.pos[particle_index] = position;
    }

    fn apply_xsph_viscosity(&mut self) {
        let mut neighbors = Vec::new();
        self.viscosity_work.clone_from(&self.vel);

        for i in 0..self.particle_count() {
            self.neighbor_grid.gather(self.pos[i], &mut neighbors);

            let mut correction = Vec2::ZERO;
            for &j in &neighbors {
                if i == j {
                    continue;
                }
                let r = self.pos[i] - self.pos[j];
                let w = kernel_poly6(r.length_squared());
                correction += (self.vel[j] - self.vel[i]) * w;
            }
            self.viscosity_work[i] = self.vel[i] + XSPH_VISCOSITY * correction;
        }

        self.vel.clone_from(&self.viscosity_work);
    }

    fn resize_work_buffers(&mut self) {
        let count = self.particle_count();
        self.prev_pos.resize(count, Vec2::ZERO);
        self.vel.resize(count, Vec2::ZERO);
        self.mass.resize(count, default_particle_mass());
        self.material.resize(count, ParticleMaterial::Water);
        self.density.resize(count, 0.0);
        self.lambda.resize(count, 0.0);
        self.delta_pos.resize(count, Vec2::ZERO);
        self.viscosity_work.resize(count, Vec2::ZERO);
    }
}

fn generate_initial_water_positions() -> Vec<Vec2> {
    let mut positions = Vec::with_capacity(INITIAL_WATER_COLS * INITIAL_WATER_ROWS);
    for row in 0..INITIAL_WATER_ROWS {
        for col in 0..INITIAL_WATER_COLS {
            positions.push(
                INITIAL_WATER_ORIGIN_M
                    + Vec2::new(col as f32, row as f32) * PARTICLE_SPACING_M
                    + Vec2::splat(PARTICLE_RADIUS_M),
            );
        }
    }
    positions
}

fn default_particle_mass() -> f32 {
    REST_DENSITY * PARTICLE_SPACING_M * PARTICLE_SPACING_M
}

fn particle_grid_cell(position: Vec2) -> IVec2 {
    IVec2::new(
        (position.x / SMOOTHING_RADIUS_M).floor() as i32,
        (position.y / SMOOTHING_RADIUS_M).floor() as i32,
    )
}

fn kernel_poly6(r2: f32) -> f32 {
    let h2 = SMOOTHING_RADIUS_M * SMOOTHING_RADIUS_M;
    if !(0.0..h2).contains(&r2) {
        return 0.0;
    }
    let x = h2 - r2;
    let coeff = 4.0 / (std::f32::consts::PI * h2.powi(4));
    coeff * x.powi(3)
}

fn kernel_spiky_grad(r: Vec2) -> Vec2 {
    let len = r.length();
    if !(1e-6..SMOOTHING_RADIUS_M).contains(&len) {
        return Vec2::ZERO;
    }
    let coeff = -30.0 / (std::f32::consts::PI * SMOOTHING_RADIUS_M.powi(5));
    r * (coeff * (SMOOTHING_RADIUS_M - len).powi(2) / len)
}

fn push_circle_out_of_cell(position: &mut Vec2, radius: f32, cell: IVec2) -> Option<Vec2> {
    let cell_min = cell.as_vec2() * CELL_SIZE_M;
    let cell_max = cell_min + Vec2::splat(CELL_SIZE_M);
    let closest = position.clamp(cell_min, cell_max);
    let delta = *position - closest;
    let dist_sq = delta.length_squared();

    if dist_sq >= radius * radius {
        return None;
    }

    if dist_sq > 1e-12 {
        let dist = dist_sq.sqrt();
        let normal = delta / dist;
        let correction = radius - dist;
        *position += normal * correction;
        return Some(normal);
    }

    let left = (position.x - cell_min.x).abs();
    let right = (cell_max.x - position.x).abs();
    let bottom = (position.y - cell_min.y).abs();
    let top = (cell_max.y - position.y).abs();

    let (normal, depth) = if left <= right && left <= bottom && left <= top {
        (Vec2::new(-1.0, 0.0), radius + left)
    } else if right <= bottom && right <= top {
        (Vec2::new(1.0, 0.0), radius + right)
    } else if bottom <= top {
        (Vec2::new(0.0, -1.0), radius + bottom)
    } else {
        (Vec2::new(0.0, 1.0), radius + top)
    };

    *position += normal * depth;
    Some(normal)
}

pub fn draw_water_particles(mut gizmos: Gizmos, particles: Res<ParticleWorld>) {
    let color = Color::srgba(0.20, 0.56, 0.95, 0.9);
    for &position in particles.positions() {
        gizmos.circle_2d(position, PARTICLE_RADIUS_M, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::terrain::TerrainWorld;

    #[test]
    fn particles_fall_under_gravity() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        let mut particles = ParticleWorld::default();
        let before_avg_y = particles.positions().iter().map(|p| p.y).sum::<f32>()
            / particles.positions().len() as f32;

        for _ in 0..10 {
            particles.step_if_running(&terrain, true);
        }
        let after_avg_y = particles.positions().iter().map(|p| p.y).sum::<f32>()
            / particles.positions().len() as f32;

        assert!(after_avg_y < before_avg_y - 0.01);
    }

    #[test]
    fn particles_do_not_penetrate_frozen_terrain() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        let mut particles = ParticleWorld::default();
        particles.pos = vec![Vec2::new(0.1, 0.6)];
        particles.prev_pos = particles.pos.clone();
        particles.vel = vec![Vec2::new(0.0, -8.0)];
        particles.mass = vec![default_particle_mass()];
        particles.material = vec![ParticleMaterial::Water];
        particles.resize_work_buffers();

        for _ in 0..12 {
            particles.step_if_running(&terrain, true);
        }

        let p = particles.pos[0];
        let min_cell = world_to_cell(p - Vec2::splat(PARTICLE_RADIUS_M));
        let max_cell = world_to_cell(p + Vec2::splat(PARTICLE_RADIUS_M));

        for y in min_cell.y..=max_cell.y {
            for x in min_cell.x..=max_cell.x {
                if !matches!(
                    terrain.get_loaded_cell_or_empty(IVec2::new(x, y)),
                    TerrainCell::Solid { .. }
                ) {
                    continue;
                }

                let cell_min = IVec2::new(x, y).as_vec2() * CELL_SIZE_M;
                let cell_max = cell_min + Vec2::splat(CELL_SIZE_M);
                let closest = p.clamp(cell_min, cell_max);
                let dist_sq = (p - closest).length_squared();
                assert!(dist_sq >= PARTICLE_RADIUS_M * PARTICLE_RADIUS_M - 1e-4);
            }
        }
    }

    #[test]
    fn paused_simulation_keeps_particle_positions() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        let mut particles = ParticleWorld::default();
        let before = particles.pos.clone();

        particles.step_if_running(&terrain, false);

        assert_eq!(before, particles.pos);
    }

    #[test]
    fn reset_restores_initial_configuration() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        let mut particles = ParticleWorld::default();
        let initial_pos = particles.pos.clone();
        let initial_vel = particles.vel.clone();

        particles.step_if_running(&terrain, true);
        particles.reset_to_initial();

        assert_eq!(particles.pos, initial_pos);
        assert_eq!(particles.vel, initial_vel);
    }
}
