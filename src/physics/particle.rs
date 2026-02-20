use std::collections::{HashMap, HashSet};

use bevy::log::tracing;
use bevy::prelude::*;
use rayon::prelude::*;

use super::material::{
    ENABLE_GRANULAR_TO_SOLID_RECONVERSION, material_properties, particles_per_cell,
    terrain_break_collision_impulse_threshold, terrain_fracture_particle,
};
use super::object::{ObjectId, ObjectPhysicsField, ObjectWorld};
use super::terrain::{CELL_SIZE_M, TerrainCell, TerrainWorld, cell_to_world_center, world_to_cell};

pub const GRAVITY_MPS2: Vec2 = Vec2::new(0.0, -9.81);
pub const FIXED_DT: f32 = 1.0 / 60.0;
pub const SUBSTEPS: usize = 2;
pub const SOLVER_ITERS: usize = 6;
pub const SOLVER_MIN_ITERS: usize = 2;
pub const SOLVER_ERROR_TOLERANCE: f32 = 0.01;
pub const EPSILON_LAMBDA: f32 = 1e-6;
pub const TERRAIN_GHOST_DENSITY_SCALE: f32 = 1.0;
pub const TERRAIN_GHOST_DELTA_SCALE: f32 = 1.0;
pub const PARALLEL_PARTICLE_THRESHOLD: usize = 512;
pub const PARTICLE_CONTACT_PUSH_FACTOR: f32 = 0.5;

const INITIAL_WATER_ORIGIN_M: Vec2 = Vec2::new(-1.6, 2.0);
const INITIAL_WATER_COLS: usize = 50;
const INITIAL_WATER_ROWS: usize = 30;

pub use super::material::ParticleMaterial;
pub use super::material::{
    PARTICLE_RADIUS_M, PARTICLE_SPACING_M, PARTICLE_SPEED_LIMIT_MPS, REST_DENSITY,
    TERRAIN_BOUNDARY_RADIUS_M, WATER_KERNEL_RADIUS_M,
};

pub fn nominal_particle_draw_radius_m() -> f32 {
    (default_particle_mass() / (std::f32::consts::PI * REST_DENSITY)).sqrt()
}

#[derive(Debug, Clone)]
pub struct ParticleRemovalResult {
    pub removed_count: usize,
    pub old_to_new: Vec<Option<usize>>,
}

#[derive(Debug, Default)]
struct NeighborGrid {
    grid_min: IVec2,
    grid_size: UVec2,
    cell_starts: Vec<u32>,
    sorted_indices: Vec<usize>,
}

impl NeighborGrid {
    fn rebuild(&mut self, positions: &[Vec2]) {
        if positions.is_empty() {
            self.grid_min = IVec2::ZERO;
            self.grid_size = UVec2::ZERO;
            self.cell_starts.clear();
            self.sorted_indices.clear();
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

        let grid_w = (max_cell.x - min_cell.x + 1).max(1) as u32;
        let grid_h = (max_cell.y - min_cell.y + 1).max(1) as u32;
        let cell_count = (grid_w as usize) * (grid_h as usize);
        self.grid_min = min_cell;
        self.grid_size = UVec2::new(grid_w, grid_h);

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

    fn gather(&self, position: Vec2, out_neighbors: &mut Vec<usize>) {
        out_neighbors.clear();
        let center = particle_grid_cell(position);
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

#[derive(Default)]
struct ComputeDeltaThreadScratch {
    object_contacts: Vec<ObjectId>,
    reaction_impulses: HashMap<ObjectId, Vec2>,
}

impl ComputeDeltaThreadScratch {
    fn merge_from(&mut self, other: Self) {
        for (object_id, impulse) in other.reaction_impulses {
            *self
                .reaction_impulses
                .entry(object_id)
                .or_insert(Vec2::ZERO) += impulse;
        }
    }

    fn accumulate_impulse(&mut self, object_id: ObjectId, impulse: Vec2) {
        *self
            .reaction_impulses
            .entry(object_id)
            .or_insert(Vec2::ZERO) += impulse;
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
    neighbor_cache: Vec<Vec<usize>>,
    viscosity_work: Vec<Vec2>,
    object_peak_strain: HashMap<ObjectId, f32>,
    pending_object_fractures: HashSet<ObjectId>,
    pending_terrain_fractures: HashSet<IVec2>,
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
            material: vec![ParticleMaterial::WaterLiquid; initial_pos.len()],
            density: Vec::new(),
            lambda: Vec::new(),
            delta_pos: Vec::new(),
            initial_pos,
            initial_vel,
            neighbor_grid: NeighborGrid::default(),
            neighbor_cache: Vec::new(),
            viscosity_work: Vec::new(),
            object_peak_strain: HashMap::new(),
            pending_object_fractures: HashSet::new(),
            pending_terrain_fractures: HashSet::new(),
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
        self.material.fill(ParticleMaterial::WaterLiquid);
        self.resize_work_buffers();
    }

    pub fn particle_count(&self) -> usize {
        self.pos.len()
    }

    pub fn positions(&self) -> &[Vec2] {
        &self.pos
    }

    pub fn masses(&self) -> &[f32] {
        &self.mass
    }

    pub fn materials(&self) -> &[ParticleMaterial] {
        &self.material
    }

    pub fn restore_from_snapshot(
        &mut self,
        positions: Vec<Vec2>,
        velocities: Vec<Vec2>,
        materials: Vec<ParticleMaterial>,
    ) -> Result<(), String> {
        if positions.len() != velocities.len() || positions.len() != materials.len() {
            return Err("particle snapshot has mismatched array lengths".to_string());
        }

        self.pos = positions;
        self.prev_pos = self.pos.clone();
        self.vel = velocities;
        self.material = materials;
        self.mass = self
            .material
            .iter()
            .map(|&material| material_properties(material).mass)
            .collect();
        self.initial_pos = self.pos.clone();
        self.initial_vel = self.vel.clone();
        self.object_peak_strain.clear();
        self.pending_object_fractures.clear();
        self.pending_terrain_fractures.clear();
        self.resize_work_buffers();
        Ok(())
    }

    pub fn add_velocity_in_radius(
        &mut self,
        center: Vec2,
        radius: f32,
        velocity_delta: Vec2,
        velocity_limit_mps: f32,
    ) {
        let radius2 = radius * radius;
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) > radius2 {
                continue;
            }
            self.vel[i] = (self.vel[i] + velocity_delta).clamp_length_max(velocity_limit_mps);
        }
    }

    pub fn spawn_water_particles_along_segment(
        &mut self,
        from: Vec2,
        to: Vec2,
        spacing: f32,
        initial_velocity: Vec2,
        spacing_carry_m: &mut f32,
    ) -> usize {
        if spacing <= 0.0 {
            return 0;
        }

        let segment = to - from;
        let length = segment.length();
        if length < 1e-6 {
            return 0;
        }

        let direction = segment / length;
        let carry = (*spacing_carry_m).clamp(0.0, spacing);
        let total = carry + length;
        let spawn_count = (total / spacing).floor() as usize;
        if spawn_count == 0 {
            *spacing_carry_m = total;
            return 0;
        }

        let first_offset = spacing - carry;
        for i in 0..spawn_count {
            let offset = first_offset + i as f32 * spacing;
            let pos = from + direction * offset;
            self.push_water_particle(pos, initial_velocity);
        }

        *spacing_carry_m = total - spawn_count as f32 * spacing;
        self.resize_work_buffers();
        spawn_count
    }

    pub fn spawn_stone_particles_from_cells(
        &mut self,
        cells: &[IVec2],
        initial_velocity: Vec2,
    ) -> Vec<usize> {
        self.spawn_material_particles_from_cells(
            cells,
            ParticleMaterial::StoneSolid,
            initial_velocity,
        )
    }

    pub fn spawn_material_particles_from_cells(
        &mut self,
        cells: &[IVec2],
        material: ParticleMaterial,
        initial_velocity: Vec2,
    ) -> Vec<usize> {
        let mut indices = Vec::new();
        for &cell in cells {
            self.append_material_particles_in_cell(cell, material, initial_velocity, &mut indices);
        }
        self.resize_work_buffers();
        indices
    }

    pub fn apply_pending_terrain_fractures(&mut self, terrain: &mut TerrainWorld) -> bool {
        if self.pending_terrain_fractures.is_empty() {
            return false;
        }

        let fracture_cells: Vec<_> = self.pending_terrain_fractures.drain().collect();
        let mut changed = false;
        let mut appended = Vec::new();
        for cell in fracture_cells {
            let current = terrain.get_loaded_cell_or_empty(cell);
            let TerrainCell::Solid { material, .. } = current else {
                continue;
            };
            let Some(target_particle) = terrain_fracture_particle(material) else {
                continue;
            };
            if !terrain.set_cell(cell, TerrainCell::Empty) {
                continue;
            }
            changed = true;
            self.append_material_particles_in_cell(
                cell,
                target_particle,
                Vec2::ZERO,
                &mut appended,
            );
        }
        if !appended.is_empty() {
            self.resize_work_buffers();
        }
        changed
    }

    pub fn remove_particles_in_radius_with_map(
        &mut self,
        center: Vec2,
        radius: f32,
    ) -> ParticleRemovalResult {
        let old_count = self.particle_count();
        let radius2 = radius * radius;
        let mut keep = vec![true; old_count];
        let mut removed_count = 0;
        for (index, position) in self.pos.iter().enumerate() {
            if position.distance_squared(center) <= radius2 {
                keep[index] = false;
                removed_count += 1;
            }
        }

        if removed_count == 0 {
            return ParticleRemovalResult {
                removed_count,
                old_to_new: (0..old_count).map(Some).collect(),
            };
        }

        let new_count = old_count - removed_count;
        let mut old_to_new = vec![None; old_count];
        let mut next = 0usize;
        for old_index in 0..old_count {
            if !keep[old_index] {
                continue;
            }
            old_to_new[old_index] = Some(next);
            next += 1;
        }

        let mut new_pos = Vec::with_capacity(new_count);
        let mut new_prev_pos = Vec::with_capacity(new_count);
        let mut new_vel = Vec::with_capacity(new_count);
        let mut new_mass = Vec::with_capacity(new_count);
        let mut new_material = Vec::with_capacity(new_count);
        for old_index in 0..old_count {
            if !keep[old_index] {
                continue;
            }
            new_pos.push(self.pos[old_index]);
            new_prev_pos.push(self.prev_pos[old_index]);
            new_vel.push(self.vel[old_index]);
            new_mass.push(self.mass[old_index]);
            new_material.push(self.material[old_index]);
        }

        self.pos = new_pos;
        self.prev_pos = new_prev_pos;
        self.vel = new_vel;
        self.mass = new_mass;
        self.material = new_material;
        self.resize_work_buffers();

        ParticleRemovalResult {
            removed_count,
            old_to_new,
        }
    }

    pub fn step_if_running(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
        running: bool,
    ) {
        if running {
            self.step_substeps(terrain, object_field, object_world);
        }
    }

    pub fn step_substeps(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
    ) {
        let _span = tracing::info_span!("physics::step_substeps").entered();
        let dt_sub = FIXED_DT / SUBSTEPS as f32;
        for substep in 0..SUBSTEPS {
            let _substep_span = tracing::info_span!("physics::substep", substep).entered();
            self.step_single_substep(terrain, object_field, object_world, dt_sub);
        }
    }

    fn step_single_substep(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
        dt_sub: f32,
    ) {
        self.object_peak_strain.clear();
        {
            let _span = tracing::info_span!("physics::clear_reaction_impulses").entered();
            object_world.clear_reaction_impulses();
        }
        {
            let _span = tracing::info_span!("physics::predict_positions").entered();
            for i in 0..self.particle_count() {
                self.prev_pos[i] = self.pos[i];
                self.vel[i] += GRAVITY_MPS2 * dt_sub;
                self.pos[i] += self.vel[i] * dt_sub;
            }
        }

        {
            let _span = tracing::info_span!("physics::rebuild_neighbor_grid").entered();
            self.neighbor_grid.rebuild(&self.pos);
        }
        for iter in 0..SOLVER_ITERS {
            let _iter_span =
                tracing::info_span!("physics::solve_density_constraints", iter).entered();
            let max_density_error =
                self.solve_density_constraints(terrain, object_field, object_world, dt_sub);
            if iter + 1 >= SOLVER_MIN_ITERS && max_density_error <= SOLVER_ERROR_TOLERANCE {
                break;
            }
        }
        {
            let _span = tracing::info_span!("physics::shape_matching").entered();
            self.solve_shape_matching_constraints(object_world);
        }

        {
            let _span = tracing::info_span!("physics::update_velocity").entered();
            for i in 0..self.particle_count() {
                self.vel[i] = (self.pos[i] - self.prev_pos[i]) / dt_sub;
                self.vel[i] = self.vel[i].clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
            }
        }
        {
            let _span = tracing::info_span!("physics::contact_velocity_response").entered();
            self.apply_contact_velocity_response(terrain, object_field, object_world);
        }

        {
            let _span = tracing::info_span!("physics::xsph_viscosity").entered();
            self.apply_xsph_viscosity();
        }
        {
            let _span = tracing::info_span!("physics::apply_object_reaction").entered();
            self.apply_object_reaction_impulses(object_world);
        }
        {
            let _span = tracing::info_span!("physics::final_velocity_clamp").entered();
            for vel in &mut self.vel {
                *vel = vel.clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
            }
        }
        {
            let _span = tracing::info_span!("physics::fracture_detection").entered();
            self.detect_fracture_candidates(terrain, object_world);
            self.apply_object_fractures(object_world);
        }
    }

    fn solve_density_constraints(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
        dt_sub: f32,
    ) -> f32 {
        let _span = tracing::info_span!("physics::density_constraint_pass").entered();
        let h_ww = WATER_KERNEL_RADIUS_M;
        let inv_dt = 1.0 / dt_sub.max(1e-6);
        let use_parallel = self.particle_count() >= PARALLEL_PARTICLE_THRESHOLD;

        {
            let _span = tracing::info_span!("physics::rebuild_neighbor_cache").entered();
            self.rebuild_neighbor_cache(use_parallel);
        }

        let max_density_error = {
            let _span = tracing::info_span!("physics::density_lambda").entered();
            if use_parallel {
                self.solve_density_lambda_parallel(terrain, h_ww)
            } else {
                self.solve_density_lambda_sequential(terrain, h_ww)
            }
        };

        let reaction_impulses = {
            let _span = tracing::info_span!("physics::compute_delta").entered();
            if use_parallel {
                self.compute_delta_parallel(terrain, object_field, object_world, inv_dt, h_ww)
            } else {
                self.compute_delta_sequential(terrain, object_field, object_world, inv_dt, h_ww)
            }
        };
        for (object_id, impulse) in reaction_impulses {
            object_world.accumulate_reaction_impulse(object_id, impulse);
        }

        {
            let _span = tracing::info_span!("physics::apply_delta").entered();
            for i in 0..self.particle_count() {
                self.pos[i] += self.delta_pos[i];
            }
        }
        max_density_error
    }

    fn rebuild_neighbor_cache(&mut self, use_parallel: bool) {
        let count = self.particle_count();
        self.neighbor_cache.resize_with(count, Vec::new);
        if use_parallel {
            let positions = &self.pos;
            let grid = &self.neighbor_grid;
            self.neighbor_cache
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, neighbors)| grid.gather(positions[i], neighbors));
        } else {
            for i in 0..count {
                self.neighbor_grid
                    .gather(self.pos[i], &mut self.neighbor_cache[i]);
            }
        }
    }

    fn solve_density_lambda_sequential(&mut self, terrain: &TerrainWorld, h_ww: f32) -> f32 {
        let mut max_density_error = 0.0f32;
        for i in 0..self.particle_count() {
            if !is_water_particle(self.material[i]) {
                self.density[i] = REST_DENSITY;
                self.lambda[i] = 0.0;
                continue;
            }
            let neighbors = &self.neighbor_cache[i];
            let mut rho = 0.0;
            let mut grad_i = Vec2::ZERO;
            let mut grad_sum_sq = 0.0;

            for &j in neighbors {
                if !is_water_particle(self.material[j]) {
                    continue;
                }
                let r = self.pos[i] - self.pos[j];
                let r2 = r.length_squared();
                if r2 >= h_ww * h_ww {
                    continue;
                }

                rho += self.mass[j] * kernel_poly6(r2, h_ww);

                if i == j {
                    continue;
                }
                let grad = (self.mass[j] / REST_DENSITY) * kernel_spiky_grad(r, h_ww);
                grad_sum_sq += grad.length_squared();
                grad_i += grad;
            }

            if let Some(ghost_r) = terrain_ghost_neighbor_vector(self.pos[i], terrain, h_ww) {
                let ghost_mass = self.mass[i];
                let ghost_r2 = ghost_r.length_squared();
                rho += ghost_mass * kernel_poly6(ghost_r2, h_ww) * TERRAIN_GHOST_DENSITY_SCALE;
                let ghost_grad = (ghost_mass / REST_DENSITY)
                    * kernel_spiky_grad(ghost_r, h_ww)
                    * TERRAIN_GHOST_DENSITY_SCALE;
                grad_sum_sq += ghost_grad.length_squared();
                grad_i += ghost_grad;
            }

            grad_sum_sq += grad_i.length_squared();
            let c_i = rho / REST_DENSITY - 1.0;

            self.density[i] = rho;
            self.lambda[i] = -c_i / (grad_sum_sq + EPSILON_LAMBDA);
            max_density_error = max_density_error.max(c_i.abs());
        }
        max_density_error
    }

    fn solve_density_lambda_parallel(&mut self, terrain: &TerrainWorld, h_ww: f32) -> f32 {
        let positions = &self.pos;
        let masses = &self.mass;
        let materials = &self.material;
        let neighbor_cache = &self.neighbor_cache;

        self.density
            .par_iter_mut()
            .zip(self.lambda.par_iter_mut())
            .enumerate()
            .map(|(i, (density_i, lambda_i))| {
                if !is_water_particle(materials[i]) {
                    *density_i = REST_DENSITY;
                    *lambda_i = 0.0;
                    return 0.0;
                }

                let mut rho = 0.0;
                let mut grad_i = Vec2::ZERO;
                let mut grad_sum_sq = 0.0;
                for &j in &neighbor_cache[i] {
                    if !is_water_particle(materials[j]) {
                        continue;
                    }
                    let r = positions[i] - positions[j];
                    let r2 = r.length_squared();
                    if r2 >= h_ww * h_ww {
                        continue;
                    }

                    rho += masses[j] * kernel_poly6(r2, h_ww);
                    if i == j {
                        continue;
                    }
                    let grad = (masses[j] / REST_DENSITY) * kernel_spiky_grad(r, h_ww);
                    grad_sum_sq += grad.length_squared();
                    grad_i += grad;
                }

                if let Some(ghost_r) = terrain_ghost_neighbor_vector(positions[i], terrain, h_ww) {
                    let ghost_mass = masses[i];
                    let ghost_r2 = ghost_r.length_squared();
                    rho += ghost_mass * kernel_poly6(ghost_r2, h_ww) * TERRAIN_GHOST_DENSITY_SCALE;
                    let ghost_grad = (ghost_mass / REST_DENSITY)
                        * kernel_spiky_grad(ghost_r, h_ww)
                        * TERRAIN_GHOST_DENSITY_SCALE;
                    grad_sum_sq += ghost_grad.length_squared();
                    grad_i += ghost_grad;
                }

                grad_sum_sq += grad_i.length_squared();
                let c_i = rho / REST_DENSITY - 1.0;
                *density_i = rho;
                *lambda_i = -c_i / (grad_sum_sq + EPSILON_LAMBDA);
                c_i.abs()
            })
            .reduce(|| 0.0, f32::max)
    }

    fn compute_delta_sequential(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
        inv_dt: f32,
        h_ww: f32,
    ) -> HashMap<ObjectId, Vec2> {
        let mut reaction_impulses = HashMap::new();
        let mut object_contacts = Vec::new();
        for i in 0..self.particle_count() {
            let material = self.material[i];
            let props = material_properties(material);
            let is_water = is_water_particle(material);
            let neighbors = &self.neighbor_cache[i];

            let mut delta = Vec2::ZERO;
            let mut boundary_push = Vec2::ZERO;
            let mut particle_contact_push = Vec2::ZERO;
            if is_water {
                for &j in neighbors {
                    if i == j || !is_water_particle(self.material[j]) {
                        continue;
                    }
                    let r = self.pos[i] - self.pos[j];
                    if r.length_squared() >= h_ww * h_ww {
                        continue;
                    }
                    delta += (self.lambda[i] + self.lambda[j]) * kernel_spiky_grad(r, h_ww);
                }
                if let Some(ghost_r) = terrain_ghost_neighbor_vector(self.pos[i], terrain, h_ww) {
                    delta += self.lambda[i]
                        * kernel_spiky_grad(ghost_r, h_ww)
                        * TERRAIN_GHOST_DELTA_SCALE;
                }
            }
            for &j in neighbors {
                if i == j {
                    continue;
                }
                let neighbor_material = self.material[j];
                if is_water && is_water_particle(neighbor_material) {
                    continue;
                }
                let neighbor_props = material_properties(neighbor_material);
                let r = self.pos[i] - self.pos[j];
                let dist2 = r.length_squared();
                let contact_radius = props.radius_m + neighbor_props.radius_m;
                if dist2 <= 1e-12 || dist2 >= contact_radius * contact_radius {
                    continue;
                }
                let dist = dist2.sqrt();
                let normal = r / dist;
                let penetration = contact_radius - dist;
                let stiffness = 0.5
                    * (props.object_repulsion_stiffness
                        + neighbor_props.object_repulsion_stiffness);
                particle_contact_push +=
                    normal * penetration * stiffness * PARTICLE_CONTACT_PUSH_FACTOR;
            }

            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            {
                let penetration = props.terrain_push_radius_m - signed_distance;
                if penetration > 0.0 {
                    boundary_push += normal * penetration * props.terrain_repulsion_stiffness;
                }
            }

            let mut object_push = Vec2::ZERO;
            let self_object = object_world.object_of_particle(i);
            if is_water || self_object.is_some() {
                object_field.gather_candidate_object_ids(self.pos[i], &mut object_contacts);

                for &object_id in &object_contacts {
                    if self_object == Some(object_id) {
                        continue;
                    }
                    let Some(sample) = object_world.evaluate_object_sdf(object_id, self.pos[i])
                    else {
                        continue;
                    };
                    let penetration = props.object_push_radius_m - sample.distance_m;
                    if penetration <= 0.0 {
                        continue;
                    }
                    let push = sample.normal_world * penetration * props.object_repulsion_stiffness;
                    object_push += push;
                    let reaction_impulse = -(self.mass[i] * push) * inv_dt;
                    *reaction_impulses.entry(object_id).or_insert(Vec2::ZERO) += reaction_impulse;
                }
            }

            let max_push = props.object_push_radius_m;
            object_push = object_push.clamp_length_max(max_push);
            self.delta_pos[i] = if is_water {
                delta / REST_DENSITY + boundary_push + object_push + particle_contact_push
            } else {
                boundary_push + object_push + particle_contact_push
            };
        }
        reaction_impulses
    }

    fn compute_delta_parallel(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
        inv_dt: f32,
        h_ww: f32,
    ) -> HashMap<ObjectId, Vec2> {
        let positions = &self.pos;
        let masses = &self.mass;
        let materials = &self.material;
        let lambdas = &self.lambda;
        let neighbor_cache = &self.neighbor_cache;

        let reduced = self
            .delta_pos
            .par_iter_mut()
            .enumerate()
            .fold(
                ComputeDeltaThreadScratch::default,
                |mut scratch, (i, delta_pos_i)| {
                    let material = materials[i];
                    let props = material_properties(material);
                    let is_water = is_water_particle(material);
                    let neighbors = &neighbor_cache[i];

                    let mut delta = Vec2::ZERO;
                    let mut boundary_push = Vec2::ZERO;
                    let mut particle_contact_push = Vec2::ZERO;
                    if is_water {
                        for &j in neighbors {
                            if i == j || !is_water_particle(materials[j]) {
                                continue;
                            }
                            let r = positions[i] - positions[j];
                            if r.length_squared() >= h_ww * h_ww {
                                continue;
                            }
                            delta += (lambdas[i] + lambdas[j]) * kernel_spiky_grad(r, h_ww);
                        }
                        if let Some(ghost_r) =
                            terrain_ghost_neighbor_vector(positions[i], terrain, h_ww)
                        {
                            delta += lambdas[i]
                                * kernel_spiky_grad(ghost_r, h_ww)
                                * TERRAIN_GHOST_DELTA_SCALE;
                        }
                    }
                    for &j in neighbors {
                        if i == j {
                            continue;
                        }
                        let neighbor_material = materials[j];
                        if is_water && is_water_particle(neighbor_material) {
                            continue;
                        }
                        let neighbor_props = material_properties(neighbor_material);
                        let r = positions[i] - positions[j];
                        let dist2 = r.length_squared();
                        let contact_radius = props.radius_m + neighbor_props.radius_m;
                        if dist2 <= 1e-12 || dist2 >= contact_radius * contact_radius {
                            continue;
                        }
                        let dist = dist2.sqrt();
                        let normal = r / dist;
                        let penetration = contact_radius - dist;
                        let stiffness = 0.5
                            * (props.object_repulsion_stiffness
                                + neighbor_props.object_repulsion_stiffness);
                        particle_contact_push +=
                            normal * penetration * stiffness * PARTICLE_CONTACT_PUSH_FACTOR;
                    }

                    if let Some((signed_distance, normal)) =
                        terrain.sample_signed_distance_and_normal(positions[i])
                    {
                        let penetration = props.terrain_push_radius_m - signed_distance;
                        if penetration > 0.0 {
                            boundary_push +=
                                normal * penetration * props.terrain_repulsion_stiffness;
                        }
                    }

                    let mut object_push = Vec2::ZERO;
                    let self_object = object_world.object_of_particle(i);
                    if is_water || self_object.is_some() {
                        object_field.gather_candidate_object_ids(
                            positions[i],
                            &mut scratch.object_contacts,
                        );

                        for contact_index in 0..scratch.object_contacts.len() {
                            let object_id = scratch.object_contacts[contact_index];
                            if self_object == Some(object_id) {
                                continue;
                            }
                            let Some(sample) =
                                object_world.evaluate_object_sdf(object_id, positions[i])
                            else {
                                continue;
                            };
                            let penetration = props.object_push_radius_m - sample.distance_m;
                            if penetration <= 0.0 {
                                continue;
                            }
                            let push = sample.normal_world
                                * penetration
                                * props.object_repulsion_stiffness;
                            object_push += push;
                            let reaction_impulse = -(masses[i] * push) * inv_dt;
                            scratch.accumulate_impulse(object_id, reaction_impulse);
                        }
                    }

                    let max_push = props.object_push_radius_m;
                    object_push = object_push.clamp_length_max(max_push);
                    *delta_pos_i = if is_water {
                        delta / REST_DENSITY + boundary_push + object_push + particle_contact_push
                    } else {
                        boundary_push + object_push + particle_contact_push
                    };

                    scratch
                },
            )
            .reduce(ComputeDeltaThreadScratch::default, |mut a, b| {
                a.merge_from(b);
                a
            });

        reduced.reaction_impulses
    }

    fn apply_contact_velocity_response(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
    ) {
        let mut object_contacts = Vec::new();
        for i in 0..self.particle_count() {
            let material = self.material[i];
            let props = material_properties(material);
            if !props.apply_contact_velocity_response {
                continue;
            }

            let mut normal_sum = Vec2::ZERO;
            let mut contact_count = 0usize;

            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            {
                let penetration = props.terrain_push_radius_m - signed_distance;
                if penetration > 0.0 {
                    normal_sum += normal;
                    contact_count += 1;
                }
            }

            let self_object = object_world.object_of_particle(i);
            object_field.gather_candidate_object_ids(self.pos[i], &mut object_contacts);
            for &object_id in &object_contacts {
                if self_object == Some(object_id) {
                    continue;
                }
                let Some(sample) = object_world.evaluate_object_sdf(object_id, self.pos[i]) else {
                    continue;
                };
                let penetration = props.object_push_radius_m - sample.distance_m;
                if penetration <= 0.0 {
                    continue;
                }
                normal_sum += sample.normal_world;
                contact_count += 1;
            }

            if contact_count == 0 {
                continue;
            }
            let contact_normal = normal_sum.normalize_or_zero();
            if contact_normal == Vec2::ZERO {
                continue;
            }

            let normal_speed = self.vel[i].dot(contact_normal);
            let tangential_velocity = self.vel[i] - contact_normal * normal_speed;
            let normal_contact_speed = normal_speed.max(0.0);
            let tangential_speed = tangential_velocity.length();
            let adjusted_tangent = if tangential_speed <= 1e-6 {
                Vec2::ZERO
            } else if tangential_speed <= props.friction_static * normal_contact_speed {
                Vec2::ZERO
            } else {
                let reduced_speed =
                    (tangential_speed - props.friction_dynamic * normal_contact_speed).max(0.0);
                tangential_velocity * (reduced_speed / tangential_speed)
            };
            let adjusted_normal_speed = if normal_contact_speed > 0.0 {
                normal_speed * props.contact_restitution.clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.vel[i] = adjusted_tangent + contact_normal * adjusted_normal_speed;
        }
    }

    fn apply_xsph_viscosity(&mut self) {
        let _span = tracing::info_span!("physics::xsph_pass").entered();
        let mut neighbors = Vec::new();
        self.viscosity_work.clone_from(&self.vel);

        for i in 0..self.particle_count() {
            if !is_water_particle(self.material[i]) {
                self.viscosity_work[i] = self.vel[i];
                continue;
            }
            self.neighbor_grid.gather(self.pos[i], &mut neighbors);

            let mut correction = Vec2::ZERO;
            for &j in &neighbors {
                if i == j || !is_water_particle(self.material[j]) {
                    continue;
                }
                let r = self.pos[i] - self.pos[j];
                let w = kernel_poly6(r.length_squared(), WATER_KERNEL_RADIUS_M);
                correction += (self.vel[j] - self.vel[i]) * w;
            }
            let viscosity = material_properties(self.material[i]).xsph_viscosity;
            self.viscosity_work[i] = self.vel[i] + viscosity * correction;
        }

        self.vel.clone_from(&self.viscosity_work);
    }

    fn apply_object_reaction_impulses(&mut self, object_world: &ObjectWorld) {
        let _span = tracing::info_span!("physics::object_reaction_pass").entered();
        for object in object_world.objects() {
            let impulse = object_world.reaction_impulse_of(object.id);
            if impulse.length_squared() <= 1e-12 {
                continue;
            }

            let mass_sum = object
                .particle_indices
                .iter()
                .filter_map(|&index| self.mass.get(index).copied())
                .sum::<f32>()
                .max(1e-6);
            let dv = impulse / mass_sum;
            for &index in &object.particle_indices {
                if index >= self.particle_count() {
                    continue;
                }
                self.vel[index] += dv;
            }
        }
    }

    fn detect_fracture_candidates(&mut self, terrain: &TerrainWorld, object_world: &ObjectWorld) {
        debug_assert!(
            !ENABLE_GRANULAR_TO_SOLID_RECONVERSION,
            "v4 keeps granular->solid reconversion disabled"
        );
        self.pending_object_fractures.clear();
        self.pending_terrain_fractures.clear();

        for object in object_world.objects() {
            let Some(&particle_index) = object.particle_indices.first() else {
                continue;
            };
            if particle_index >= self.particle_count() {
                continue;
            }
            let source_material = self.material[particle_index];
            let props = material_properties(source_material);
            if props.fracture_to.is_none() {
                continue;
            }

            let collision_impulse = object_world.reaction_impulse_of(object.id).length();
            let strain = self
                .object_peak_strain
                .get(&object.id)
                .copied()
                .unwrap_or(0.0);
            if collision_impulse >= props.break_collision_impulse_threshold
                || strain >= props.break_strain_threshold
            {
                self.pending_object_fractures.insert(object.id);
            }
        }

        let mut terrain_impulse = HashMap::<IVec2, f32>::new();
        for i in 0..self.particle_count() {
            let particle_material = self.material[i];
            let props = material_properties(particle_material);
            let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            else {
                continue;
            };
            let penetration = props.terrain_push_radius_m - signed_distance;
            if penetration <= 0.0 {
                continue;
            }
            let impulse = self.mass[i] * self.vel[i].dot(normal).abs();
            if impulse <= 1e-6 {
                continue;
            }
            let Some(cell) =
                resolve_terrain_contact_cell(self.pos[i], signed_distance, normal, terrain)
            else {
                continue;
            };
            let entry = terrain_impulse.entry(cell).or_insert(0.0);
            *entry = (*entry).max(impulse);
        }

        for (cell, impulse) in terrain_impulse {
            let TerrainCell::Solid { material, .. } = terrain.get_loaded_cell_or_empty(cell) else {
                continue;
            };
            if material.form() != super::material::MaterialForm::Solid {
                continue;
            }
            if terrain_fracture_particle(material).is_none() {
                continue;
            }
            let threshold = terrain_break_collision_impulse_threshold(material);
            if impulse >= threshold {
                self.pending_terrain_fractures.insert(cell);
            }
        }
    }

    fn apply_object_fractures(&mut self, object_world: &mut ObjectWorld) {
        if self.pending_object_fractures.is_empty() {
            return;
        }

        let mut removed = HashSet::new();
        for object in object_world.objects() {
            if !self.pending_object_fractures.contains(&object.id) {
                continue;
            }

            let Some(&seed_index) = object.particle_indices.first() else {
                continue;
            };
            if seed_index >= self.particle_count() {
                continue;
            }
            let source_material = self.material[seed_index];
            let Some(target_material) = material_properties(source_material).fracture_to else {
                continue;
            };
            let target_mass = material_properties(target_material).mass;
            for &index in &object.particle_indices {
                if index >= self.particle_count() {
                    continue;
                }
                self.material[index] = target_material;
                self.mass[index] = target_mass;
            }
            removed.insert(object.id);
        }

        object_world.remove_objects_by_ids(&removed);
        self.pending_object_fractures.clear();
    }

    fn solve_shape_matching_constraints(&mut self, object_world: &mut ObjectWorld) {
        let _span = tracing::info_span!("physics::shape_matching_pass").entered();
        for object in object_world.objects_mut() {
            if object.particle_indices.is_empty() || object.rest_local.is_empty() {
                continue;
            }
            let alpha = object.shape_stiffness_alpha.clamp(0.0, 1.0);
            if alpha <= 0.0 {
                continue;
            }

            let mut object_peak_strain = 0.0f32;
            let shape_iters = object.shape_iters.max(1);
            for _ in 0..shape_iters {
                let mut mass_sum = 0.0;
                let mut com = Vec2::ZERO;
                for &index in &object.particle_indices {
                    if index >= self.particle_count() {
                        continue;
                    }
                    let mass = self.mass[index];
                    mass_sum += mass;
                    com += self.pos[index] * mass;
                }
                if mass_sum <= 1e-6 {
                    continue;
                }
                com /= mass_sum;
                object.mass_sum = mass_sum;

                let mut a00 = 0.0;
                let mut a01 = 0.0;
                let mut a10 = 0.0;
                let mut a11 = 0.0;
                for (slot, &index) in object.particle_indices.iter().enumerate() {
                    if index >= self.particle_count() || slot >= object.rest_local.len() {
                        continue;
                    }
                    let mass = self.mass[index];
                    let p = self.pos[index] - com;
                    let q = object.rest_local[slot];
                    a00 += mass * p.x * q.x;
                    a01 += mass * p.x * q.y;
                    a10 += mass * p.y * q.x;
                    a11 += mass * p.y * q.y;
                }

                let theta = (a10 - a01).atan2(a00 + a11);
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                for (slot, &index) in object.particle_indices.iter().enumerate() {
                    if index >= self.particle_count() || slot >= object.rest_local.len() {
                        continue;
                    }
                    let q = object.rest_local[slot];
                    let rotated = Vec2::new(cos_t * q.x - sin_t * q.y, sin_t * q.x + cos_t * q.y);
                    let goal = com + rotated;
                    let current = self.pos[index];
                    let strain = (goal - current).length() / CELL_SIZE_M.max(1e-6);
                    object_peak_strain = object_peak_strain.max(strain);
                    self.pos[index] = current + (goal - current) * alpha;
                }
            }
            self.object_peak_strain
                .insert(object.id, object_peak_strain);
        }
    }

    fn push_water_particle(&mut self, position: Vec2, velocity: Vec2) {
        self.pos.push(position);
        self.prev_pos.push(position);
        self.vel.push(velocity);
        self.mass.push(default_particle_mass());
        self.material.push(ParticleMaterial::WaterLiquid);
    }

    fn append_material_particles_in_cell(
        &mut self,
        cell: IVec2,
        material: ParticleMaterial,
        velocity: Vec2,
        out_indices: &mut Vec<usize>,
    ) {
        let count = particles_per_cell(material);
        let axis = particle_grid_axis(count);
        let axis_f = axis as f32;
        let spacing = CELL_SIZE_M / axis_f.max(1.0);
        let cell_min = cell_to_world_center(cell) - Vec2::splat(CELL_SIZE_M * 0.5);
        let particle_mass = material_properties(material).mass;

        for y in 0..axis {
            for x in 0..axis {
                let local = Vec2::new((x as f32 + 0.5) * spacing, (y as f32 + 0.5) * spacing);
                let position = cell_min + local;
                let index = self.pos.len();
                self.pos.push(position);
                self.prev_pos.push(position);
                self.vel.push(velocity);
                self.mass.push(particle_mass);
                self.material.push(material);
                out_indices.push(index);
            }
        }
    }

    fn resize_work_buffers(&mut self) {
        let count = self.particle_count();
        self.prev_pos.resize(count, Vec2::ZERO);
        self.vel.resize(count, Vec2::ZERO);
        self.mass.resize(count, default_particle_mass());
        self.material.resize(count, ParticleMaterial::WaterLiquid);
        self.density.resize(count, 0.0);
        self.lambda.resize(count, 0.0);
        self.delta_pos.resize(count, Vec2::ZERO);
        self.neighbor_cache.resize_with(count, Vec::new);
        self.viscosity_work.resize(count, Vec2::ZERO);
    }
}

fn particle_grid_axis(particles_per_cell: u32) -> u32 {
    if particles_per_cell <= 1 {
        return 1;
    }
    let mut axis = 1u32;
    while axis.saturating_mul(axis) < particles_per_cell {
        axis += 1;
    }
    if axis.saturating_mul(axis) == particles_per_cell {
        axis
    } else {
        1
    }
}

fn resolve_terrain_contact_cell(
    position: Vec2,
    signed_distance: f32,
    normal: Vec2,
    terrain: &TerrainWorld,
) -> Option<IVec2> {
    let boundary_point = position - normal * signed_distance.max(0.0);
    let probe = boundary_point - normal * (CELL_SIZE_M * 0.25);
    let center_cell = world_to_cell(probe);

    let mut best: Option<(f32, IVec2)> = None;
    for y in (center_cell.y - 1)..=(center_cell.y + 1) {
        for x in (center_cell.x - 1)..=(center_cell.x + 1) {
            let cell = IVec2::new(x, y);
            if !matches!(
                terrain.get_loaded_cell_or_empty(cell),
                TerrainCell::Solid { .. }
            ) {
                continue;
            }
            let d2 = distance_sq_to_cell_aabb(boundary_point, cell);
            let replace = match best {
                Some((best_d2, _)) => d2 < best_d2,
                None => true,
            };
            if replace {
                best = Some((d2, cell));
            }
        }
    }
    best.map(|(_, cell)| cell)
}

fn distance_sq_to_cell_aabb(point: Vec2, cell: IVec2) -> f32 {
    let center = cell_to_world_center(cell);
    let half = CELL_SIZE_M * 0.5;
    let min = center - Vec2::splat(half);
    let max = center + Vec2::splat(half);
    let closest = Vec2::new(point.x.clamp(min.x, max.x), point.y.clamp(min.y, max.y));
    point.distance_squared(closest)
}

fn generate_initial_water_positions() -> Vec<Vec2> {
    let mut positions = Vec::with_capacity(INITIAL_WATER_COLS * INITIAL_WATER_ROWS);
    for row in 0..INITIAL_WATER_ROWS {
        for col in 0..INITIAL_WATER_COLS {
            positions.push(
                INITIAL_WATER_ORIGIN_M
                    + Vec2::new(col as f32, row as f32) * PARTICLE_SPACING_M
                    + Vec2::splat(material_properties(ParticleMaterial::WaterLiquid).radius_m),
            );
        }
    }
    positions
}

fn default_particle_mass() -> f32 {
    material_properties(ParticleMaterial::WaterLiquid).mass
}

fn particle_grid_cell(position: Vec2) -> IVec2 {
    IVec2::new(
        (position.x / WATER_KERNEL_RADIUS_M).floor() as i32,
        (position.y / WATER_KERNEL_RADIUS_M).floor() as i32,
    )
}

fn kernel_poly6(r2: f32, support_radius: f32) -> f32 {
    let h2 = support_radius * support_radius;
    if !(0.0..h2).contains(&r2) {
        return 0.0;
    }
    let x = h2 - r2;
    let coeff = 4.0 / (std::f32::consts::PI * h2.powi(4));
    coeff * x.powi(3)
}

fn kernel_spiky_grad(r: Vec2, support_radius: f32) -> Vec2 {
    let len = r.length();
    if !(1e-6..support_radius).contains(&len) {
        return Vec2::ZERO;
    }
    let coeff = -30.0 / (std::f32::consts::PI * support_radius.powi(5));
    r * (coeff * (support_radius - len).powi(2) / len)
}

fn is_water_particle(material: ParticleMaterial) -> bool {
    matches!(material, ParticleMaterial::WaterLiquid)
}

fn terrain_ghost_neighbor_vector(
    position: Vec2,
    terrain: &TerrainWorld,
    support_radius: f32,
) -> Option<Vec2> {
    let (signed_distance, normal) = terrain.sample_signed_distance_and_normal(position)?;
    let d = signed_distance.max(0.0);
    if d >= support_radius {
        return None;
    }
    Some(normal * (2.0 * d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::object::{OBJECT_SHAPE_ITERS, OBJECT_SHAPE_STIFFNESS_ALPHA};
    use crate::physics::terrain::{CHUNK_WORLD_SIZE_M, TerrainWorld, WORLD_MIN_CHUNK_X};

    #[test]
    fn particles_fall_under_gravity() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();
        let before_avg_y = particles.positions().iter().map(|p| p.y).sum::<f32>()
            / particles.positions().len() as f32;

        for _ in 0..10 {
            particles.step_if_running(&terrain, &object_field, &mut object_world, true);
        }
        let after_avg_y = particles.positions().iter().map(|p| p.y).sum::<f32>()
            / particles.positions().len() as f32;

        assert!(after_avg_y < before_avg_y - 0.01);
    }

    #[test]
    fn long_run_stability_has_no_nan() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();

        for _ in 0..300 {
            particles.step_if_running(&terrain, &object_field, &mut object_world, true);
        }

        assert!(particles.pos.iter().all(|p| p.is_finite()));
        assert!(particles.vel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn static_terrain_boundary_push_affects_motion() {
        let mut terrain_static = TerrainWorld::default();
        terrain_static.reset_fixed_world();
        terrain_static.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let terrain_empty = TerrainWorld::default();

        let mut with_terrain = ParticleWorld::default();
        with_terrain.pos = vec![Vec2::new(0.1, 0.6)];
        with_terrain.prev_pos = with_terrain.pos.clone();
        with_terrain.vel = vec![Vec2::new(0.0, -8.0)];
        with_terrain.mass = vec![default_particle_mass()];
        with_terrain.material = vec![ParticleMaterial::WaterLiquid];
        with_terrain.resize_work_buffers();

        let mut without_terrain = ParticleWorld::default();
        without_terrain.pos = with_terrain.pos.clone();
        without_terrain.prev_pos = without_terrain.pos.clone();
        without_terrain.vel = with_terrain.vel.clone();
        without_terrain.mass = with_terrain.mass.clone();
        without_terrain.material = with_terrain.material.clone();
        without_terrain.resize_work_buffers();
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();

        for _ in 0..12 {
            with_terrain.step_if_running(&terrain_static, &object_field, &mut object_world, true);
            without_terrain.step_if_running(&terrain_empty, &object_field, &mut object_world, true);
        }

        let y_with = with_terrain.pos[0].y;
        let y_without = without_terrain.pos[0].y;
        assert!(
            y_with > y_without + 0.2,
            "boundary push did not keep particle away from terrain (with={y_with}, without={y_without})"
        );
    }

    #[test]
    fn particles_do_not_penetrate_frozen_terrain() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        particles.pos = vec![Vec2::new(0.1, 0.6)];
        particles.prev_pos = particles.pos.clone();
        particles.vel = vec![Vec2::new(0.0, -8.0)];
        particles.mass = vec![default_particle_mass()];
        particles.material = vec![ParticleMaterial::WaterLiquid];
        particles.resize_work_buffers();
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();

        for _ in 0..12 {
            particles.step_if_running(&terrain, &object_field, &mut object_world, true);
        }

        let p = particles.pos[0];
        assert!(
            p.y > -0.25,
            "particle tunneled too deep into terrain: y={}",
            p.y
        );
    }

    #[test]
    fn paused_simulation_keeps_particle_positions() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();
        let before = particles.pos.clone();

        particles.step_if_running(&terrain, &object_field, &mut object_world, false);

        assert_eq!(before, particles.pos);
    }

    #[test]
    fn reset_restores_initial_configuration() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();
        let initial_pos = particles.pos.clone();
        let initial_vel = particles.vel.clone();

        particles.step_if_running(&terrain, &object_field, &mut object_world, true);
        particles.reset_to_initial();

        assert_eq!(particles.pos, initial_pos);
        assert_eq!(particles.vel, initial_vel);
    }

    #[test]
    fn boundary_overcrowding_remains_stable() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let left_wall_x = WORLD_MIN_CHUNK_X as f32 * CHUNK_WORLD_SIZE_M;

        for (i, pos) in particles.pos.iter_mut().enumerate() {
            let row = (i / 20) as f32;
            let col = (i % 20) as f32;
            *pos = Vec2::new(left_wall_x + 0.25 + col * 0.03, 0.4 + row * 0.03);
        }
        particles.prev_pos.clone_from(&particles.pos);
        particles.vel.fill(Vec2::ZERO);
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();

        for _ in 0..30 {
            particles.step_if_running(&terrain, &object_field, &mut object_world, true);
        }

        for pos in &particles.pos {
            assert!(pos.is_finite());
        }
        for vel in &particles.vel {
            assert!(vel.is_finite());
            assert!(vel.length() <= PARTICLE_SPEED_LIMIT_MPS + 1e-4);
        }
    }

    #[test]
    fn shape_matching_keeps_stone_object_cohesive() {
        let terrain = TerrainWorld::default();
        let mut particles = ParticleWorld::default();
        particles.pos.clear();
        particles.prev_pos.clear();
        particles.vel.clear();
        particles.mass.clear();
        particles.material.clear();
        particles.density.clear();
        particles.lambda.clear();
        particles.delta_pos.clear();
        particles.viscosity_work.clear();
        particles.resize_work_buffers();

        let cells = [IVec2::new(0, 6), IVec2::new(1, 6), IVec2::new(2, 6)];
        let indices = particles.spawn_stone_particles_from_cells(&cells, Vec2::ZERO);
        let initial_span = particles.pos[indices[2]].x - particles.pos[indices[0]].x;

        let mut objects = ObjectWorld::default();
        let _ = objects.create_object(
            indices,
            particles.positions(),
            particles.masses(),
            OBJECT_SHAPE_STIFFNESS_ALPHA,
            OBJECT_SHAPE_ITERS,
        );
        let object_field = ObjectPhysicsField::default();

        for _ in 0..30 {
            particles.step_if_running(&terrain, &object_field, &mut objects, true);
        }

        let span = particles.pos[2].x - particles.pos[0].x;
        assert!(
            (span - initial_span).abs() < 0.05,
            "shape matching drifted too much: initial={initial_span}, current={span}"
        );
    }

    #[test]
    fn water_object_contact_pushes_object_motion() {
        let terrain = TerrainWorld::default();
        let stone_cell = IVec2::new(0, 6);
        let water_pos = cell_to_world_center(stone_cell);

        let mut with_water = ParticleWorld::default();
        with_water.pos.clear();
        with_water.prev_pos.clear();
        with_water.vel.clear();
        with_water.mass.clear();
        with_water.material.clear();
        with_water.density.clear();
        with_water.lambda.clear();
        with_water.delta_pos.clear();
        with_water.viscosity_work.clear();
        with_water.resize_work_buffers();
        let stone_indices = with_water.spawn_stone_particles_from_cells(&[stone_cell], Vec2::ZERO);
        with_water.pos.push(water_pos);
        with_water.prev_pos.push(water_pos);
        with_water.vel.push(Vec2::ZERO);
        with_water.mass.push(default_particle_mass());
        with_water.material.push(ParticleMaterial::WaterLiquid);
        with_water.resize_work_buffers();

        let mut with_water_objects = ObjectWorld::default();
        let _ = with_water_objects
            .create_object(
                stone_indices,
                with_water.positions(),
                with_water.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            )
            .expect("object should be created");
        let mut with_water_field = ObjectPhysicsField::default();
        with_water_objects.update_physics_field(
            with_water.positions(),
            with_water.masses(),
            &mut with_water_field,
        );
        with_water.step_if_running(&terrain, &with_water_field, &mut with_water_objects, true);

        let mut without_water = ParticleWorld::default();
        without_water.pos.clear();
        without_water.prev_pos.clear();
        without_water.vel.clear();
        without_water.mass.clear();
        without_water.material.clear();
        without_water.density.clear();
        without_water.lambda.clear();
        without_water.delta_pos.clear();
        without_water.viscosity_work.clear();
        without_water.resize_work_buffers();
        let stone_indices_no_water =
            without_water.spawn_stone_particles_from_cells(&[stone_cell], Vec2::ZERO);

        let mut without_water_objects = ObjectWorld::default();
        let _ = without_water_objects
            .create_object(
                stone_indices_no_water,
                without_water.positions(),
                without_water.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            )
            .expect("object should be created");
        let mut without_water_field = ObjectPhysicsField::default();
        without_water_objects.update_physics_field(
            without_water.positions(),
            without_water.masses(),
            &mut without_water_field,
        );
        without_water.step_if_running(
            &terrain,
            &without_water_field,
            &mut without_water_objects,
            true,
        );

        let with_vel = with_water.vel[0];
        let without_vel = without_water.vel[0];
        assert!(
            (with_vel - without_vel).length() > 1e-3,
            "expected water contact to alter object velocity (with={with_vel:?}, without={without_vel:?})"
        );
    }
}
