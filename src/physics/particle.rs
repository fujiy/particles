use std::collections::{HashMap, HashSet};

use bevy::log::tracing;
use bevy::prelude::*;
use rayon::prelude::*;

use super::connectivity::{FOUR_NEIGHBOR_OFFSETS, flood_fill_4_limited};
use super::material::{
    ENABLE_GRANULAR_TO_SOLID_RECONVERSION, particle_properties, particles_per_cell,
    solid_break_properties, terrain_break_collision_impulse_threshold, terrain_fracture_particle,
    terrain_solid_particle,
};
use super::object::{
    OBJECT_SHAPE_ITERS, OBJECT_SHAPE_STIFFNESS_ALPHA, ObjectId, ObjectPhysicsField, ObjectWorld,
};
use super::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, TerrainCell, TerrainWorld, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y,
    WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y, cell_to_world_center, world_to_cell,
};

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
pub const DETACH_FLOOD_FILL_MAX_CELLS: usize = 128;
pub const SLEEP_DISP_THRESHOLD: f32 = 0.0020;
pub const SLEEP_VEL_THRESHOLD: f32 = 0.10;
pub const SLEEP_FRAMES: u16 = 30;
pub const WAKE_DISP_THRESHOLD: f32 = 0.0040;
pub const WAKE_RADIUS: f32 = CELL_SIZE_M * 1.75;
pub const ACTIVE_MIN_FRAMES: u16 = 8;
pub const GRANULAR_SUBSTEPS: usize = 1;
pub const GRANULAR_ITERS: usize = 8;
pub const XPBD_CONTACT_COMPLIANCE_N: f32 = 1e-6;
pub const XPBD_CONTACT_COMPLIANCE_T: f32 = 6e-6;
pub const WAKE_IMPULSE_THRESHOLD: f32 = 2.5;
pub const FRACTURE_WAKE_LOCK_FRAMES: u16 = 45;
pub const TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS: u32 = 2;
pub const TERRAIN_LOAD_STRAIN_THRESHOLD: f32 = 0.28;
pub const TERRAIN_LOAD_BREAK_DURATION_SECONDS: f32 = 0.45;
pub const TERRAIN_LOAD_DECAY_PER_SAMPLE: f32 = 0.8;
pub const GRANULAR_CONTACT_FRICTION_SCALE: f32 = 2.0;
pub const GRANULAR_GRANULAR_CONTACT_FRICTION_BOOST: f32 = 2.5;
pub const GRANULAR_SOLID_CONTACT_FRICTION_BOOST: f32 = 1.5;
pub const TERRAIN_CONTACT_FRICTION_SCALE: f32 = 1.75;
pub const GRANULAR_CONTACT_NORMAL_DAMPING: f32 = 0.50;
pub const TERRAIN_CONTACT_NORMAL_DAMPING: f32 = 0.65;
pub const GRANULAR_SPAWN_JITTER_RATIO: f32 = 0.01;
const PARTICLE_ESCAPE_MARGIN_X_CELLS: i32 = CHUNK_SIZE_I32;
const PARTICLE_ESCAPE_MARGIN_BOTTOM_CELLS: i32 = CHUNK_SIZE_I32;
const PARTICLE_ESCAPE_MARGIN_TOP_CELLS: i32 = CHUNK_SIZE_I32 * 8;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleActivityState {
    Active,
    Sleeping,
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GranularObjectContactKey {
    particle: usize,
    object_id: ObjectId,
}

#[derive(Default)]
struct GranularSolver;

impl GranularSolver {
    fn solve_contacts(
        particles: &mut ParticleWorld,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
        dt_sub: f32,
    ) -> HashMap<ObjectId, Vec2> {
        if GRANULAR_ITERS == 0 || GRANULAR_SUBSTEPS == 0 || particles.particle_count() == 0 {
            return HashMap::new();
        }

        let mut reaction_impulses = HashMap::<ObjectId, Vec2>::new();
        let n = particles.particle_count();
        let mut object_contacts = Vec::new();
        let granular_substeps = GRANULAR_SUBSTEPS.max(1);
        let dt_granular = dt_sub / granular_substeps as f32;
        let inv_dt_granular = 1.0 / dt_granular.max(1e-6);
        let alpha_n = XPBD_CONTACT_COMPLIANCE_N / dt_granular.max(1e-6).powi(2);
        let alpha_t = XPBD_CONTACT_COMPLIANCE_T / dt_granular.max(1e-6).powi(2);

        let mut lambda_n_pair = HashMap::<u64, f32>::new();
        let mut lambda_t_pair = HashMap::<u64, f32>::new();
        let mut lambda_n_terrain = vec![0.0f32; n];
        let mut lambda_t_terrain = vec![0.0f32; n];
        let mut lambda_n_object = HashMap::<GranularObjectContactKey, f32>::new();
        let mut lambda_t_object = HashMap::<GranularObjectContactKey, f32>::new();
        let mut wake_impulses = vec![0.0f32; n];

        for _ in 0..granular_substeps {
            particles.neighbor_grid.rebuild(&particles.pos);
            particles.rebuild_neighbor_cache(false);

            for _ in 0..GRANULAR_ITERS {
                let mut delta_pos = vec![Vec2::ZERO; n];
                wake_impulses.fill(0.0);

                for i in 0..n {
                    if !particles.is_active_particle(i) || is_water_particle(particles.material[i])
                    {
                        continue;
                    }

                    let props_i = particle_properties(particles.material[i]);
                    let inv_mass_i = 1.0 / particles.mass[i].max(1e-6);

                    for &j in &particles.neighbor_cache[i] {
                        if j <= i
                            || !particles.is_active_particle(j)
                            || is_water_particle(particles.material[j])
                        {
                            continue;
                        }

                        let props_j = particle_properties(particles.material[j]);
                        let r = particles.pos[i] - particles.pos[j];
                        let dist2 = r.length_squared();
                        if dist2 <= 1e-12 {
                            continue;
                        }
                        let contact_radius = props_i.radius_m + props_j.radius_m;
                        if dist2 >= contact_radius * contact_radius {
                            continue;
                        }

                        let dist = dist2.sqrt();
                        let n_ij = r / dist;
                        let c_n = dist - contact_radius;
                        let inv_mass_j = 1.0 / particles.mass[j].max(1e-6);
                        let w_sum = inv_mass_i + inv_mass_j;
                        if w_sum <= 1e-8 {
                            continue;
                        }

                        let pair_key = pack_pair_key(i, j);
                        let lambda_n = lambda_n_pair.entry(pair_key).or_insert(0.0);
                        let delta_lambda_n = (-c_n - alpha_n * *lambda_n) / (w_sum + alpha_n);
                        *lambda_n += delta_lambda_n;
                        let normal_correction = n_ij * delta_lambda_n;
                        let corr_i_n = normal_correction * inv_mass_i;
                        let corr_j_n = -normal_correction * inv_mass_j;
                        delta_pos[i] += corr_i_n;
                        delta_pos[j] += corr_j_n;

                        if let Some(scale) = granular_contact_friction_pair_scale(
                            particles.material[i],
                            particles.material[j],
                        ) {
                            let mu_k =
                                0.5 * (props_i.friction_dynamic + props_j.friction_dynamic) * scale;
                            let rel_vel = particles.vel[i] - particles.vel[j];
                            let tangent = rel_vel - n_ij * rel_vel.dot(n_ij);
                            let tangent_len = tangent.length();
                            if tangent_len > 1e-6 {
                                let t_hat = tangent / tangent_len;
                                let c_t = rel_vel.dot(t_hat) * dt_granular;
                                let lambda_t = lambda_t_pair.entry(pair_key).or_insert(0.0);
                                let delta_lambda_t =
                                    (-c_t - alpha_t * *lambda_t) / (w_sum + alpha_t);
                                let max_tangent = mu_k.max(0.0) * lambda_n.abs();
                                let next_lambda_t =
                                    (*lambda_t + delta_lambda_t).clamp(-max_tangent, max_tangent);
                                let applied_t = next_lambda_t - *lambda_t;
                                *lambda_t = next_lambda_t;
                                let tangent_correction = t_hat * applied_t;
                                delta_pos[i] += tangent_correction * inv_mass_i;
                                delta_pos[j] += -tangent_correction * inv_mass_j;
                            }
                        }

                        let impulse_i = particles.mass[i] * corr_i_n.length() * inv_dt_granular;
                        let impulse_j = particles.mass[j] * corr_j_n.length() * inv_dt_granular;
                        wake_impulses[i] += impulse_i;
                        wake_impulses[j] += impulse_j;
                    }

                    if let Some((signed_distance, normal)) =
                        terrain.sample_signed_distance_and_normal(particles.pos[i])
                    {
                        let c_n = signed_distance - props_i.terrain_push_radius_m;
                        if c_n < 0.0 {
                            let lambda_n = &mut lambda_n_terrain[i];
                            let delta_lambda_n =
                                (-c_n - alpha_n * *lambda_n) / (inv_mass_i + alpha_n);
                            *lambda_n += delta_lambda_n;
                            let corr_n = normal * (delta_lambda_n * inv_mass_i);
                            delta_pos[i] += corr_n;

                            let rel_vel = particles.vel[i];
                            let tangent = rel_vel - normal * rel_vel.dot(normal);
                            let tangent_len = tangent.length();
                            if tangent_len > 1e-6 {
                                let t_hat = tangent / tangent_len;
                                let c_t = rel_vel.dot(t_hat) * dt_granular;
                                let lambda_t = &mut lambda_t_terrain[i];
                                let delta_lambda_t =
                                    (-c_t - alpha_t * *lambda_t) / (inv_mass_i + alpha_t);
                                let max_tangent = (props_i.friction_dynamic
                                    * TERRAIN_CONTACT_FRICTION_SCALE)
                                    .max(0.0)
                                    * lambda_n.abs();
                                let next_lambda_t =
                                    (*lambda_t + delta_lambda_t).clamp(-max_tangent, max_tangent);
                                let applied_t = next_lambda_t - *lambda_t;
                                *lambda_t = next_lambda_t;
                                delta_pos[i] += t_hat * (applied_t * inv_mass_i);
                            }

                            wake_impulses[i] +=
                                particles.mass[i] * corr_n.length() * inv_dt_granular;
                        }
                    }

                    object_field
                        .gather_candidate_object_ids(particles.pos[i], &mut object_contacts);
                    let mut best_contact: Option<(ObjectId, Vec2, f32)> = None;
                    for &object_id in &object_contacts {
                        if object_world.object_of_particle(i) == Some(object_id) {
                            continue;
                        }
                        let Some(sample) =
                            object_world.evaluate_object_sdf(object_id, particles.pos[i])
                        else {
                            continue;
                        };
                        let c_n = sample.distance_m - props_i.object_push_radius_m;
                        if c_n >= 0.0 {
                            continue;
                        }
                        let replace = match best_contact {
                            Some((_, _, best_c_n)) => c_n < best_c_n,
                            None => true,
                        };
                        if replace {
                            best_contact = Some((object_id, sample.normal_world, c_n));
                        }
                    }

                    if let Some((object_id, normal, c_n)) = best_contact {
                        let key = GranularObjectContactKey {
                            particle: i,
                            object_id,
                        };
                        let lambda_n = lambda_n_object.entry(key).or_insert(0.0);
                        let delta_lambda_n = (-c_n - alpha_n * *lambda_n) / (inv_mass_i + alpha_n);
                        *lambda_n += delta_lambda_n;
                        let corr_n = normal * (delta_lambda_n * inv_mass_i);
                        delta_pos[i] += corr_n;

                        let rel_vel = particles.vel[i];
                        let tangent = rel_vel - normal * rel_vel.dot(normal);
                        let tangent_len = tangent.length();
                        if tangent_len > 1e-6 {
                            let t_hat = tangent / tangent_len;
                            let c_t = rel_vel.dot(t_hat) * dt_granular;
                            let lambda_t = lambda_t_object.entry(key).or_insert(0.0);
                            let delta_lambda_t =
                                (-c_t - alpha_t * *lambda_t) / (inv_mass_i + alpha_t);
                            let max_tangent = props_i.friction_dynamic.max(0.0) * lambda_n.abs();
                            let next_lambda_t =
                                (*lambda_t + delta_lambda_t).clamp(-max_tangent, max_tangent);
                            let applied_t = next_lambda_t - *lambda_t;
                            *lambda_t = next_lambda_t;
                            delta_pos[i] += t_hat * (applied_t * inv_mass_i);
                        }

                        let reaction_impulse = -(particles.mass[i] * corr_n) * inv_dt_granular;
                        *reaction_impulses.entry(object_id).or_insert(Vec2::ZERO) +=
                            reaction_impulse;
                        wake_impulses[i] += particles.mass[i] * corr_n.length() * inv_dt_granular;
                    }
                }

                for i in 0..n {
                    if !particles.is_active_particle(i) || is_water_particle(particles.material[i])
                    {
                        continue;
                    }
                    particles.pos[i] += delta_pos[i];
                    if wake_impulses[i] > WAKE_IMPULSE_THRESHOLD {
                        particles.request_wake(i);
                    }
                }

                particles.neighbor_grid.rebuild(&particles.pos);
                particles.rebuild_neighbor_cache(false);
            }
        }

        reaction_impulses
    }

    fn apply_restitution(particles: &mut ParticleWorld, _dt_sub: f32) {
        let mut neighbors = Vec::new();
        for i in 0..particles.particle_count() {
            if !particles.is_active_particle(i) || is_water_particle(particles.material[i]) {
                continue;
            }
            particles
                .neighbor_grid
                .gather(particles.pos[i], &mut neighbors);
            for &j in &neighbors {
                if j <= i
                    || !particles.is_active_particle(j)
                    || is_water_particle(particles.material[j])
                {
                    continue;
                }
                let r = particles.pos[i] - particles.pos[j];
                let dist2 = r.length_squared();
                if dist2 <= 1e-12 {
                    continue;
                }
                let props_i = particle_properties(particles.material[i]);
                let props_j = particle_properties(particles.material[j]);
                let contact_radius = props_i.radius_m + props_j.radius_m;
                if dist2 >= contact_radius * contact_radius {
                    continue;
                }

                let n_ij = r.normalize_or_zero();
                let rel_normal_speed = (particles.vel[i] - particles.vel[j]).dot(n_ij);
                if rel_normal_speed >= 0.0 {
                    continue;
                }
                let restitution = 0.5 * (props_i.contact_restitution + props_j.contact_restitution);
                let inv_mass_i = 1.0 / particles.mass[i].max(1e-6);
                let inv_mass_j = 1.0 / particles.mass[j].max(1e-6);
                let denom = (inv_mass_i + inv_mass_j).max(1e-6);
                let impulse = -(1.0 + restitution.clamp(0.0, 1.0)) * rel_normal_speed / denom;
                let impulse_vec = n_ij * impulse;
                particles.vel[i] += impulse_vec * inv_mass_i;
                particles.vel[j] -= impulse_vec * inv_mass_j;

                if impulse.abs() > WAKE_IMPULSE_THRESHOLD {
                    particles.request_wake(i);
                    particles.request_wake(j);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TerrainFractureSeed {
    impulse_sum: f32,
    velocity_weighted_sum: Vec2,
    contact_weighted_sum: Vec2,
}

impl TerrainFractureSeed {
    fn accumulate(&mut self, impulse: f32, velocity: Vec2, contact_pos: Vec2) {
        self.impulse_sum += impulse;
        self.velocity_weighted_sum += velocity * impulse;
        self.contact_weighted_sum += contact_pos * impulse;
    }

    fn velocity(self) -> Vec2 {
        if self.impulse_sum <= 1e-6 {
            Vec2::ZERO
        } else {
            self.velocity_weighted_sum / self.impulse_sum
        }
    }

    fn contact_pos(self) -> Vec2 {
        if self.impulse_sum <= 1e-6 {
            Vec2::ZERO
        } else {
            self.contact_weighted_sum / self.impulse_sum
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TerrainPersistentLoadState {
    strain_metric: f32,
    sustained_seconds: f32,
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
    activity_state: Vec<ParticleActivityState>,
    sleep_candidate_frames: Vec<u16>,
    active_hold_frames: Vec<u16>,
    sleep_lock_frames: Vec<u16>,
    pending_wake: Vec<bool>,
    initial_pos: Vec<Vec2>,
    initial_vel: Vec<Vec2>,
    neighbor_grid: NeighborGrid,
    neighbor_cache: Vec<Vec<usize>>,
    viscosity_work: Vec<Vec2>,
    object_peak_strain: HashMap<ObjectId, f32>,
    object_peak_strain_particle: HashMap<ObjectId, usize>,
    pending_object_fractures: HashSet<ObjectId>,
    pending_object_fracture_particles: HashMap<ObjectId, HashSet<usize>>,
    pending_terrain_fractures: HashSet<IVec2>,
    pending_terrain_fracture_seeds: HashMap<IVec2, TerrainFractureSeed>,
    terrain_persistent_load: HashMap<IVec2, TerrainPersistentLoadState>,
    terrain_load_substep_counter: u64,
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
            activity_state: Vec::new(),
            sleep_candidate_frames: Vec::new(),
            active_hold_frames: Vec::new(),
            sleep_lock_frames: Vec::new(),
            pending_wake: Vec::new(),
            initial_pos,
            initial_vel,
            neighbor_grid: NeighborGrid::default(),
            neighbor_cache: Vec::new(),
            viscosity_work: Vec::new(),
            object_peak_strain: HashMap::new(),
            object_peak_strain_particle: HashMap::new(),
            pending_object_fractures: HashSet::new(),
            pending_object_fracture_particles: HashMap::new(),
            pending_terrain_fractures: HashSet::new(),
            pending_terrain_fracture_seeds: HashMap::new(),
            terrain_persistent_load: HashMap::new(),
            terrain_load_substep_counter: 0,
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
        self.object_peak_strain.clear();
        self.object_peak_strain_particle.clear();
        self.pending_object_fractures.clear();
        self.pending_object_fracture_particles.clear();
        self.pending_terrain_fractures.clear();
        self.pending_terrain_fracture_seeds.clear();
        self.terrain_persistent_load.clear();
        self.terrain_load_substep_counter = 0;
        self.activity_state.fill(ParticleActivityState::Active);
        self.sleep_candidate_frames.fill(0);
        self.active_hold_frames.fill(0);
        self.sleep_lock_frames.fill(0);
        self.pending_wake.fill(false);
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

    pub fn activity_states(&self) -> &[ParticleActivityState] {
        &self.activity_state
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
            .map(|&material| particle_properties(material).mass)
            .collect();
        self.initial_pos = self.pos.clone();
        self.initial_vel = self.vel.clone();
        self.object_peak_strain.clear();
        self.object_peak_strain_particle.clear();
        self.pending_object_fractures.clear();
        self.pending_object_fracture_particles.clear();
        self.pending_terrain_fractures.clear();
        self.pending_terrain_fracture_seeds.clear();
        self.terrain_persistent_load.clear();
        self.terrain_load_substep_counter = 0;
        self.activity_state.fill(ParticleActivityState::Active);
        self.sleep_candidate_frames.fill(0);
        self.active_hold_frames.fill(0);
        self.sleep_lock_frames.fill(0);
        self.pending_wake.fill(false);
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
            self.request_wake(i);
        }
    }

    pub fn wake_particles_in_radius(&mut self, center: Vec2, radius: f32) {
        let radius2 = radius * radius;
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) <= radius2 {
                self.request_wake(i);
            }
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
            self.request_wake_near(cell_to_world_center(cell), WAKE_RADIUS);
        }
        self.resize_work_buffers();
        indices
    }

    pub fn apply_pending_terrain_fractures(
        &mut self,
        terrain: &mut TerrainWorld,
        object_world: &mut ObjectWorld,
    ) -> bool {
        if self.pending_terrain_fractures.is_empty() {
            return false;
        }

        let fracture_cells: Vec<_> = self.pending_terrain_fractures.drain().collect();
        let fracture_cell_set: HashSet<_> = fracture_cells.iter().copied().collect();
        let mut changed = false;
        let mut appended_indices = Vec::new();
        for &cell in &fracture_cells {
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
                self.terrain_fracture_seed_velocity(cell),
                &mut appended_indices,
            );
        }

        let detached_components =
            self.collect_detached_terrain_components(terrain, &fracture_cell_set);

        for component_cells in detached_components {
            let spawn_plan =
                self.build_terrain_detach_spawn_plan(&component_cells, terrain, &fracture_cell_set);
            if spawn_plan.cells.is_empty() {
                continue;
            }
            let indices = self.spawn_terrain_detach_component(&spawn_plan);
            if indices.is_empty() {
                continue;
            }

            for cell in &spawn_plan.cells {
                changed |= terrain.set_cell(*cell, TerrainCell::Empty);
            }
            let _ = object_world.create_object(
                indices,
                self.positions(),
                self.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            );
        }

        if !appended_indices.is_empty() || changed {
            for &cell in &fracture_cells {
                self.request_wake_near(cell_to_world_center(cell), WAKE_RADIUS);
            }
            self.resize_work_buffers();
        }
        if changed {
            self.auto_fracture_single_cell_objects(object_world);
        }
        self.pending_terrain_fracture_seeds.clear();
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
        let mut new_activity_state = Vec::with_capacity(new_count);
        let mut new_sleep_candidate_frames = Vec::with_capacity(new_count);
        let mut new_active_hold_frames = Vec::with_capacity(new_count);
        let mut new_sleep_lock_frames = Vec::with_capacity(new_count);
        let mut new_pending_wake = Vec::with_capacity(new_count);
        for old_index in 0..old_count {
            if !keep[old_index] {
                continue;
            }
            new_pos.push(self.pos[old_index]);
            new_prev_pos.push(self.prev_pos[old_index]);
            new_vel.push(self.vel[old_index]);
            new_mass.push(self.mass[old_index]);
            new_material.push(self.material[old_index]);
            new_activity_state.push(self.activity_state[old_index]);
            new_sleep_candidate_frames.push(self.sleep_candidate_frames[old_index]);
            new_active_hold_frames.push(self.active_hold_frames[old_index]);
            new_sleep_lock_frames.push(self.sleep_lock_frames[old_index]);
            new_pending_wake.push(self.pending_wake[old_index]);
        }

        self.pos = new_pos;
        self.prev_pos = new_prev_pos;
        self.vel = new_vel;
        self.mass = new_mass;
        self.material = new_material;
        self.activity_state = new_activity_state;
        self.sleep_candidate_frames = new_sleep_candidate_frames;
        self.active_hold_frames = new_active_hold_frames;
        self.sleep_lock_frames = new_sleep_lock_frames;
        self.pending_wake = new_pending_wake;
        self.request_wake_near(center, WAKE_RADIUS);
        self.resize_work_buffers();

        ParticleRemovalResult {
            removed_count,
            old_to_new,
        }
    }

    pub fn fracture_solid_particles_in_radius(
        &mut self,
        center: Vec2,
        radius: f32,
    ) -> HashSet<usize> {
        let radius2 = radius * radius;
        let mut detached = HashSet::new();
        let mut spawned_particles = Vec::new();
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) > radius2 {
                continue;
            }
            let Some(target_material) =
                solid_break_properties(self.material[i]).and_then(|props| props.fracture_to)
            else {
                continue;
            };
            detached.insert(i);
            self.fracture_particle_to_target_material(i, target_material, &mut spawned_particles);
        }
        for (position, velocity, mass, material) in spawned_particles {
            let new_index = self.pos.len();
            self.pos.push(position);
            self.prev_pos.push(position);
            self.vel.push(velocity);
            self.mass.push(mass);
            self.material.push(material);
            if is_granular_particle(material) {
                self.request_wake(new_index);
                self.sleep_lock_frames[new_index] = FRACTURE_WAKE_LOCK_FRAMES;
            }
        }
        if !detached.is_empty() {
            self.request_wake_near(center, WAKE_RADIUS);
            self.resize_work_buffers();
        }
        detached
    }

    pub fn detach_terrain_components_after_cell_removal(
        &mut self,
        terrain: &mut TerrainWorld,
        object_world: &mut ObjectWorld,
        removed_cells: &HashSet<IVec2>,
    ) -> bool {
        if removed_cells.is_empty() {
            return false;
        }
        let detached_components = self.collect_detached_terrain_components(terrain, removed_cells);
        if detached_components.is_empty() {
            return false;
        }

        let mut changed = false;
        let no_seed_cells = HashSet::new();
        for component_cells in detached_components {
            let spawn_plan =
                self.build_terrain_detach_spawn_plan(&component_cells, terrain, &no_seed_cells);
            if spawn_plan.cells.is_empty() {
                continue;
            }
            let indices = self.spawn_terrain_detach_component(&spawn_plan);
            if indices.is_empty() {
                continue;
            }
            for cell in &spawn_plan.cells {
                changed |= terrain.set_cell(*cell, TerrainCell::Empty);
            }
            let _ = object_world.create_object(
                indices,
                self.positions(),
                self.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            );
        }
        if changed {
            for &cell in removed_cells {
                self.request_wake_near(cell_to_world_center(cell), WAKE_RADIUS);
            }
            self.resize_work_buffers();
            self.auto_fracture_single_cell_objects(object_world);
        }
        changed
    }

    pub fn postprocess_objects_after_topology_edit(&mut self, object_world: &mut ObjectWorld) {
        object_world.split_all_disconnected_objects(self.positions(), self.masses());
        self.auto_fracture_single_cell_objects(object_world);
    }

    pub fn detach_and_postprocess_objects(
        &mut self,
        object_world: &mut ObjectWorld,
        detached_particles: &HashSet<usize>,
    ) {
        object_world.split_objects_after_detach(
            detached_particles,
            self.positions(),
            self.masses(),
        );
        self.auto_fracture_single_cell_objects(object_world);
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
        self.terrain_load_substep_counter = self.terrain_load_substep_counter.wrapping_add(1);
        self.object_peak_strain.clear();
        self.object_peak_strain_particle.clear();
        {
            let _span = tracing::info_span!("physics::clear_reaction_impulses").entered();
            object_world.clear_reaction_impulses();
        }
        {
            let _span = tracing::info_span!("physics::predict_positions").entered();
            for i in 0..self.particle_count() {
                if !self.is_active_particle(i) {
                    self.prev_pos[i] = self.pos[i];
                    self.vel[i] = Vec2::ZERO;
                    continue;
                }
                self.prev_pos[i] = self.pos[i];
                self.vel[i] += GRAVITY_MPS2 * dt_sub;
                self.pos[i] += self.vel[i] * dt_sub;
            }
        }
        {
            let _span = tracing::info_span!("physics::cull_escaped_particles").entered();
            self.cull_escaped_particles(object_world);
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
            let _span = tracing::info_span!("physics::granular_solver").entered();
            let reaction_impulses =
                GranularSolver::solve_contacts(self, terrain, object_field, object_world, dt_sub);
            for (object_id, impulse) in reaction_impulses {
                object_world.accumulate_reaction_impulse(object_id, impulse);
            }
        }
        {
            let _span = tracing::info_span!("physics::shape_matching").entered();
            self.solve_shape_matching_constraints(object_world);
        }

        {
            let _span = tracing::info_span!("physics::update_velocity").entered();
            for i in 0..self.particle_count() {
                if !self.is_active_particle(i) {
                    self.vel[i] = Vec2::ZERO;
                    self.prev_pos[i] = self.pos[i];
                    continue;
                }
                self.vel[i] = (self.pos[i] - self.prev_pos[i]) / dt_sub;
                self.vel[i] = self.vel[i].clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
            }
        }
        {
            let _span = tracing::info_span!("physics::granular_restitution").entered();
            GranularSolver::apply_restitution(self, dt_sub);
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
            for i in 0..self.particle_count() {
                if self.is_active_particle(i) {
                    self.vel[i] = self.vel[i].clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
                } else {
                    self.vel[i] = Vec2::ZERO;
                }
            }
        }
        {
            let _span = tracing::info_span!("physics::wake_detection").entered();
            self.neighbor_grid.rebuild(&self.pos);
            self.detect_wake_events(terrain, object_field, object_world);
            self.propagate_and_apply_wake_requests();
        }
        {
            let _span = tracing::info_span!("physics::fracture_detection").entered();
            self.detect_fracture_candidates(terrain, object_field, object_world);
            self.apply_object_fractures(object_world);
        }
        {
            let _span = tracing::info_span!("physics::sleep_update").entered();
            self.update_sleep_states();
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
            if !self.is_active_particle(i) || !is_water_particle(self.material[i]) {
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
        let activity_state = &self.activity_state;
        let neighbor_cache = &self.neighbor_cache;

        self.density
            .par_iter_mut()
            .zip(self.lambda.par_iter_mut())
            .enumerate()
            .map(|(i, (density_i, lambda_i))| {
                if activity_state[i] == ParticleActivityState::Sleeping
                    || !is_water_particle(materials[i])
                {
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
        let dt_sub = 1.0 / inv_dt.max(1e-6);
        let mut reaction_impulses = HashMap::new();
        let mut object_contacts = Vec::new();
        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) {
                self.delta_pos[i] = Vec2::ZERO;
                continue;
            }
            let material = self.material[i];
            let props = particle_properties(material);
            let is_water = is_water_particle(material);
            if !is_water {
                self.delta_pos[i] = Vec2::ZERO;
                continue;
            }
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
                let neighbor_props = particle_properties(neighbor_material);
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
                let normal_push = normal * penetration * stiffness * PARTICLE_CONTACT_PUSH_FACTOR;
                particle_contact_push += normal_push;
                if let Some(contact_pair_scale) =
                    granular_contact_friction_pair_scale(material, neighbor_material)
                {
                    let static_friction = 0.5
                        * (props.friction_static + neighbor_props.friction_static)
                        * GRANULAR_CONTACT_FRICTION_SCALE
                        * contact_pair_scale;
                    let dynamic_friction = 0.5
                        * (props.friction_dynamic + neighbor_props.friction_dynamic)
                        * GRANULAR_CONTACT_FRICTION_SCALE
                        * contact_pair_scale;
                    let rel_velocity = self.vel[i] - self.vel[j];
                    particle_contact_push += granular_tangential_friction_push(
                        normal,
                        normal_push,
                        rel_velocity,
                        static_friction,
                        dynamic_friction,
                        dt_sub,
                    );
                    let rel_normal_speed = (self.vel[i] - self.vel[j]).dot(normal);
                    if rel_normal_speed < 0.0 {
                        let normal_damping_push = -normal
                            * rel_normal_speed
                            * dt_sub
                            * GRANULAR_CONTACT_NORMAL_DAMPING
                            * contact_pair_scale;
                        let max_damping_push = penetration * 0.35;
                        particle_contact_push +=
                            normal_damping_push.clamp_length_max(max_damping_push);
                    }
                }
            }

            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            {
                let penetration = props.terrain_push_radius_m - signed_distance;
                if penetration > 0.0 {
                    let normal_push = normal * penetration * props.terrain_repulsion_stiffness;
                    boundary_push += normal_push;
                    if !is_water {
                        boundary_push += granular_tangential_friction_push(
                            normal,
                            normal_push,
                            self.vel[i],
                            props.friction_static * TERRAIN_CONTACT_FRICTION_SCALE,
                            props.friction_dynamic * TERRAIN_CONTACT_FRICTION_SCALE,
                            dt_sub,
                        );
                        let normal_speed = self.vel[i].dot(normal);
                        if normal_speed < 0.0 {
                            let normal_damping_push =
                                -normal * normal_speed * dt_sub * TERRAIN_CONTACT_NORMAL_DAMPING;
                            boundary_push +=
                                normal_damping_push.clamp_length_max(penetration * 0.35);
                        }
                    }
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
        let dt_sub = 1.0 / inv_dt.max(1e-6);
        let positions = &self.pos;
        let velocities = &self.vel;
        let masses = &self.mass;
        let materials = &self.material;
        let activity_state = &self.activity_state;
        let lambdas = &self.lambda;
        let neighbor_cache = &self.neighbor_cache;

        let reduced = self
            .delta_pos
            .par_iter_mut()
            .enumerate()
            .fold(
                ComputeDeltaThreadScratch::default,
                |mut scratch, (i, delta_pos_i)| {
                    if activity_state[i] == ParticleActivityState::Sleeping {
                        *delta_pos_i = Vec2::ZERO;
                        return scratch;
                    }
                    let material = materials[i];
                    let props = particle_properties(material);
                    let is_water = is_water_particle(material);
                    if !is_water {
                        *delta_pos_i = Vec2::ZERO;
                        return scratch;
                    }
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
                        let neighbor_props = particle_properties(neighbor_material);
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
                        let normal_push =
                            normal * penetration * stiffness * PARTICLE_CONTACT_PUSH_FACTOR;
                        particle_contact_push += normal_push;
                        if let Some(contact_pair_scale) =
                            granular_contact_friction_pair_scale(material, neighbor_material)
                        {
                            let static_friction = 0.5
                                * (props.friction_static + neighbor_props.friction_static)
                                * GRANULAR_CONTACT_FRICTION_SCALE
                                * contact_pair_scale;
                            let dynamic_friction = 0.5
                                * (props.friction_dynamic + neighbor_props.friction_dynamic)
                                * GRANULAR_CONTACT_FRICTION_SCALE
                                * contact_pair_scale;
                            let rel_velocity = velocities[i] - velocities[j];
                            particle_contact_push += granular_tangential_friction_push(
                                normal,
                                normal_push,
                                rel_velocity,
                                static_friction,
                                dynamic_friction,
                                dt_sub,
                            );
                            let rel_normal_speed = (velocities[i] - velocities[j]).dot(normal);
                            if rel_normal_speed < 0.0 {
                                let normal_damping_push = -normal
                                    * rel_normal_speed
                                    * dt_sub
                                    * GRANULAR_CONTACT_NORMAL_DAMPING
                                    * contact_pair_scale;
                                let max_damping_push = penetration * 0.35;
                                particle_contact_push +=
                                    normal_damping_push.clamp_length_max(max_damping_push);
                            }
                        }
                    }

                    if let Some((signed_distance, normal)) =
                        terrain.sample_signed_distance_and_normal(positions[i])
                    {
                        let penetration = props.terrain_push_radius_m - signed_distance;
                        if penetration > 0.0 {
                            let normal_push =
                                normal * penetration * props.terrain_repulsion_stiffness;
                            boundary_push += normal_push;
                            if !is_water {
                                boundary_push += granular_tangential_friction_push(
                                    normal,
                                    normal_push,
                                    velocities[i],
                                    props.friction_static * TERRAIN_CONTACT_FRICTION_SCALE,
                                    props.friction_dynamic * TERRAIN_CONTACT_FRICTION_SCALE,
                                    dt_sub,
                                );
                                let normal_speed = velocities[i].dot(normal);
                                if normal_speed < 0.0 {
                                    let normal_damping_push = -normal
                                        * normal_speed
                                        * dt_sub
                                        * TERRAIN_CONTACT_NORMAL_DAMPING;
                                    boundary_push +=
                                        normal_damping_push.clamp_length_max(penetration * 0.35);
                                }
                            }
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
            if !self.is_active_particle(i) {
                continue;
            }
            let material = self.material[i];
            let props = particle_properties(material);
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
            let normal_contact_speed = normal_speed.abs();
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
            let adjusted_normal_speed = if normal_speed < 0.0 {
                -normal_speed * props.contact_restitution.clamp(0.0, 1.0)
            } else {
                normal_speed
            };
            self.vel[i] = adjusted_tangent + contact_normal * adjusted_normal_speed;
        }
    }

    fn apply_xsph_viscosity(&mut self) {
        let _span = tracing::info_span!("physics::xsph_pass").entered();
        let mut neighbors = Vec::new();
        self.viscosity_work.clone_from(&self.vel);

        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) || !is_water_particle(self.material[i]) {
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
            let viscosity = particle_properties(self.material[i]).xsph_viscosity;
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

    fn detect_fracture_candidates(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
    ) {
        debug_assert!(
            !ENABLE_GRANULAR_TO_SOLID_RECONVERSION,
            "v4 keeps granular->solid reconversion disabled"
        );
        self.pending_object_fractures.clear();
        self.pending_object_fracture_particles.clear();
        self.pending_terrain_fractures.clear();
        self.pending_terrain_fracture_seeds.clear();

        let mut terrain_impulse = HashMap::<IVec2, f32>::new();
        let mut terrain_load_metric = HashMap::<IVec2, f32>::new();
        let mut terrain_seed = HashMap::<IVec2, TerrainFractureSeed>::new();
        let mut object_terrain_impulse = HashMap::<ObjectId, f32>::new();
        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) {
                continue;
            }
            let particle_material = self.material[i];
            let props = particle_properties(particle_material);
            let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            else {
                continue;
            };
            let penetration = props.terrain_push_radius_m - signed_distance;
            if penetration <= 0.0 {
                continue;
            }
            let dt_sub = FIXED_DT / SUBSTEPS as f32;
            let step_normal_speed =
                (self.pos[i] - self.prev_pos[i]).dot(normal).abs() / dt_sub.max(1e-6);
            let normal_speed = self.vel[i].dot(normal).abs().max(step_normal_speed);
            let impulse = self.mass[i] * normal_speed;
            if impulse <= 1e-6 {
                continue;
            }
            if let Some(object_id) = object_world.object_of_particle(i) {
                *object_terrain_impulse.entry(object_id).or_insert(0.0) += impulse;
                self.pending_object_fracture_particles
                    .entry(object_id)
                    .or_default()
                    .insert(i);
            }
            let Some(cell) =
                resolve_terrain_contact_cell(self.pos[i], signed_distance, normal, terrain)
            else {
                continue;
            };
            let strain_metric = (penetration / props.terrain_push_radius_m.max(1e-6)).max(0.0);
            terrain_load_metric
                .entry(cell)
                .and_modify(|metric| *metric = metric.max(strain_metric))
                .or_insert(strain_metric);
            let entry = terrain_impulse.entry(cell).or_insert(0.0);
            *entry = (*entry).max(impulse);
            terrain_seed
                .entry(cell)
                .or_default()
                .accumulate(impulse, self.vel[i], self.pos[i]);
        }

        self.evaluate_terrain_persistent_load(terrain, &terrain_load_metric);

        let mut object_contacts = Vec::new();
        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) {
                continue;
            }
            let Some(object_id) = object_world.object_of_particle(i) else {
                continue;
            };
            let props = particle_properties(self.material[i]);
            object_field.gather_candidate_object_ids(self.pos[i], &mut object_contacts);
            for &contact_id in &object_contacts {
                if contact_id == object_id {
                    continue;
                }
                let Some(sample) = object_world.evaluate_object_sdf(contact_id, self.pos[i]) else {
                    continue;
                };
                let penetration = props.object_push_radius_m - sample.distance_m;
                if penetration <= 0.0 {
                    continue;
                }
                self.pending_object_fracture_particles
                    .entry(object_id)
                    .or_default()
                    .insert(i);
                break;
            }
        }

        for object in object_world.objects() {
            let Some(&particle_index) = object.particle_indices.first() else {
                continue;
            };
            if particle_index >= self.particle_count() {
                continue;
            }
            let source_material = self.material[particle_index];
            let Some(break_props) = solid_break_properties(source_material) else {
                continue;
            };

            let external_collision_impulse = object_world.reaction_impulse_of(object.id).length();
            let terrain_collision_impulse = object_terrain_impulse
                .get(&object.id)
                .copied()
                .unwrap_or(0.0);
            let collision_impulse = external_collision_impulse + terrain_collision_impulse;
            let strain = self
                .object_peak_strain
                .get(&object.id)
                .copied()
                .unwrap_or(0.0);
            let should_fracture = collision_impulse
                >= break_props.break_collision_impulse_threshold
                || strain >= break_props.break_strain_threshold;
            if should_fracture {
                self.pending_object_fractures.insert(object.id);
                let fracture_particles = self
                    .pending_object_fracture_particles
                    .entry(object.id)
                    .or_default();
                if fracture_particles.is_empty() {
                    if let Some(&peak_particle) = self.object_peak_strain_particle.get(&object.id) {
                        fracture_particles.insert(peak_particle);
                    } else {
                        fracture_particles.insert(particle_index);
                    }
                }
            }
        }

        for (cell, impulse) in terrain_impulse {
            let TerrainCell::Solid { material, .. } = terrain.get_loaded_cell_or_empty(cell) else {
                continue;
            };
            if terrain_fracture_particle(material).is_none() {
                continue;
            }
            let threshold = terrain_break_collision_impulse_threshold(material);
            if impulse >= threshold {
                self.pending_terrain_fractures.insert(cell);
                if let Some(seed) = terrain_seed.get(&cell).copied() {
                    self.pending_terrain_fracture_seeds.insert(cell, seed);
                }
            }
        }
    }

    fn apply_object_fractures(&mut self, object_world: &mut ObjectWorld) {
        if self.pending_object_fractures.is_empty() {
            return;
        }

        let mut fracture_plans = Vec::new();
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
            let Some(target_material) =
                solid_break_properties(source_material).and_then(|props| props.fracture_to)
            else {
                continue;
            };
            let mut detached = HashSet::new();
            if let Some(fracture_particles) = self.pending_object_fracture_particles.get(&object.id)
            {
                for &index in &object.particle_indices {
                    if fracture_particles.contains(&index) {
                        detached.insert(index);
                    }
                }
            }
            if detached.is_empty() {
                detached.insert(seed_index);
            }
            fracture_plans.push((object.id, target_material, detached));
        }

        let mut spawned_particles = Vec::new();
        for (object_id, target_material, detached_particles) in fracture_plans {
            for &index in &detached_particles {
                self.fracture_particle_to_target_material(
                    index,
                    target_material,
                    &mut spawned_particles,
                );
            }

            object_world.split_object_after_detach(
                object_id,
                &detached_particles,
                self.positions(),
                self.masses(),
            );
        }

        for (position, velocity, mass, material) in spawned_particles {
            let new_index = self.pos.len();
            self.pos.push(position);
            self.prev_pos.push(position);
            self.vel.push(velocity);
            self.mass.push(mass);
            self.material.push(material);
            if is_granular_particle(material) {
                self.request_wake(new_index);
                self.sleep_lock_frames[new_index] = FRACTURE_WAKE_LOCK_FRAMES;
            }
        }
        self.resize_work_buffers();

        self.pending_object_fractures.clear();
        self.pending_object_fracture_particles.clear();
        self.auto_fracture_single_cell_objects(object_world);
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
            let mut object_peak_particle = None;
            let shape_iters = object.shape_iters.max(1);
            for _ in 0..shape_iters {
                let mut mass_sum = 0.0;
                let mut com = Vec2::ZERO;
                for &index in &object.particle_indices {
                    if index >= self.particle_count() || !self.is_active_particle(index) {
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
                    if !self.is_active_particle(index) {
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
                    if !self.is_active_particle(index) {
                        continue;
                    }
                    let q = object.rest_local[slot];
                    let rotated = Vec2::new(cos_t * q.x - sin_t * q.y, sin_t * q.x + cos_t * q.y);
                    let goal = com + rotated;
                    let current = self.pos[index];
                    let strain = (goal - current).length() / CELL_SIZE_M.max(1e-6);
                    if strain > object_peak_strain {
                        object_peak_strain = strain;
                        object_peak_particle = Some(index);
                    }
                    self.pos[index] = current + (goal - current) * alpha;
                }
            }
            self.object_peak_strain
                .insert(object.id, object_peak_strain);
            if let Some(particle_index) = object_peak_particle {
                self.object_peak_strain_particle
                    .insert(object.id, particle_index);
            } else {
                self.object_peak_strain_particle.remove(&object.id);
            }
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
        let props = particle_properties(material);
        let count = particles_per_cell(material);
        let axis = particle_grid_axis(count);
        let axis_f = axis as f32;
        let spacing = CELL_SIZE_M / axis_f.max(1.0);
        let cell_min = cell_to_world_center(cell) - Vec2::splat(CELL_SIZE_M * 0.5);
        let particle_mass = props.mass;
        let jitter_scale = spacing * GRANULAR_SPAWN_JITTER_RATIO;
        let min_bound = cell_min + Vec2::splat(props.radius_m);
        let max_bound = cell_min + Vec2::splat(CELL_SIZE_M - props.radius_m);
        let granular_spawn = is_granular_particle(material) && jitter_scale > 0.0;

        for y in 0..axis {
            for x in 0..axis {
                let local = Vec2::new((x as f32 + 0.5) * spacing, (y as f32 + 0.5) * spacing);
                let mut position = cell_min + local;
                if granular_spawn {
                    position += spawn_jitter_offset(cell, material, x, y, jitter_scale);
                    position = Vec2::new(
                        position.x.clamp(min_bound.x, max_bound.x),
                        position.y.clamp(min_bound.y, max_bound.y),
                    );
                }
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

    fn fracture_particle_to_target_material(
        &mut self,
        index: usize,
        target_material: ParticleMaterial,
        spawned_particles: &mut Vec<(Vec2, Vec2, f32, ParticleMaterial)>,
    ) {
        if index >= self.particle_count() {
            return;
        }
        let target_props = particle_properties(target_material);
        let target_mass = target_props.mass;
        let split_count = target_props.particles_per_cell.max(1) as usize;
        let base_pos = self.pos[index];
        let base_vel = self.vel[index];
        self.material[index] = target_material;
        self.mass[index] = target_mass;
        if is_granular_particle(target_material) {
            self.request_wake(index);
            self.sleep_lock_frames[index] = FRACTURE_WAKE_LOCK_FRAMES;
        }
        if split_count <= 1 {
            return;
        }

        let axis = (split_count as f32).sqrt().ceil() as usize;
        let spacing = CELL_SIZE_M / axis as f32;
        let cell_min = base_pos - Vec2::splat(CELL_SIZE_M * 0.5);
        let mut placed = 0usize;
        'grid: for y in 0..axis {
            for x in 0..axis {
                if placed >= split_count {
                    break 'grid;
                }
                let split_pos =
                    cell_min + Vec2::new((x as f32 + 0.5) * spacing, (y as f32 + 0.5) * spacing);
                if placed == 0 {
                    self.pos[index] = split_pos;
                    self.prev_pos[index] = split_pos;
                } else {
                    spawned_particles.push((split_pos, base_vel, target_mass, target_material));
                }
                placed += 1;
            }
        }
    }

    fn auto_fracture_single_cell_objects(&mut self, object_world: &mut ObjectWorld) {
        let mut objects_to_remove = HashSet::new();
        let mut targets = Vec::<(usize, ParticleMaterial)>::new();
        for object in object_world.objects() {
            if object.particle_indices.is_empty() {
                continue;
            }
            let mut occupied_cells = HashSet::new();
            for &index in &object.particle_indices {
                if index >= self.particle_count() {
                    continue;
                }
                occupied_cells.insert(world_to_cell(self.pos[index]));
                if occupied_cells.len() > 1 {
                    break;
                }
            }
            if occupied_cells.len() != 1 {
                continue;
            }
            let mut has_fracturable = false;
            for &index in &object.particle_indices {
                if index >= self.particle_count() {
                    continue;
                }
                let Some(target_material) = solid_break_properties(self.material[index])
                    .and_then(|props| props.fracture_to)
                else {
                    continue;
                };
                has_fracturable = true;
                targets.push((index, target_material));
            }
            if has_fracturable {
                objects_to_remove.insert(object.id);
            }
        }
        if targets.is_empty() {
            return;
        }

        let mut spawned_particles = Vec::new();
        for (index, target_material) in targets {
            self.fracture_particle_to_target_material(
                index,
                target_material,
                &mut spawned_particles,
            );
        }
        for (position, velocity, mass, material) in spawned_particles {
            let new_index = self.pos.len();
            self.pos.push(position);
            self.prev_pos.push(position);
            self.vel.push(velocity);
            self.mass.push(mass);
            self.material.push(material);
            if is_granular_particle(material) {
                self.request_wake(new_index);
                self.sleep_lock_frames[new_index] = FRACTURE_WAKE_LOCK_FRAMES;
            }
        }
        self.resize_work_buffers();
        object_world.remove_objects_by_ids(&objects_to_remove);
    }

    fn is_active_particle(&self, index: usize) -> bool {
        self.activity_state
            .get(index)
            .copied()
            .unwrap_or(ParticleActivityState::Active)
            == ParticleActivityState::Active
    }

    fn request_wake(&mut self, index: usize) {
        if index >= self.particle_count() {
            return;
        }
        if self.pending_wake.len() < self.particle_count() {
            self.pending_wake.resize(self.particle_count(), false);
        }
        if self.activity_state.len() < self.particle_count() {
            self.activity_state
                .resize(self.particle_count(), ParticleActivityState::Active);
        }
        if self.sleep_candidate_frames.len() < self.particle_count() {
            self.sleep_candidate_frames.resize(self.particle_count(), 0);
        }
        if self.active_hold_frames.len() < self.particle_count() {
            self.active_hold_frames.resize(self.particle_count(), 0);
        }
        if self.sleep_lock_frames.len() < self.particle_count() {
            self.sleep_lock_frames.resize(self.particle_count(), 0);
        }
        self.pending_wake[index] = true;
    }

    fn request_wake_near(&mut self, center: Vec2, radius: f32) {
        let radius2 = radius * radius;
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) <= radius2 {
                self.request_wake(i);
            }
        }
    }

    fn propagate_and_apply_wake_requests(&mut self) {
        let wake_radius2 = WAKE_RADIUS * WAKE_RADIUS;
        let mut seeds = Vec::new();
        for i in 0..self.particle_count() {
            if self.pending_wake[i] {
                seeds.push(i);
            }
        }

        let mut neighbors = Vec::new();
        for &seed in &seeds {
            self.neighbor_grid.gather(self.pos[seed], &mut neighbors);
            for &j in &neighbors {
                if j == seed || self.is_active_particle(j) {
                    continue;
                }
                if self.pos[seed].distance_squared(self.pos[j]) <= wake_radius2 {
                    self.pending_wake[j] = true;
                }
            }
        }

        for i in 0..self.particle_count() {
            if !self.pending_wake[i] {
                continue;
            }
            self.activity_state[i] = ParticleActivityState::Active;
            self.sleep_candidate_frames[i] = 0;
            self.active_hold_frames[i] = ACTIVE_MIN_FRAMES;
            self.pending_wake[i] = false;
        }
    }

    fn update_sleep_states(&mut self) {
        for i in 0..self.particle_count() {
            if self.pending_wake[i] {
                continue;
            }
            if self.activity_state[i] == ParticleActivityState::Sleeping {
                self.vel[i] = Vec2::ZERO;
                self.prev_pos[i] = self.pos[i];
                continue;
            }

            if self.active_hold_frames[i] > 0 {
                self.active_hold_frames[i] -= 1;
                self.sleep_candidate_frames[i] = 0;
                continue;
            }
            if self.sleep_lock_frames[i] > 0 {
                self.sleep_lock_frames[i] -= 1;
                self.sleep_candidate_frames[i] = 0;
                continue;
            }

            let displacement = (self.pos[i] - self.prev_pos[i]).length();
            let speed = self.vel[i].length();
            let should_sleep_sample =
                displacement < SLEEP_DISP_THRESHOLD && speed < SLEEP_VEL_THRESHOLD;

            if should_sleep_sample {
                self.sleep_candidate_frames[i] = self.sleep_candidate_frames[i].saturating_add(1);
                if self.sleep_candidate_frames[i] >= SLEEP_FRAMES {
                    self.activity_state[i] = ParticleActivityState::Sleeping;
                    self.sleep_candidate_frames[i] = 0;
                    self.vel[i] = Vec2::ZERO;
                    self.prev_pos[i] = self.pos[i];
                }
            } else {
                self.sleep_candidate_frames[i] = 0;
            }
        }
    }

    fn detect_wake_events(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
    ) {
        let mut neighbors = Vec::new();
        let mut object_contacts = Vec::new();
        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) {
                continue;
            }
            let displacement_i = self.pos[i] - self.prev_pos[i];
            if displacement_i.length() > WAKE_DISP_THRESHOLD {
                self.request_wake(i);
            }

            self.neighbor_grid.gather(self.pos[i], &mut neighbors);
            for &j in &neighbors {
                if i == j || self.is_active_particle(j) {
                    continue;
                }
                let r = self.pos[i] - self.pos[j];
                let contact_radius = particle_properties(self.material[i]).radius_m
                    + particle_properties(self.material[j]).radius_m;
                let dist2 = r.length_squared();
                if dist2 <= 1e-12 || dist2 >= contact_radius * contact_radius {
                    continue;
                }
                let dist = dist2.sqrt();
                let normal = r / dist;
                let displacement_j = self.pos[j] - self.prev_pos[j];
                let relative_normal_displacement =
                    (displacement_i - displacement_j).dot(normal).abs();
                if relative_normal_displacement > WAKE_DISP_THRESHOLD {
                    self.request_wake(i);
                    self.request_wake(j);
                }
            }

            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            {
                let props = particle_properties(self.material[i]);
                let penetration = props.terrain_push_radius_m - signed_distance;
                if penetration > 0.0 {
                    let normal_displacement = displacement_i.dot(normal).abs();
                    if normal_displacement > WAKE_DISP_THRESHOLD {
                        self.request_wake(i);
                        self.request_wake_near(self.pos[i], WAKE_RADIUS);
                    }
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
                let props = particle_properties(self.material[i]);
                let penetration = props.object_push_radius_m - sample.distance_m;
                if penetration <= 0.0 {
                    continue;
                }
                let normal_displacement = displacement_i.dot(sample.normal_world).abs();
                if normal_displacement <= WAKE_DISP_THRESHOLD {
                    continue;
                }
                self.request_wake(i);
                if let Some(target_object) = object_world
                    .objects()
                    .iter()
                    .find(|object| object.id == object_id)
                {
                    for &index in &target_object.particle_indices {
                        self.request_wake(index);
                    }
                }
            }
        }
    }

    fn cull_escaped_particles(&mut self, object_world: &mut ObjectWorld) {
        if self.particle_count() == 0 {
            return;
        }

        let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32 - PARTICLE_ESCAPE_MARGIN_X_CELLS;
        let max_cell_x =
            (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1 + PARTICLE_ESCAPE_MARGIN_X_CELLS;
        let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32 - PARTICLE_ESCAPE_MARGIN_BOTTOM_CELLS;
        let max_cell_y =
            (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1 + PARTICLE_ESCAPE_MARGIN_TOP_CELLS;

        let min_world = cell_to_world_center(IVec2::new(min_cell_x, min_cell_y))
            - Vec2::splat(CELL_SIZE_M * 0.5);
        let max_world = cell_to_world_center(IVec2::new(max_cell_x, max_cell_y))
            + Vec2::splat(CELL_SIZE_M * 0.5);

        let old_count = self.particle_count();
        let mut keep = vec![true; old_count];
        let mut removed_count = 0usize;
        for (index, position) in self.pos.iter().enumerate() {
            let out_of_bounds = !position.is_finite()
                || position.x < min_world.x
                || position.x > max_world.x
                || position.y < min_world.y
                || position.y > max_world.y;
            if out_of_bounds {
                keep[index] = false;
                removed_count += 1;
            }
        }
        if removed_count == 0 {
            return;
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
        let mut new_activity_state = Vec::with_capacity(new_count);
        let mut new_sleep_candidate_frames = Vec::with_capacity(new_count);
        let mut new_active_hold_frames = Vec::with_capacity(new_count);
        let mut new_sleep_lock_frames = Vec::with_capacity(new_count);
        let mut new_pending_wake = Vec::with_capacity(new_count);
        for old_index in 0..old_count {
            if !keep[old_index] {
                continue;
            }
            new_pos.push(self.pos[old_index]);
            new_prev_pos.push(self.prev_pos[old_index]);
            new_vel.push(self.vel[old_index]);
            new_mass.push(self.mass[old_index]);
            new_material.push(self.material[old_index]);
            new_activity_state.push(self.activity_state[old_index]);
            new_sleep_candidate_frames.push(self.sleep_candidate_frames[old_index]);
            new_active_hold_frames.push(self.active_hold_frames[old_index]);
            new_sleep_lock_frames.push(self.sleep_lock_frames[old_index]);
            new_pending_wake.push(self.pending_wake[old_index]);
        }

        self.pos = new_pos;
        self.prev_pos = new_prev_pos;
        self.vel = new_vel;
        self.mass = new_mass;
        self.material = new_material;
        self.activity_state = new_activity_state;
        self.sleep_candidate_frames = new_sleep_candidate_frames;
        self.active_hold_frames = new_active_hold_frames;
        self.sleep_lock_frames = new_sleep_lock_frames;
        self.pending_wake = new_pending_wake;
        object_world.apply_particle_remap(&old_to_new, self.masses());
        self.resize_work_buffers();
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
        self.activity_state
            .resize(count, ParticleActivityState::Active);
        self.sleep_candidate_frames.resize(count, 0);
        self.active_hold_frames.resize(count, 0);
        self.sleep_lock_frames.resize(count, 0);
        self.pending_wake.resize(count, false);
        self.neighbor_cache.resize_with(count, Vec::new);
        self.viscosity_work.resize(count, Vec2::ZERO);
    }

    fn terrain_fracture_seed_velocity(&self, fracture_cell: IVec2) -> Vec2 {
        self.pending_terrain_fracture_seeds
            .get(&fracture_cell)
            .copied()
            .map(TerrainFractureSeed::velocity)
            .unwrap_or(Vec2::ZERO)
    }

    fn collect_detached_terrain_components(
        &self,
        terrain: &TerrainWorld,
        removed_cells: &HashSet<IVec2>,
    ) -> Vec<Vec<IVec2>> {
        let mut checked = HashSet::new();
        let mut detached_components = Vec::<Vec<IVec2>>::new();
        for &removed_cell in removed_cells {
            for offset in FOUR_NEIGHBOR_OFFSETS {
                let start = removed_cell + offset;
                if checked.contains(&start)
                    || !matches!(
                        terrain.get_loaded_cell_or_empty(start),
                        TerrainCell::Solid { .. }
                    )
                {
                    continue;
                }
                let fill = flood_fill_4_limited(start, DETACH_FLOOD_FILL_MAX_CELLS, |cell| {
                    matches!(
                        terrain.get_loaded_cell_or_empty(cell),
                        TerrainCell::Solid { .. }
                    )
                });
                checked.extend(fill.cells.iter().copied());
                if fill.reached_limit {
                    continue;
                }
                let mut cells = fill.cells.into_iter().collect::<Vec<_>>();
                cells.sort_by_key(|cell| (cell.y, cell.x));
                detached_components.push(cells);
            }
        }
        detached_components
    }

    fn evaluate_terrain_persistent_load(
        &mut self,
        terrain: &TerrainWorld,
        terrain_load_metric: &HashMap<IVec2, f32>,
    ) {
        if TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS == 0 {
            return;
        }
        if self.terrain_load_substep_counter % TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS as u64 != 0 {
            return;
        }
        let dt_sample = (FIXED_DT / SUBSTEPS as f32) * TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS as f32;
        let mut seen_cells = HashSet::new();

        for (&cell, &metric) in terrain_load_metric {
            if !matches!(
                terrain.get_loaded_cell_or_empty(cell),
                TerrainCell::Solid { .. }
            ) {
                self.terrain_persistent_load.remove(&cell);
                continue;
            }
            seen_cells.insert(cell);
            let mut should_wake = false;
            let mut should_fracture = false;
            {
                let state = self.terrain_persistent_load.entry(cell).or_default();
                state.strain_metric = metric;
                if metric >= TERRAIN_LOAD_STRAIN_THRESHOLD {
                    state.sustained_seconds += dt_sample;
                    should_wake = true;
                    should_fracture =
                        state.sustained_seconds >= TERRAIN_LOAD_BREAK_DURATION_SECONDS;
                } else {
                    state.sustained_seconds = (state.sustained_seconds - dt_sample).max(0.0);
                }
            }
            if should_wake {
                self.request_wake_near(cell_to_world_center(cell), WAKE_RADIUS);
            }
            if should_fracture {
                self.pending_terrain_fractures.insert(cell);
            }
        }

        let cells: Vec<_> = self.terrain_persistent_load.keys().copied().collect();
        for cell in cells {
            if seen_cells.contains(&cell) {
                continue;
            }
            if !matches!(
                terrain.get_loaded_cell_or_empty(cell),
                TerrainCell::Solid { .. }
            ) {
                self.terrain_persistent_load.remove(&cell);
                continue;
            }
            if let Some(state) = self.terrain_persistent_load.get_mut(&cell) {
                state.strain_metric *= TERRAIN_LOAD_DECAY_PER_SAMPLE;
                state.sustained_seconds = (state.sustained_seconds - dt_sample).max(0.0);
                if state.sustained_seconds <= 0.0 && state.strain_metric <= 1e-4 {
                    self.terrain_persistent_load.remove(&cell);
                }
            }
        }
    }

    fn build_terrain_detach_spawn_plan(
        &self,
        component_cells: &[IVec2],
        terrain: &TerrainWorld,
        fracture_cell_set: &HashSet<IVec2>,
    ) -> TerrainDetachSpawnPlan {
        let mut plan_cells = Vec::with_capacity(component_cells.len());
        let mut cell_materials = Vec::with_capacity(component_cells.len());
        for &cell in component_cells {
            let TerrainCell::Solid { material, .. } = terrain.get_loaded_cell_or_empty(cell) else {
                continue;
            };
            let Some(particle_material) = terrain_solid_particle(material) else {
                continue;
            };
            plan_cells.push(cell);
            cell_materials.push((cell, particle_material));
        }

        if plan_cells.is_empty() {
            return TerrainDetachSpawnPlan::default();
        }

        let mut influencing_fracture_cells = HashSet::new();
        for &cell in &plan_cells {
            for offset in FOUR_NEIGHBOR_OFFSETS {
                let fracture_cell = cell + offset;
                if fracture_cell_set.contains(&fracture_cell) {
                    influencing_fracture_cells.insert(fracture_cell);
                }
            }
        }

        let mut seed_weight_sum = 0.0f32;
        let mut seed_velocity_weighted_sum = Vec2::ZERO;
        let mut seed_contact_weighted_sum = Vec2::ZERO;
        for fracture_cell in &influencing_fracture_cells {
            let Some(seed) = self
                .pending_terrain_fracture_seeds
                .get(fracture_cell)
                .copied()
            else {
                continue;
            };
            seed_weight_sum += seed.impulse_sum.max(1e-6);
            seed_velocity_weighted_sum += seed.velocity_weighted_sum;
            seed_contact_weighted_sum += seed.contact_weighted_sum;
        }

        let linear_velocity = if seed_weight_sum > 1e-6 {
            seed_velocity_weighted_sum / seed_weight_sum
        } else {
            Vec2::ZERO
        };
        let angular_origin = if seed_weight_sum > 1e-6 {
            seed_contact_weighted_sum / seed_weight_sum
        } else {
            let mut weighted_com = Vec2::ZERO;
            let mut mass_sum = 0.0f32;
            for &(cell, material) in &cell_materials {
                let mass = particle_properties(material).mass;
                weighted_com += cell_to_world_center(cell) * mass;
                mass_sum += mass;
            }
            if mass_sum > 1e-6 {
                weighted_com / mass_sum
            } else {
                cell_to_world_center(plan_cells[0])
            }
        };

        let mut angular_velocity = 0.0f32;
        if seed_weight_sum > 1e-6 {
            let mut numerator = 0.0f32;
            let mut denominator = 0.0f32;
            for fracture_cell in &influencing_fracture_cells {
                let Some(seed) = self
                    .pending_terrain_fracture_seeds
                    .get(fracture_cell)
                    .copied()
                else {
                    continue;
                };
                let weight = seed.impulse_sum.max(1e-6);
                let r = seed.contact_pos() - angular_origin;
                let rel_v = seed.velocity() - linear_velocity;
                numerator += weight * (r.x * rel_v.y - r.y * rel_v.x);
                denominator += weight * r.length_squared();
            }
            if denominator > 1e-6 {
                angular_velocity = numerator / denominator;
            }
        }

        TerrainDetachSpawnPlan {
            cells: plan_cells,
            cell_materials,
            linear_velocity,
            angular_velocity,
            angular_origin,
        }
    }

    fn spawn_terrain_detach_component(&mut self, plan: &TerrainDetachSpawnPlan) -> Vec<usize> {
        let mut indices = Vec::new();
        for &(cell, material) in &plan.cell_materials {
            let mut cell_indices = Vec::new();
            self.append_material_particles_in_cell(cell, material, Vec2::ZERO, &mut cell_indices);
            indices.extend(cell_indices);
        }

        for &index in &indices {
            let position = self.pos[index];
            let r = position - plan.angular_origin;
            let angular_velocity_vec = Vec2::new(-r.y, r.x) * plan.angular_velocity;
            let velocity = plan.linear_velocity + angular_velocity_vec;
            self.vel[index] = velocity;
            self.prev_pos[index] = self.pos[index] - velocity * FIXED_DT;
        }
        indices
    }
}

#[derive(Debug, Default)]
struct TerrainDetachSpawnPlan {
    cells: Vec<IVec2>,
    cell_materials: Vec<(IVec2, ParticleMaterial)>,
    linear_velocity: Vec2,
    angular_velocity: f32,
    angular_origin: Vec2,
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

fn pack_pair_key(i: usize, j: usize) -> u64 {
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    ((a as u64) << 32) | b as u64
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
                    + Vec2::splat(particle_properties(ParticleMaterial::WaterLiquid).radius_m),
            );
        }
    }
    positions
}

fn default_particle_mass() -> f32 {
    particle_properties(ParticleMaterial::WaterLiquid).mass
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

fn is_granular_particle(material: ParticleMaterial) -> bool {
    matches!(
        material,
        ParticleMaterial::StoneGranular
            | ParticleMaterial::SoilGranular
            | ParticleMaterial::SandGranular
    )
}

fn is_solid_particle(material: ParticleMaterial) -> bool {
    matches!(
        material,
        ParticleMaterial::StoneSolid | ParticleMaterial::SoilSolid | ParticleMaterial::SandSolid
    )
}

fn granular_contact_friction_pair_scale(
    material_a: ParticleMaterial,
    material_b: ParticleMaterial,
) -> Option<f32> {
    if is_water_particle(material_a) || is_water_particle(material_b) {
        return None;
    }
    if !(is_granular_particle(material_a) || is_granular_particle(material_b)) {
        return None;
    }
    let pair_scale = if is_granular_particle(material_a) && is_granular_particle(material_b) {
        GRANULAR_GRANULAR_CONTACT_FRICTION_BOOST
    } else if is_solid_particle(material_a) || is_solid_particle(material_b) {
        GRANULAR_SOLID_CONTACT_FRICTION_BOOST
    } else {
        1.0
    };
    Some(pair_scale)
}

fn granular_tangential_friction_push(
    contact_normal: Vec2,
    normal_push: Vec2,
    relative_velocity: Vec2,
    static_friction: f32,
    dynamic_friction: f32,
    dt_sub: f32,
) -> Vec2 {
    let normal_push_mag = normal_push.length();
    if normal_push_mag <= 1e-8 {
        return Vec2::ZERO;
    }
    let tangential_velocity =
        relative_velocity - contact_normal * relative_velocity.dot(contact_normal);
    let tangential_speed = tangential_velocity.length();
    if tangential_speed <= 1e-6 {
        return Vec2::ZERO;
    }

    let tangential_displacement = tangential_velocity * dt_sub.max(0.0);
    let tangential_displacement_mag = tangential_displacement.length();
    if tangential_displacement_mag <= 1e-8 {
        return Vec2::ZERO;
    }

    let static_limit = static_friction.max(0.0) * normal_push_mag;
    if tangential_displacement_mag <= static_limit {
        return -tangential_displacement;
    }

    let dynamic_limit = dynamic_friction.max(0.0) * normal_push_mag;
    if dynamic_limit <= 1e-8 {
        return Vec2::ZERO;
    }
    let friction_mag = dynamic_limit.min(tangential_displacement_mag);
    -tangential_displacement.normalize_or_zero() * friction_mag
}

fn spawn_jitter_offset(
    cell: IVec2,
    material: ParticleMaterial,
    slot_x: u32,
    slot_y: u32,
    max_offset: f32,
) -> Vec2 {
    if max_offset <= 0.0 {
        return Vec2::ZERO;
    }
    let base = mix_u32(
        (cell.x as u32).wrapping_mul(0x45d9_f3b)
            ^ (cell.y as u32).wrapping_mul(0x27d4_eb2d)
            ^ material_seed(material)
            ^ slot_x.wrapping_mul(0x1656_67b1)
            ^ slot_y.wrapping_mul(0x9e37_79b9),
    );
    let hx = mix_u32(base ^ 0x68bc_21eb);
    let hy = mix_u32(base ^ 0x02e5_be93);
    let fx = (hx as f32 / u32::MAX as f32) * 2.0 - 1.0;
    let fy = (hy as f32 / u32::MAX as f32) * 2.0 - 1.0;
    Vec2::new(fx, fy) * max_offset
}

fn material_seed(material: ParticleMaterial) -> u32 {
    match material {
        ParticleMaterial::WaterLiquid => 0x1111_1111,
        ParticleMaterial::StoneSolid => 0x2222_2222,
        ParticleMaterial::StoneGranular => 0x3333_3333,
        ParticleMaterial::SoilSolid => 0x4444_4444,
        ParticleMaterial::SoilGranular => 0x5555_5555,
        ParticleMaterial::SandSolid => 0x6666_6666,
        ParticleMaterial::SandGranular => 0x7777_7777,
    }
}

fn mix_u32(mut state: u32) -> u32 {
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    state
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

    fn clear_particles(particles: &mut ParticleWorld) {
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
    }

    fn clear_fixed_terrain(terrain: &mut TerrainWorld) {
        let min_cell = IVec2::new(
            WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32,
            WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32,
        );
        let max_cell = IVec2::new(
            (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1,
            (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1,
        );
        terrain.fill_rect(min_cell, max_cell, TerrainCell::Empty);
    }

    #[test]
    fn particle_transitions_to_sleep_after_threshold_frames() {
        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);
        let _ = particles.spawn_material_particles_from_cells(
            &[IVec2::new(0, 4)],
            ParticleMaterial::SandSolid,
            Vec2::ZERO,
        );
        let index = 0usize;
        particles.activity_state[index] = ParticleActivityState::Active;
        particles.active_hold_frames[index] = 0;
        particles.sleep_candidate_frames[index] = 0;
        particles.pending_wake[index] = false;
        particles.vel[index] = Vec2::ZERO;
        particles.prev_pos[index] = particles.pos[index];

        for _ in 0..SLEEP_FRAMES {
            particles.update_sleep_states();
        }

        assert_eq!(
            particles.activity_state[index],
            ParticleActivityState::Sleeping
        );
    }

    #[test]
    fn wake_event_reactivates_sleeping_particle() {
        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);
        let _ = particles.spawn_material_particles_from_cells(
            &[IVec2::new(0, 4)],
            ParticleMaterial::SandSolid,
            Vec2::ZERO,
        );
        let index = 0usize;
        particles.activity_state[index] = ParticleActivityState::Sleeping;
        particles.vel[index] = Vec2::ZERO;
        particles.prev_pos[index] = particles.pos[index];

        particles.wake_particles_in_radius(particles.pos[index], WAKE_RADIUS * 0.5);
        particles.neighbor_grid.rebuild(&particles.pos);
        particles.propagate_and_apply_wake_requests();

        assert_eq!(
            particles.activity_state[index],
            ParticleActivityState::Active
        );
        assert!(particles.active_hold_frames[index] > 0);
    }

    #[test]
    fn sleeping_particle_skips_gravity_integration() {
        let terrain = TerrainWorld::default();
        let object_field = ObjectPhysicsField::default();
        let mut objects = ObjectWorld::default();
        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);
        let _ = particles.spawn_material_particles_from_cells(
            &[IVec2::new(0, 4)],
            ParticleMaterial::SandSolid,
            Vec2::ZERO,
        );
        let index = 0usize;
        particles.activity_state[index] = ParticleActivityState::Sleeping;
        particles.vel[index] = Vec2::ZERO;
        particles.prev_pos[index] = particles.pos[index];
        particles.pending_wake[index] = false;
        let before = particles.pos[index];

        particles.step_if_running(&terrain, &object_field, &mut objects, true);

        assert!(
            particles.pos[index].distance(before) < 1e-6,
            "sleeping particle should not move while sleeping"
        );
    }

    #[test]
    fn terrain_breaks_under_sustained_load() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        clear_fixed_terrain(&mut terrain);
        let load_cell = IVec2::new(0, 0);
        assert!(terrain.set_cell(load_cell, TerrainCell::stone()));

        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);
        let mut objects = ObjectWorld::default();
        let samples_needed = (TERRAIN_LOAD_BREAK_DURATION_SECONDS
            / ((FIXED_DT / SUBSTEPS as f32) * TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS as f32))
            .ceil() as usize
            + 2;
        for _ in 0..samples_needed {
            particles.terrain_load_substep_counter += TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS as u64;
            let mut metric = HashMap::new();
            metric.insert(load_cell, TERRAIN_LOAD_STRAIN_THRESHOLD + 0.2);
            particles.evaluate_terrain_persistent_load(&terrain, &metric);
            if particles.apply_pending_terrain_fractures(&mut terrain, &mut objects) {
                break;
            }
        }

        assert!(
            matches!(
                terrain.get_loaded_cell_or_empty(load_cell),
                TerrainCell::Empty
            ),
            "terrain should fracture after sustained load duration"
        );
    }

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

    #[test]
    fn solid_object_breaks_on_terrain_impact() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);

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

        let indices = particles.spawn_material_particles_from_cells(
            &[IVec2::new(0, 1)],
            ParticleMaterial::SandSolid,
            Vec2::ZERO,
        );
        let index = indices[0];
        particles.pos[index] = Vec2::new(0.0, 0.05);
        particles.prev_pos[index] = particles.pos[index];
        particles.vel[index] = Vec2::new(0.0, -10.0);

        let mut objects = ObjectWorld::default();
        let _ = objects
            .create_object(
                indices,
                particles.positions(),
                particles.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            )
            .expect("object should be created");
        let mut object_field = ObjectPhysicsField::default();
        objects.update_physics_field(particles.positions(), particles.masses(), &mut object_field);

        particles.step_if_running(&terrain, &object_field, &mut objects, true);

        assert!(
            objects.objects().is_empty(),
            "object should fracture on terrain impact"
        );
        assert!(
            particles
                .materials()
                .iter()
                .all(|&m| !matches!(m, ParticleMaterial::SandSolid)),
            "fractured particles should no longer remain solid"
        );
    }

    #[test]
    fn object_fracture_detaches_only_target_particles() {
        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);

        let indices = particles.spawn_material_particles_from_cells(
            &[IVec2::new(0, 5), IVec2::new(1, 5), IVec2::new(2, 5)],
            ParticleMaterial::SandSolid,
            Vec2::ZERO,
        );
        let left = indices[0];
        let middle = indices[1];
        let right = indices[2];

        let mut objects = ObjectWorld::default();
        let object_id = objects
            .create_object(
                indices.clone(),
                particles.positions(),
                particles.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            )
            .expect("object should be created");

        particles.pending_object_fractures.insert(object_id);
        particles
            .pending_object_fracture_particles
            .entry(object_id)
            .or_default()
            .insert(middle);
        particles.apply_object_fractures(&mut objects);

        assert_eq!(particles.material[left], ParticleMaterial::SandGranular);
        assert_eq!(particles.material[right], ParticleMaterial::SandGranular);
        assert_eq!(particles.material[middle], ParticleMaterial::SandGranular);
        assert!(objects.objects().is_empty());
    }

    #[test]
    fn terrain_small_component_detaches_into_object() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        clear_fixed_terrain(&mut terrain);

        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);
        let mut objects = ObjectWorld::default();

        let fracture_cell = IVec2::new(0, 20);
        let detached_cell = IVec2::new(1, 20);
        assert!(terrain.set_cell(fracture_cell, TerrainCell::stone()));
        assert!(terrain.set_cell(detached_cell, TerrainCell::stone()));
        let impact_velocity = Vec2::new(2.0, 0.5);
        let impact_pos = cell_to_world_center(fracture_cell);

        particles.pending_terrain_fractures.insert(fracture_cell);
        particles.pending_terrain_fracture_seeds.insert(
            fracture_cell,
            TerrainFractureSeed {
                impulse_sum: 10.0,
                velocity_weighted_sum: impact_velocity * 10.0,
                contact_weighted_sum: impact_pos * 10.0,
            },
        );

        let changed = particles.apply_pending_terrain_fractures(&mut terrain, &mut objects);
        assert!(changed);
        assert!(matches!(
            terrain.get_loaded_cell_or_empty(fracture_cell),
            TerrainCell::Empty
        ));
        assert!(matches!(
            terrain.get_loaded_cell_or_empty(detached_cell),
            TerrainCell::Empty
        ));
        assert!(
            objects.objects().is_empty(),
            "single-cell detached object should be auto-fractured"
        );
        assert!(
            particles
                .material
                .iter()
                .any(|&m| matches!(m, ParticleMaterial::StoneGranular)),
            "fracture cell should spawn granular particles"
        );
    }

    #[test]
    fn terrain_large_component_stays_in_terrain() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        clear_fixed_terrain(&mut terrain);

        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);
        let mut objects = ObjectWorld::default();

        let base = IVec2::new(-6, 30);
        let width = 12;
        let height = 11;
        for y in 0..height {
            for x in 0..width {
                let cell = base + IVec2::new(x, y);
                assert!(terrain.set_cell(cell, TerrainCell::stone()));
            }
        }

        let fracture_cell = base + IVec2::new(0, 0);
        particles.pending_terrain_fractures.insert(fracture_cell);
        particles.pending_terrain_fracture_seeds.insert(
            fracture_cell,
            TerrainFractureSeed {
                impulse_sum: 8.0,
                velocity_weighted_sum: Vec2::new(1.0, 0.0) * 8.0,
                contact_weighted_sum: cell_to_world_center(fracture_cell) * 8.0,
            },
        );

        let changed = particles.apply_pending_terrain_fractures(&mut terrain, &mut objects);
        assert!(changed);
        assert!(matches!(
            terrain.get_loaded_cell_or_empty(fracture_cell),
            TerrainCell::Empty
        ));
        let remaining_cell = base + IVec2::new(width - 1, height - 1);
        assert!(matches!(
            terrain.get_loaded_cell_or_empty(remaining_cell),
            TerrainCell::Solid { .. }
        ));
        assert!(objects.objects().is_empty());
        assert!(
            particles
                .material
                .iter()
                .all(|&m| !matches!(m, ParticleMaterial::StoneSolid)),
            "large terrain component should not be detached into object particles"
        );
    }

    #[test]
    fn edit_removed_cells_detach_small_terrain_component_into_object() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        clear_fixed_terrain(&mut terrain);

        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);
        let mut objects = ObjectWorld::default();

        let removed_cell = IVec2::new(0, 20);
        let detached_cells = [IVec2::new(1, 20), IVec2::new(2, 20)];
        for &cell in &detached_cells {
            assert!(terrain.set_cell(cell, TerrainCell::stone()));
        }

        let removed = HashSet::from([removed_cell]);
        let changed = particles.detach_terrain_components_after_cell_removal(
            &mut terrain,
            &mut objects,
            &removed,
        );
        assert!(changed);
        for &cell in &detached_cells {
            assert!(matches!(
                terrain.get_loaded_cell_or_empty(cell),
                TerrainCell::Empty
            ));
        }
        assert_eq!(objects.objects().len(), 1);
        assert_eq!(
            objects.objects()[0].particle_indices.len(),
            detached_cells.len()
        );
    }

    #[test]
    fn single_cell_object_is_auto_fractured_on_postprocess() {
        let mut particles = ParticleWorld::default();
        clear_particles(&mut particles);
        let indices = particles.spawn_material_particles_from_cells(
            &[IVec2::new(0, 4)],
            ParticleMaterial::SandSolid,
            Vec2::ZERO,
        );
        let mut objects = ObjectWorld::default();
        let _ = objects
            .create_object(
                indices,
                particles.positions(),
                particles.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            )
            .expect("object should be created");

        particles.postprocess_objects_after_topology_edit(&mut objects);

        assert!(objects.objects().is_empty());
        assert!(
            particles
                .materials()
                .iter()
                .any(|&m| matches!(m, ParticleMaterial::SandGranular)),
            "single-cell object should be converted to granular particles"
        );
    }
}
