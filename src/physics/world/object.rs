use std::collections::{HashMap, HashSet};

use bevy::log::tracing;
use bevy::prelude::*;

use super::terrain::{
    CELL_SIZE_M, CHUNK_WORLD_SIZE_M, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X,
    WORLD_MIN_CHUNK_Y, world_to_cell,
};
use crate::physics::connectivity::connected_components_4;
use crate::physics::solver::params_defaults::DEFAULT_SOLVER_PARAMS;

pub const OBJECT_SHAPE_STIFFNESS_ALPHA: f32 = DEFAULT_SOLVER_PARAMS.object_shape_stiffness_alpha;
pub const OBJECT_SHAPE_ITERS: usize = DEFAULT_SOLVER_PARAMS.object_shape_iters;
pub const OBJECT_LOCAL_SDF_SAMPLES_PER_CELL: i32 =
    DEFAULT_SOLVER_PARAMS.object_local_sdf_samples_per_cell;
pub const OBJECT_PHYSICS_SDF_CELL_SIZE_M: f32 = DEFAULT_SOLVER_PARAMS.object_physics_sdf_cell_size_m;
pub const OBJECT_BROADPHASE_CELL_SIZE_M: f32 = DEFAULT_SOLVER_PARAMS.object_broadphase_cell_size_m;
pub const OBJECT_SDF_MAX_DISTANCE_M: f32 = DEFAULT_SOLVER_PARAMS.object_sdf_max_distance_m;
pub const OBJECT_SDF_MAX_SPLATS_PER_CELL: usize = DEFAULT_SOLVER_PARAMS.object_sdf_max_splats_per_cell;
pub const OBJECT_SDF_MAX_CONTACTS_PER_QUERY: usize =
    DEFAULT_SOLVER_PARAMS.object_sdf_max_contacts_per_query;

pub type ObjectId = u32;

const SDF_INF: f32 = 1.0e9;
const SDF_DIAGONAL_COST: f32 = std::f32::consts::SQRT_2;

#[derive(Debug, Clone, Copy, Default)]
pub struct Aabb2 {
    pub min: Vec2,
    pub max: Vec2,
}

impl Aabb2 {
    fn expanded(self, radius: f32) -> Self {
        Self {
            min: self.min - Vec2::splat(radius),
            max: self.max + Vec2::splat(radius),
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: Vec2::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y)),
            max: Vec2::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y)),
        }
    }

    fn contains(self, point: Vec2) -> bool {
        (self.min.x..=self.max.x).contains(&point.x) && (self.min.y..=self.max.y).contains(&point.y)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ObjectLocalSdf {
    samples: Vec<f32>,
    width: usize,
    height: usize,
    spacing_m: f32,
    origin_m: Vec2,
}

impl ObjectLocalSdf {
    fn sample_distance_and_normal(&self, local_pos: Vec2) -> Option<(f32, Vec2)> {
        let d = self.sample_distance(local_pos)?;
        let eps = self.spacing_m.max(1e-4);
        let dx = self
            .sample_distance(local_pos + Vec2::new(eps, 0.0))
            .unwrap_or(d)
            - self
                .sample_distance(local_pos - Vec2::new(eps, 0.0))
                .unwrap_or(d);
        let dy = self
            .sample_distance(local_pos + Vec2::new(0.0, eps))
            .unwrap_or(d)
            - self
                .sample_distance(local_pos - Vec2::new(0.0, eps))
                .unwrap_or(d);
        let normal = Vec2::new(dx, dy).normalize_or_zero();
        Some((
            d,
            if normal == Vec2::ZERO {
                Vec2::Y
            } else {
                normal
            },
        ))
    }

    fn sample_distance(&self, local_pos: Vec2) -> Option<f32> {
        if self.samples.is_empty() || self.width == 0 || self.height == 0 {
            return None;
        }

        let width_max = (self.width.saturating_sub(1)) as f32;
        let height_max = (self.height.saturating_sub(1)) as f32;
        let fx_raw = (local_pos.x - self.origin_m.x) / self.spacing_m;
        let fy_raw = (local_pos.y - self.origin_m.y) / self.spacing_m;

        let outside_x = if fx_raw < 0.0 {
            -fx_raw * self.spacing_m
        } else if fx_raw > width_max {
            (fx_raw - width_max) * self.spacing_m
        } else {
            0.0
        };
        let outside_y = if fy_raw < 0.0 {
            -fy_raw * self.spacing_m
        } else if fy_raw > height_max {
            (fy_raw - height_max) * self.spacing_m
        } else {
            0.0
        };

        let fx = fx_raw.clamp(0.0, width_max);
        let fy = fy_raw.clamp(0.0, height_max);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let d00 = self.samples[sdf_index(self.width, x0, y0)];
        let d10 = self.samples[sdf_index(self.width, x1, y0)];
        let d01 = self.samples[sdf_index(self.width, x0, y1)];
        let d11 = self.samples[sdf_index(self.width, x1, y1)];
        let dx0 = d00 + (d10 - d00) * tx;
        let dx1 = d01 + (d11 - d01) * tx;
        let d = dx0 + (dx1 - dx0) * ty;

        Some(d + Vec2::new(outside_x, outside_y).length())
    }
}

#[derive(Debug, Clone)]
pub struct ObjectData {
    pub id: ObjectId,
    pub particle_indices: Vec<usize>,
    pub rest_local: Vec<Vec2>,
    pub mass_sum: f32,
    pub shape_stiffness_alpha: f32,
    pub shape_iters: usize,
    pub shape_dirty: bool,
    pub physics_dirty: bool,
    pub local_sdf: ObjectLocalSdf,
    pub pose_center: Vec2,
    pub pose_theta: f32,
    pub prev_pose_center: Vec2,
    pub prev_pose_theta: f32,
    pub aabb_world: Aabb2,
    pub prev_aabb_world: Aabb2,
    pub pose_initialized: bool,
}

#[derive(Debug, Clone)]
pub struct ObjectSnapshotData {
    pub id: ObjectId,
    pub particle_indices: Vec<usize>,
    pub rest_local: Vec<Vec2>,
    pub shape_stiffness_alpha: f32,
    pub shape_iters: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ObjectPhysicsSdfSample {
    pub distance_m: f32,
    pub normal_world: Vec2,
}

#[derive(Clone, Copy, Debug)]
struct ObjectSampleCell {
    count: u8,
    ids: [ObjectId; OBJECT_SDF_MAX_SPLATS_PER_CELL],
}

impl Default for ObjectSampleCell {
    fn default() -> Self {
        Self {
            count: 0,
            ids: [0; OBJECT_SDF_MAX_SPLATS_PER_CELL],
        }
    }
}

#[derive(Resource, Debug)]
pub struct ObjectPhysicsField {
    sample_cell_size_m: f32,
    sample_grid_min: IVec2,
    sample_grid_size: UVec2,
    sample_cells: Vec<ObjectSampleCell>,
    sample_visit_marks: Vec<bool>,
    sample_visit_list: Vec<usize>,
    broadphase_cell_size_m: f32,
    broadphase_grid_min: IVec2,
    broadphase_grid_size: UVec2,
    broadphase_cells: Vec<Vec<ObjectId>>,
    max_distance_m: f32,
}

impl Default for ObjectPhysicsField {
    fn default() -> Self {
        let sample_cell_size_m = OBJECT_PHYSICS_SDF_CELL_SIZE_M;
        let broadphase_cell_size_m = OBJECT_BROADPHASE_CELL_SIZE_M;
        let max_distance_m = OBJECT_SDF_MAX_DISTANCE_M;
        let sample_padding = max_distance_m + sample_cell_size_m * 2.0;
        let broadphase_padding = max_distance_m + broadphase_cell_size_m * 2.0;
        let (sample_grid_min, sample_grid_size) =
            build_grid_layout(sample_cell_size_m, sample_padding);
        let (broadphase_grid_min, broadphase_grid_size) =
            build_grid_layout(broadphase_cell_size_m, broadphase_padding);
        let sample_len = grid_len(sample_grid_size);
        let broadphase_len = grid_len(broadphase_grid_size);

        Self {
            sample_cell_size_m,
            sample_grid_min,
            sample_grid_size,
            sample_cells: vec![ObjectSampleCell::default(); sample_len],
            sample_visit_marks: vec![false; sample_len],
            sample_visit_list: Vec::new(),
            broadphase_cell_size_m,
            broadphase_grid_min,
            broadphase_grid_size,
            broadphase_cells: vec![Vec::new(); broadphase_len],
            max_distance_m,
        }
    }
}

impl ObjectPhysicsField {
    pub fn clear(&mut self) {
        for sample_cell in &mut self.sample_cells {
            sample_cell.count = 0;
        }
        self.sample_visit_marks.fill(false);
        self.sample_visit_list.clear();
        for bucket in &mut self.broadphase_cells {
            bucket.clear();
        }
    }

    pub fn gather_candidate_object_ids(&self, world_pos: Vec2, out: &mut Vec<ObjectId>) {
        out.clear();
        let center = sample_grid_cell(world_pos, self.sample_cell_size_m);
        for y in (center.y - 1)..=(center.y + 1) {
            for x in (center.x - 1)..=(center.x + 1) {
                let cell = IVec2::new(x, y);
                let Some(index) =
                    grid_cell_index(cell, self.sample_grid_min, self.sample_grid_size)
                else {
                    continue;
                };
                let sample_cell = &self.sample_cells[index];
                for slot in 0..sample_cell.count as usize {
                    let object_id = sample_cell.ids[slot];
                    if out.contains(&object_id) {
                        continue;
                    }
                    out.push(object_id);
                    if out.len() >= OBJECT_SDF_MAX_CONTACTS_PER_QUERY {
                        return;
                    }
                }
            }
        }
    }

    fn rebuild_broadphase(&mut self, objects: &[ObjectData]) {
        let _span = tracing::info_span!("physics::object_rebuild_broadphase").entered();
        for bucket in &mut self.broadphase_cells {
            bucket.clear();
        }
        for object in objects {
            if !object.pose_initialized {
                continue;
            }
            let expanded = object.aabb_world.expanded(self.max_distance_m);
            let min_cell = broadphase_grid_cell(expanded.min, self.broadphase_cell_size_m);
            let max_cell = broadphase_grid_cell(expanded.max, self.broadphase_cell_size_m);
            let Some((min_cell, max_cell)) = clamp_grid_range(
                min_cell,
                max_cell,
                self.broadphase_grid_min,
                self.broadphase_grid_size,
            ) else {
                continue;
            };
            for y in min_cell.y..=max_cell.y {
                for x in min_cell.x..=max_cell.x {
                    let cell = IVec2::new(x, y);
                    let Some(index) =
                        grid_cell_index(cell, self.broadphase_grid_min, self.broadphase_grid_size)
                    else {
                        continue;
                    };
                    self.broadphase_cells[index].push(object.id);
                }
            }
        }
    }

    fn reproject_dirty_objects(&mut self, objects: &[ObjectData], dirty_ids: &HashSet<ObjectId>) {
        let _span = tracing::info_span!("physics::object_reproject_dirty").entered();
        if dirty_ids.is_empty() {
            return;
        }

        let object_by_id: HashMap<ObjectId, &ObjectData> =
            objects.iter().map(|object| (object.id, object)).collect();
        self.sample_visit_list.clear();
        for &object_id in dirty_ids {
            let Some(object) = object_by_id.get(&object_id) else {
                continue;
            };
            let union = object
                .aabb_world
                .union(object.prev_aabb_world)
                .expanded(self.max_distance_m + self.sample_cell_size_m);
            let min_cell = sample_grid_cell(union.min, self.sample_cell_size_m);
            let max_cell = sample_grid_cell(union.max, self.sample_cell_size_m);
            let Some((min_cell, max_cell)) = clamp_grid_range(
                min_cell,
                max_cell,
                self.sample_grid_min,
                self.sample_grid_size,
            ) else {
                continue;
            };
            for y in min_cell.y..=max_cell.y {
                for x in min_cell.x..=max_cell.x {
                    let cell = IVec2::new(x, y);
                    let Some(index) =
                        grid_cell_index(cell, self.sample_grid_min, self.sample_grid_size)
                    else {
                        continue;
                    };
                    if self.sample_visit_marks[index] {
                        continue;
                    }
                    self.sample_visit_marks[index] = true;
                    self.sample_visit_list.push(index);
                }
            }
        }

        let mut candidates = Vec::new();
        let mut ordered_candidates = Vec::new();
        for &sample_index in &self.sample_visit_list {
            let cell =
                grid_index_to_cell(sample_index, self.sample_grid_min, self.sample_grid_size);
            let world_pos = sample_cell_center(cell, self.sample_cell_size_m);
            self.collect_broadphase_candidates(world_pos, &mut candidates);

            ordered_candidates.clear();
            for &object_id in &candidates {
                let Some(object) = object_by_id.get(&object_id) else {
                    continue;
                };
                if !object
                    .aabb_world
                    .expanded(self.max_distance_m + self.sample_cell_size_m)
                    .contains(world_pos)
                {
                    continue;
                }

                let local = world_to_local(world_pos, object.pose_center, object.pose_theta);
                let Some(distance_m) = object.local_sdf.sample_distance(local) else {
                    continue;
                };
                if distance_m <= self.max_distance_m {
                    ordered_candidates.push((object.rest_local.len(), object_id));
                }
            }
            ordered_candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            ordered_candidates.truncate(OBJECT_SDF_MAX_SPLATS_PER_CELL);

            let sample_cell = &mut self.sample_cells[sample_index];
            sample_cell.count = 0;
            for (slot, (_, object_id)) in ordered_candidates.iter().copied().enumerate() {
                sample_cell.ids[slot] = object_id;
                sample_cell.count += 1;
            }
        }

        for &sample_index in &self.sample_visit_list {
            self.sample_visit_marks[sample_index] = false;
        }
        self.sample_visit_list.clear();
    }

    fn collect_broadphase_candidates(&self, world_pos: Vec2, out: &mut Vec<ObjectId>) {
        out.clear();
        let center = broadphase_grid_cell(world_pos, self.broadphase_cell_size_m);
        for y in (center.y - 1)..=(center.y + 1) {
            for x in (center.x - 1)..=(center.x + 1) {
                let cell = IVec2::new(x, y);
                let Some(index) =
                    grid_cell_index(cell, self.broadphase_grid_min, self.broadphase_grid_size)
                else {
                    continue;
                };
                for &object_id in &self.broadphase_cells[index] {
                    if out.contains(&object_id) {
                        continue;
                    }
                    out.push(object_id);
                }
            }
        }
    }

    fn tracked_object_ids(&self) -> HashSet<ObjectId> {
        let mut ids = HashSet::new();
        for sample_cell in &self.sample_cells {
            for slot in 0..sample_cell.count as usize {
                let object_id = sample_cell.ids[slot];
                ids.insert(object_id);
            }
        }
        for bucket in &self.broadphase_cells {
            ids.extend(bucket.iter().copied());
        }
        ids
    }
}

#[derive(Resource, Debug, Default)]
pub struct ObjectWorld {
    next_id: ObjectId,
    objects: Vec<ObjectData>,
    object_index_by_id: HashMap<ObjectId, usize>,
    reaction_impulses: HashMap<ObjectId, Vec2>,
    particle_owner: Vec<Option<ObjectId>>,
}

impl ObjectWorld {
    pub fn clear(&mut self) {
        self.objects.clear();
        self.object_index_by_id.clear();
        self.reaction_impulses.clear();
        self.particle_owner.clear();
        self.next_id = 0;
    }

    pub fn objects_mut(&mut self) -> &mut [ObjectData] {
        &mut self.objects
    }

    pub fn objects(&self) -> &[ObjectData] {
        &self.objects
    }

    pub fn clear_reaction_impulses(&mut self) {
        self.reaction_impulses.clear();
    }

    pub fn accumulate_reaction_impulse(&mut self, object_id: ObjectId, impulse: Vec2) {
        *self
            .reaction_impulses
            .entry(object_id)
            .or_insert(Vec2::ZERO) += impulse;
    }

    #[allow(dead_code)]
    pub fn reaction_impulse_of(&self, object_id: ObjectId) -> Vec2 {
        self.reaction_impulses
            .get(&object_id)
            .copied()
            .unwrap_or(Vec2::ZERO)
    }

    pub fn object_of_particle(&self, particle_index: usize) -> Option<ObjectId> {
        self.particle_owner.get(particle_index).copied().flatten()
    }

    pub fn evaluate_object_sdf(
        &self,
        object_id: ObjectId,
        world_pos: Vec2,
    ) -> Option<ObjectPhysicsSdfSample> {
        let &object_index = self.object_index_by_id.get(&object_id)?;
        let object = self.objects.get(object_index)?;
        if !object.pose_initialized {
            return None;
        }
        let local = world_to_local(world_pos, object.pose_center, object.pose_theta);
        let (distance_m, normal_local) = object.local_sdf.sample_distance_and_normal(local)?;
        let normal_world =
            rotate_local_to_world(normal_local, object.pose_theta).normalize_or_zero();
        Some(ObjectPhysicsSdfSample {
            distance_m,
            normal_world,
        })
    }

    pub fn create_object(
        &mut self,
        particle_indices: Vec<usize>,
        particle_pos: &[Vec2],
        particle_mass: &[f32],
        shape_stiffness_alpha: f32,
        shape_iters: usize,
    ) -> Option<ObjectId> {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let object = build_object_data(
            id,
            particle_indices,
            particle_pos,
            particle_mass,
            shape_stiffness_alpha,
            shape_iters,
        )?;
        self.objects.push(object);
        self.rebuild_object_index_map();
        Some(id)
    }

    pub fn split_object_after_detach(
        &mut self,
        object_id: ObjectId,
        detached_particle_indices: &HashSet<usize>,
        particle_pos: &[Vec2],
        particle_mass: &[f32],
    ) {
        let Some(&object_index) = self.object_index_by_id.get(&object_id) else {
            return;
        };
        let (shape_stiffness_alpha, shape_iters, object_particles) = {
            let object = &self.objects[object_index];
            (
                object.shape_stiffness_alpha,
                object.shape_iters,
                object.particle_indices.clone(),
            )
        };

        let remaining_particles: Vec<usize> = object_particles
            .into_iter()
            .filter(|index| !detached_particle_indices.contains(index))
            .collect();

        if remaining_particles.is_empty() {
            self.objects.remove(object_index);
            self.reaction_impulses.remove(&object_id);
            self.rebuild_object_index_map();
            return;
        }

        let mut particles_by_cell: HashMap<IVec2, Vec<usize>> = HashMap::new();
        for &particle_index in &remaining_particles {
            if particle_index >= particle_pos.len() || particle_index >= particle_mass.len() {
                continue;
            }
            particles_by_cell
                .entry(world_to_cell(particle_pos[particle_index]))
                .or_default()
                .push(particle_index);
        }
        if particles_by_cell.is_empty() {
            self.objects.remove(object_index);
            self.reaction_impulses.remove(&object_id);
            self.rebuild_object_index_map();
            return;
        }

        let occupied_cells: HashSet<IVec2> = particles_by_cell.keys().copied().collect();
        let mut components = connected_components_4(&occupied_cells);
        components.sort_by_key(|component| std::cmp::Reverse(component.len()));

        let mut component_particles = Vec::new();
        for component in components {
            let mut indices = Vec::new();
            for cell in component {
                if let Some(cell_particles) = particles_by_cell.get(&cell) {
                    indices.extend(cell_particles.iter().copied());
                }
            }
            if indices.is_empty() {
                continue;
            }
            indices.sort_unstable();
            component_particles.push(indices);
        }

        if component_particles.is_empty() {
            self.objects.remove(object_index);
            self.reaction_impulses.remove(&object_id);
            self.rebuild_object_index_map();
            return;
        }

        let largest_component = component_particles.remove(0);
        let Some(rebuilt) = build_object_data(
            object_id,
            largest_component,
            particle_pos,
            particle_mass,
            shape_stiffness_alpha,
            shape_iters,
        ) else {
            self.objects.remove(object_index);
            self.reaction_impulses.remove(&object_id);
            self.rebuild_object_index_map();
            return;
        };
        self.objects[object_index] = rebuilt;

        for particles in component_particles {
            let id = self.next_id;
            self.next_id = self.next_id.wrapping_add(1);
            let Some(object) = build_object_data(
                id,
                particles,
                particle_pos,
                particle_mass,
                shape_stiffness_alpha,
                shape_iters,
            ) else {
                continue;
            };
            self.objects.push(object);
        }

        self.rebuild_object_index_map();
    }

    pub fn split_objects_after_detach(
        &mut self,
        detached_particle_indices: &HashSet<usize>,
        particle_pos: &[Vec2],
        particle_mass: &[f32],
    ) {
        if detached_particle_indices.is_empty() {
            return;
        }
        let object_ids: Vec<_> = self
            .objects
            .iter()
            .filter(|object| {
                object
                    .particle_indices
                    .iter()
                    .any(|index| detached_particle_indices.contains(index))
            })
            .map(|object| object.id)
            .collect();
        for object_id in object_ids {
            self.split_object_after_detach(
                object_id,
                detached_particle_indices,
                particle_pos,
                particle_mass,
            );
        }
    }

    pub fn split_all_disconnected_objects(&mut self, particle_pos: &[Vec2], particle_mass: &[f32]) {
        let object_ids: Vec<_> = self.objects.iter().map(|object| object.id).collect();
        let detached = HashSet::new();
        for object_id in object_ids {
            self.split_object_after_detach(object_id, &detached, particle_pos, particle_mass);
        }
    }

    pub fn apply_particle_remap(&mut self, old_to_new: &[Option<usize>], particle_mass: &[f32]) {
        self.objects.retain_mut(|object| {
            let mut next_indices = Vec::with_capacity(object.particle_indices.len());
            let mut next_rest_local = Vec::with_capacity(object.rest_local.len());

            for (slot, &old_index) in object.particle_indices.iter().enumerate() {
                if old_index >= old_to_new.len() {
                    continue;
                }
                let Some(next_index) = old_to_new[old_index] else {
                    continue;
                };
                next_indices.push(next_index);
                next_rest_local.push(object.rest_local[slot]);
            }

            if next_indices.is_empty() {
                return false;
            }

            object.particle_indices = next_indices;
            object.rest_local = next_rest_local;
            recenter_rest_local(
                &mut object.rest_local,
                &object.particle_indices,
                particle_mass,
            );
            object.local_sdf = build_local_sdf(&object.rest_local);
            object.mass_sum = object
                .particle_indices
                .iter()
                .map(|&index| particle_mass[index])
                .sum::<f32>()
                .max(1e-6);
            object.shape_dirty = true;
            object.physics_dirty = true;
            true
        });
        self.rebuild_object_index_map();
    }

    pub fn remove_objects_by_ids(&mut self, ids: &HashSet<ObjectId>) {
        if ids.is_empty() {
            return;
        }
        self.objects.retain(|object| !ids.contains(&object.id));
        for &id in ids {
            self.reaction_impulses.remove(&id);
        }
        self.rebuild_object_index_map();
    }

    pub fn snapshot_data(&self) -> Vec<ObjectSnapshotData> {
        self.objects
            .iter()
            .map(|object| ObjectSnapshotData {
                id: object.id,
                particle_indices: object.particle_indices.clone(),
                rest_local: object.rest_local.clone(),
                shape_stiffness_alpha: object.shape_stiffness_alpha,
                shape_iters: object.shape_iters,
            })
            .collect()
    }

    pub fn restore_from_snapshot_data(
        &mut self,
        snapshots: &[ObjectSnapshotData],
        particle_pos: &[Vec2],
        particle_mass: &[f32],
    ) -> Result<(), String> {
        self.clear();
        let mut seen_ids = HashSet::new();
        let mut max_id = 0u32;

        for snapshot in snapshots {
            if !seen_ids.insert(snapshot.id) {
                return Err(format!("duplicate object id: {}", snapshot.id));
            }
            if snapshot.particle_indices.is_empty() {
                return Err(format!("object {} has no particles", snapshot.id));
            }
            if snapshot.particle_indices.len() != snapshot.rest_local.len() {
                return Err(format!(
                    "object {} has mismatched particle/rest counts",
                    snapshot.id
                ));
            }
            for &index in &snapshot.particle_indices {
                if index >= particle_pos.len() || index >= particle_mass.len() {
                    return Err(format!(
                        "object {} references out-of-range particle index {}",
                        snapshot.id, index
                    ));
                }
            }

            let mass_sum = snapshot
                .particle_indices
                .iter()
                .map(|&index| particle_mass[index])
                .sum::<f32>()
                .max(1e-6);
            let local_sdf = build_local_sdf(&snapshot.rest_local);
            self.objects.push(ObjectData {
                id: snapshot.id,
                particle_indices: snapshot.particle_indices.clone(),
                rest_local: snapshot.rest_local.clone(),
                mass_sum,
                shape_stiffness_alpha: snapshot.shape_stiffness_alpha.clamp(0.0, 1.0),
                shape_iters: snapshot.shape_iters.max(1),
                shape_dirty: true,
                physics_dirty: true,
                local_sdf,
                pose_center: Vec2::ZERO,
                pose_theta: 0.0,
                prev_pose_center: Vec2::ZERO,
                prev_pose_theta: 0.0,
                aabb_world: Aabb2::default(),
                prev_aabb_world: Aabb2::default(),
                pose_initialized: false,
            });
            max_id = max_id.max(snapshot.id);
        }

        self.next_id = max_id.wrapping_add(1);
        self.rebuild_object_index_map();
        Ok(())
    }

    pub fn update_physics_field(
        &mut self,
        particle_pos: &[Vec2],
        particle_mass: &[f32],
        field: &mut ObjectPhysicsField,
    ) {
        let _span = tracing::info_span!("physics::update_object_physics_field").entered();
        self.clear_reaction_impulses();
        {
            let _span = tracing::info_span!("physics::object_rebuild_index").entered();
            self.rebuild_object_index_map();
        }
        {
            let _span = tracing::info_span!("physics::object_rebuild_owner_map").entered();
            self.rebuild_particle_owner_map(particle_pos.len());
        }
        let mut dirty_ids = HashSet::new();
        let previous_ids = {
            let _span = tracing::info_span!("physics::object_tracked_ids").entered();
            field.tracked_object_ids()
        };
        {
            let _span = tracing::info_span!("physics::object_pose_update").entered();
            for object in &mut self.objects {
                let Some((center, theta, mass_sum, aabb)) =
                    compute_object_pose_and_aabb(object, particle_pos, particle_mass)
                else {
                    continue;
                };
                object.mass_sum = mass_sum;

                if !object.pose_initialized {
                    object.pose_center = center;
                    object.pose_theta = theta;
                    object.prev_pose_center = center;
                    object.prev_pose_theta = theta;
                    object.aabb_world = aabb;
                    object.prev_aabb_world = aabb;
                    object.pose_initialized = true;
                    dirty_ids.insert(object.id);
                    continue;
                }

                object.prev_pose_center = object.pose_center;
                object.prev_pose_theta = object.pose_theta;
                object.prev_aabb_world = object.aabb_world;
                object.pose_center = center;
                object.pose_theta = theta;
                object.aabb_world = aabb;
                let moved = object.pose_center.distance(object.prev_pose_center) > 1e-4
                    || shortest_angle_delta(object.pose_theta, object.prev_pose_theta).abs() > 1e-4;
                if moved || object.physics_dirty {
                    dirty_ids.insert(object.id);
                }
            }
        }

        let current_ids: HashSet<ObjectId> = self.objects.iter().map(|object| object.id).collect();
        let has_removed_object = previous_ids.iter().any(|id| !current_ids.contains(id));
        if has_removed_object {
            let _span = tracing::info_span!("physics::object_field_clear_for_removed").entered();
            field.clear();
            dirty_ids.extend(current_ids.iter().copied());
        }

        {
            let _span = tracing::info_span!("physics::object_field_broadphase").entered();
            field.rebuild_broadphase(&self.objects);
        }
        {
            let _span = tracing::info_span!("physics::object_field_reproject").entered();
            field.reproject_dirty_objects(&self.objects, &dirty_ids);
        }
        {
            let _span = tracing::info_span!("physics::object_clear_dirty_flags").entered();
            for object in &mut self.objects {
                if dirty_ids.contains(&object.id) {
                    object.physics_dirty = false;
                }
            }
        }
    }

    fn rebuild_particle_owner_map(&mut self, particle_count: usize) {
        self.particle_owner.clear();
        self.particle_owner.resize(particle_count, None);
        for object in &self.objects {
            for &particle_index in &object.particle_indices {
                if particle_index < particle_count {
                    self.particle_owner[particle_index] = Some(object.id);
                }
            }
        }
    }

    fn rebuild_object_index_map(&mut self) {
        self.object_index_by_id.clear();
        for (index, object) in self.objects.iter().enumerate() {
            self.object_index_by_id.insert(object.id, index);
        }
    }
}

fn build_object_data(
    id: ObjectId,
    particle_indices: Vec<usize>,
    particle_pos: &[Vec2],
    particle_mass: &[f32],
    shape_stiffness_alpha: f32,
    shape_iters: usize,
) -> Option<ObjectData> {
    if particle_indices.is_empty() {
        return None;
    }

    let mut mass_sum = 0.0;
    let mut com = Vec2::ZERO;
    let mut valid_indices = Vec::with_capacity(particle_indices.len());
    for &index in &particle_indices {
        if index >= particle_pos.len() || index >= particle_mass.len() {
            continue;
        }
        let mass = particle_mass[index];
        mass_sum += mass;
        com += particle_pos[index] * mass;
        valid_indices.push(index);
    }
    if mass_sum <= 1e-6 || valid_indices.is_empty() {
        return None;
    }
    com /= mass_sum;

    let mut rest_local = Vec::with_capacity(valid_indices.len());
    for &index in &valid_indices {
        rest_local.push(particle_pos[index] - com);
    }
    recenter_rest_local(&mut rest_local, &valid_indices, particle_mass);
    let local_sdf = build_local_sdf(&rest_local);

    Some(ObjectData {
        id,
        particle_indices: valid_indices,
        rest_local,
        mass_sum,
        shape_stiffness_alpha: shape_stiffness_alpha.clamp(0.0, 1.0),
        shape_iters: shape_iters.max(1),
        shape_dirty: true,
        physics_dirty: true,
        local_sdf,
        pose_center: Vec2::ZERO,
        pose_theta: 0.0,
        prev_pose_center: Vec2::ZERO,
        prev_pose_theta: 0.0,
        aabb_world: Aabb2::default(),
        prev_aabb_world: Aabb2::default(),
        pose_initialized: false,
    })
}

fn compute_object_pose_and_aabb(
    object: &ObjectData,
    particle_pos: &[Vec2],
    particle_mass: &[f32],
) -> Option<(Vec2, f32, f32, Aabb2)> {
    if object.particle_indices.is_empty() || object.rest_local.is_empty() {
        return None;
    }

    let mut mass_sum = 0.0;
    let mut center = Vec2::ZERO;
    for &index in &object.particle_indices {
        if index >= particle_pos.len() || index >= particle_mass.len() {
            continue;
        }
        let mass = particle_mass[index];
        mass_sum += mass;
        center += particle_pos[index] * mass;
    }
    if mass_sum <= 1e-6 {
        return None;
    }
    center /= mass_sum;

    let mut a00 = 0.0;
    let mut a01 = 0.0;
    let mut a10 = 0.0;
    let mut a11 = 0.0;
    for (slot, &index) in object.particle_indices.iter().enumerate() {
        if slot >= object.rest_local.len()
            || index >= particle_pos.len()
            || index >= particle_mass.len()
        {
            continue;
        }
        let mass = particle_mass[index];
        let p = particle_pos[index] - center;
        let q = object.rest_local[slot];
        a00 += mass * p.x * q.x;
        a01 += mass * p.x * q.y;
        a10 += mass * p.y * q.x;
        a11 += mass * p.y * q.y;
    }
    let theta = (a10 - a01).atan2(a00 + a11);

    let half_cell = CELL_SIZE_M * 0.5;
    let cos_t = theta.cos().abs();
    let sin_t = theta.sin().abs();
    let ext = Vec2::new(
        cos_t * half_cell + sin_t * half_cell,
        sin_t * half_cell + cos_t * half_cell,
    );
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for q in &object.rest_local {
        let world_center = center + rotate_local_to_world(*q, theta);
        let aabb_min = world_center - ext;
        let aabb_max = world_center + ext;
        min = Vec2::new(min.x.min(aabb_min.x), min.y.min(aabb_min.y));
        max = Vec2::new(max.x.max(aabb_max.x), max.y.max(aabb_max.y));
    }

    Some((center, theta, mass_sum, Aabb2 { min, max }))
}

fn build_local_sdf(rest_local: &[Vec2]) -> ObjectLocalSdf {
    if rest_local.is_empty() {
        return ObjectLocalSdf::default();
    }

    let half_cell = CELL_SIZE_M * 0.5;
    let padding = OBJECT_SDF_MAX_DISTANCE_M + half_cell;
    let mut extent = Vec2::splat(half_cell + padding);
    for center in rest_local {
        extent.x = extent.x.max(center.x.abs() + half_cell + padding);
        extent.y = extent.y.max(center.y.abs() + half_cell + padding);
    }

    let samples_per_cell = OBJECT_LOCAL_SDF_SAMPLES_PER_CELL.max(1) as f32;
    let spacing_m = CELL_SIZE_M / samples_per_cell;
    let width = ((extent.x * 2.0) / spacing_m).ceil() as usize;
    let height = ((extent.y * 2.0) / spacing_m).ceil() as usize;
    let width = width.max(1);
    let height = height.max(1);
    let origin_m = Vec2::new(-extent.x + spacing_m * 0.5, -extent.y + spacing_m * 0.5);

    let mut solid_mask = vec![false; width * height];
    for y in 0..height {
        for x in 0..width {
            let sample = origin_m + Vec2::new(x as f32 * spacing_m, y as f32 * spacing_m);
            let inside = rest_local.iter().any(|center| {
                (sample.x - center.x).abs() <= half_cell && (sample.y - center.y).abs() <= half_cell
            });
            solid_mask[sdf_index(width, x, y)] = inside;
        }
    }

    let non_solid_mask = solid_mask.iter().map(|&value| !value).collect::<Vec<_>>();
    let distance_to_solid = distance_transform(&solid_mask, width, height, spacing_m);
    let distance_to_empty = distance_transform(&non_solid_mask, width, height, spacing_m);
    let mut samples = vec![0.0; width * height];
    let half_sample = spacing_m * 0.5;
    for i in 0..samples.len() {
        samples[i] = if solid_mask[i] {
            -distance_to_empty[i] + half_sample
        } else {
            distance_to_solid[i] - half_sample
        };
    }

    ObjectLocalSdf {
        samples,
        width,
        height,
        spacing_m,
        origin_m,
    }
}

fn distance_transform(mask: &[bool], width: usize, height: usize, spacing: f32) -> Vec<f32> {
    let mut dist = vec![SDF_INF; width * height];
    for y in 0..height {
        for x in 0..width {
            if mask[sdf_index(width, x, y)] {
                dist[sdf_index(width, x, y)] = 0.0;
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

    for value in &mut dist {
        if *value < SDF_INF {
            *value *= spacing;
        }
    }
    dist
}

fn recenter_rest_local(rest_local: &mut [Vec2], particle_indices: &[usize], particle_mass: &[f32]) {
    if rest_local.is_empty() {
        return;
    }
    let mut mass_sum = 0.0;
    let mut weighted = Vec2::ZERO;
    for (slot, &index) in particle_indices.iter().enumerate() {
        let mass = particle_mass[index];
        mass_sum += mass;
        weighted += rest_local[slot] * mass;
    }
    if mass_sum <= 1e-6 {
        return;
    }
    let center = weighted / mass_sum;
    for value in rest_local.iter_mut() {
        *value -= center;
    }
}

fn world_bounds_min() -> Vec2 {
    Vec2::new(
        WORLD_MIN_CHUNK_X as f32 * CHUNK_WORLD_SIZE_M,
        WORLD_MIN_CHUNK_Y as f32 * CHUNK_WORLD_SIZE_M,
    )
}

fn world_bounds_max() -> Vec2 {
    Vec2::new(
        (WORLD_MAX_CHUNK_X + 1) as f32 * CHUNK_WORLD_SIZE_M,
        (WORLD_MAX_CHUNK_Y + 1) as f32 * CHUNK_WORLD_SIZE_M,
    )
}

fn build_grid_layout(cell_size: f32, padding: f32) -> (IVec2, UVec2) {
    let min_world = world_bounds_min() - Vec2::splat(padding);
    let max_world = world_bounds_max() + Vec2::splat(padding);
    let min_cell = sample_grid_cell(min_world, cell_size);
    let max_cell = sample_grid_cell(max_world, cell_size);
    let width = (max_cell.x - min_cell.x + 1).max(1) as u32;
    let height = (max_cell.y - min_cell.y + 1).max(1) as u32;
    (min_cell, UVec2::new(width, height))
}

fn grid_len(size: UVec2) -> usize {
    size.x as usize * size.y as usize
}

fn grid_max_cell(min_cell: IVec2, size: UVec2) -> IVec2 {
    min_cell + IVec2::new(size.x as i32 - 1, size.y as i32 - 1)
}

fn clamp_grid_range(
    min_cell: IVec2,
    max_cell: IVec2,
    grid_min: IVec2,
    grid_size: UVec2,
) -> Option<(IVec2, IVec2)> {
    let grid_max = grid_max_cell(grid_min, grid_size);
    let clamped_min = IVec2::new(min_cell.x.max(grid_min.x), min_cell.y.max(grid_min.y));
    let clamped_max = IVec2::new(max_cell.x.min(grid_max.x), max_cell.y.min(grid_max.y));
    if clamped_min.x > clamped_max.x || clamped_min.y > clamped_max.y {
        return None;
    }
    Some((clamped_min, clamped_max))
}

fn grid_cell_index(cell: IVec2, grid_min: IVec2, grid_size: UVec2) -> Option<usize> {
    let local = cell - grid_min;
    if local.x < 0 || local.y < 0 {
        return None;
    }
    if local.x >= grid_size.x as i32 || local.y >= grid_size.y as i32 {
        return None;
    }
    Some(local.y as usize * grid_size.x as usize + local.x as usize)
}

fn grid_index_to_cell(index: usize, grid_min: IVec2, grid_size: UVec2) -> IVec2 {
    let width = grid_size.x as usize;
    let x = index % width;
    let y = index / width;
    grid_min + IVec2::new(x as i32, y as i32)
}

fn sample_grid_cell(world_pos: Vec2, cell_size: f32) -> IVec2 {
    IVec2::new(
        (world_pos.x / cell_size).floor() as i32,
        (world_pos.y / cell_size).floor() as i32,
    )
}

fn sample_cell_center(cell: IVec2, cell_size: f32) -> Vec2 {
    (cell.as_vec2() + Vec2::splat(0.5)) * cell_size
}

fn broadphase_grid_cell(world_pos: Vec2, cell_size: f32) -> IVec2 {
    IVec2::new(
        (world_pos.x / cell_size).floor() as i32,
        (world_pos.y / cell_size).floor() as i32,
    )
}

fn sdf_index(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn world_to_local(world: Vec2, center: Vec2, theta: f32) -> Vec2 {
    let dx = world - center;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    Vec2::new(cos_t * dx.x + sin_t * dx.y, -sin_t * dx.x + cos_t * dx.y)
}

fn rotate_local_to_world(local: Vec2, theta: f32) -> Vec2 {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    Vec2::new(
        cos_t * local.x - sin_t * local.y,
        sin_t * local.x + cos_t * local.y,
    )
}

fn shortest_angle_delta(a: f32, b: f32) -> f32 {
    let mut delta = a - b;
    while delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    }
    while delta < -std::f32::consts::PI {
        delta += std::f32::consts::TAU;
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn created_object_rest_local_is_com_centered() {
        let mut objects = ObjectWorld::default();
        let positions = vec![
            Vec2::new(1.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(5.0, 0.0),
        ];
        let masses = vec![1.0, 2.0, 1.0];
        let created = objects.create_object(vec![0, 1, 2], &positions, &masses, 0.9, 2);
        assert!(created.is_some());

        let object = &objects.objects[0];
        let mut weighted = Vec2::ZERO;
        for (slot, &index) in object.particle_indices.iter().enumerate() {
            weighted += object.rest_local[slot] * masses[index];
        }
        assert!(
            weighted.length() < 1e-5,
            "rest_local weighted sum is not centered"
        );
    }

    #[test]
    fn local_sdf_is_negative_inside_positive_outside() {
        let sdf = build_local_sdf(&[Vec2::ZERO]);
        let inside = sdf.sample_distance(Vec2::ZERO).unwrap_or(f32::INFINITY);
        let outside = sdf
            .sample_distance(Vec2::new(CELL_SIZE_M * 2.0, 0.0))
            .unwrap_or(0.0);
        assert!(inside < 0.0);
        assert!(outside > 0.0);
    }

    #[test]
    fn physics_field_returns_nearest_object_sample() {
        let mut objects = ObjectWorld::default();
        let positions = vec![Vec2::ZERO];
        let masses = vec![1.0];
        let object_id = objects
            .create_object(vec![0], &positions, &masses, 0.9, 2)
            .unwrap_or(0);
        let mut field = ObjectPhysicsField::default();
        objects.update_physics_field(&positions, &masses, &mut field);

        let mut candidates = Vec::new();
        field.gather_candidate_object_ids(Vec2::ZERO, &mut candidates);
        assert!(
            !candidates.is_empty(),
            "no candidate returned for object center"
        );
        let nearest = candidates
            .iter()
            .filter_map(|&id| objects.evaluate_object_sdf(id, Vec2::ZERO))
            .min_by(|a, b| {
                a.distance_m
                    .partial_cmp(&b.distance_m)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(ObjectPhysicsSdfSample {
                distance_m: f32::INFINITY,
                normal_world: Vec2::ZERO,
            });
        assert!(candidates.contains(&object_id));
        assert!(nearest.distance_m < 0.0);
    }

    #[test]
    fn physics_field_keeps_multiple_object_splats_per_cell() {
        let mut objects = ObjectWorld::default();
        let positions = vec![Vec2::new(-0.2, 0.0), Vec2::new(0.2, 0.0)];
        let masses = vec![1.0, 1.0];
        objects
            .create_object(vec![0], &positions, &masses, 0.9, 2)
            .unwrap_or(0);
        objects
            .create_object(vec![1], &positions, &masses, 0.9, 2)
            .unwrap_or(1);

        let mut field = ObjectPhysicsField::default();
        objects.update_physics_field(&positions, &masses, &mut field);

        let mut candidates = Vec::new();
        field.gather_candidate_object_ids(Vec2::ZERO, &mut candidates);
        let unique_ids: HashSet<ObjectId> = candidates.iter().copied().collect();
        assert!(
            unique_ids.len() >= 2,
            "expected at least two object splats near query point"
        );
    }

    #[test]
    fn split_object_after_detach_creates_new_components() {
        let mut objects = ObjectWorld::default();
        let positions = vec![
            Vec2::new(-CELL_SIZE_M, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(CELL_SIZE_M, 0.0),
        ];
        let masses = vec![1.0, 1.0, 1.0];
        let object_id = objects
            .create_object(vec![0, 1, 2], &positions, &masses, 0.9, 2)
            .expect("object should be created");
        let detached = HashSet::from([1usize]);

        objects.split_object_after_detach(object_id, &detached, &positions, &masses);

        assert_eq!(objects.objects().len(), 2);
        assert!(
            objects
                .objects()
                .iter()
                .any(|object| object.id == object_id)
        );
        assert!(
            objects
                .objects()
                .iter()
                .all(|object| !object.particle_indices.contains(&1)),
            "detached particle should not remain in any object"
        );
    }
}
