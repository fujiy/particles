use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use super::terrain::CELL_SIZE_M;

pub type ObjectId = u32;

pub const OBJECT_SHAPE_STIFFNESS_ALPHA: f32 = 0.92;
pub const OBJECT_SHAPE_ITERS: usize = 3;
pub const OBJECT_LOCAL_SDF_SAMPLES_PER_CELL: i32 = 2;
pub const OBJECT_PHYSICS_SDF_CELL_SIZE_M: f32 = CELL_SIZE_M * 0.5;
pub const OBJECT_BROADPHASE_CELL_SIZE_M: f32 = CELL_SIZE_M * 4.0;
pub const OBJECT_SDF_MAX_DISTANCE_M: f32 = CELL_SIZE_M * 2.5;
pub const OBJECT_BOUNDARY_PUSH_RADIUS_M: f32 = CELL_SIZE_M * 0.35;
pub const OBJECT_REPULSION_STIFFNESS: f32 = 0.70;

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
    pub local_sdf: ObjectLocalSdf,
    pub pose_center: Vec2,
    pub pose_theta: f32,
    pub prev_pose_center: Vec2,
    pub prev_pose_theta: f32,
    pub aabb_world: Aabb2,
    pub prev_aabb_world: Aabb2,
    pub pose_initialized: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ObjectPhysicsSdfSample {
    pub distance_m: f32,
    pub normal_world: Vec2,
    pub object_id: ObjectId,
}

#[derive(Resource, Debug)]
pub struct ObjectPhysicsField {
    sample_cell_size_m: f32,
    broadphase_cell_size_m: f32,
    max_distance_m: f32,
    samples: HashMap<IVec2, ObjectPhysicsSdfSample>,
    broadphase_buckets: HashMap<IVec2, Vec<ObjectId>>,
}

impl Default for ObjectPhysicsField {
    fn default() -> Self {
        Self {
            sample_cell_size_m: OBJECT_PHYSICS_SDF_CELL_SIZE_M,
            broadphase_cell_size_m: OBJECT_BROADPHASE_CELL_SIZE_M,
            max_distance_m: OBJECT_SDF_MAX_DISTANCE_M,
            samples: HashMap::new(),
            broadphase_buckets: HashMap::new(),
        }
    }
}

impl ObjectPhysicsField {
    pub fn clear(&mut self) {
        self.samples.clear();
        self.broadphase_buckets.clear();
    }

    pub fn sample_signed_distance_and_normal(
        &self,
        world_pos: Vec2,
    ) -> Option<(f32, Vec2, ObjectId)> {
        let center = sample_grid_cell(world_pos, self.sample_cell_size_m);
        let mut best: Option<ObjectPhysicsSdfSample> = None;
        for y in (center.y - 1)..=(center.y + 1) {
            for x in (center.x - 1)..=(center.x + 1) {
                let Some(sample) = self.samples.get(&IVec2::new(x, y)) else {
                    continue;
                };
                if best
                    .map(|current| sample.distance_m < current.distance_m)
                    .unwrap_or(true)
                {
                    best = Some(*sample);
                }
            }
        }
        best.map(|sample| (sample.distance_m, sample.normal_world, sample.object_id))
    }

    fn rebuild_broadphase(&mut self, objects: &[ObjectData]) {
        self.broadphase_buckets.clear();
        for object in objects {
            if !object.pose_initialized {
                continue;
            }
            let expanded = object.aabb_world.expanded(self.max_distance_m);
            let min_cell = broadphase_grid_cell(expanded.min, self.broadphase_cell_size_m);
            let max_cell = broadphase_grid_cell(expanded.max, self.broadphase_cell_size_m);
            for y in min_cell.y..=max_cell.y {
                for x in min_cell.x..=max_cell.x {
                    self.broadphase_buckets
                        .entry(IVec2::new(x, y))
                        .or_default()
                        .push(object.id);
                }
            }
        }
    }

    fn reproject_dirty_objects(&mut self, objects: &[ObjectData], dirty_ids: &HashSet<ObjectId>) {
        if dirty_ids.is_empty() {
            return;
        }

        let object_by_id: HashMap<ObjectId, &ObjectData> =
            objects.iter().map(|object| (object.id, object)).collect();
        let mut affected_sample_cells = HashSet::new();
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
            for y in min_cell.y..=max_cell.y {
                for x in min_cell.x..=max_cell.x {
                    affected_sample_cells.insert(IVec2::new(x, y));
                }
            }
        }

        for cell in affected_sample_cells {
            let world_pos = sample_cell_center(cell, self.sample_cell_size_m);
            let candidates = self.collect_broadphase_candidates(world_pos);

            let mut best: Option<ObjectPhysicsSdfSample> = None;
            for object_id in candidates {
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
                let Some((distance_m, normal_local)) =
                    object.local_sdf.sample_distance_and_normal(local)
                else {
                    continue;
                };
                let normal_world = rotate_local_to_world(normal_local, object.pose_theta);
                let candidate = ObjectPhysicsSdfSample {
                    distance_m,
                    normal_world: normal_world.normalize_or_zero(),
                    object_id,
                };
                if best
                    .map(|current| candidate.distance_m < current.distance_m)
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }

            if let Some(best_sample) = best {
                if best_sample.distance_m <= self.max_distance_m {
                    self.samples.insert(cell, best_sample);
                } else {
                    self.samples.remove(&cell);
                }
            } else {
                self.samples.remove(&cell);
            }
        }
    }

    fn collect_broadphase_candidates(&self, world_pos: Vec2) -> Vec<ObjectId> {
        let center = broadphase_grid_cell(world_pos, self.broadphase_cell_size_m);
        let mut ids = HashSet::new();
        for y in (center.y - 1)..=(center.y + 1) {
            for x in (center.x - 1)..=(center.x + 1) {
                let Some(bucket) = self.broadphase_buckets.get(&IVec2::new(x, y)) else {
                    continue;
                };
                ids.extend(bucket.iter().copied());
            }
        }
        ids.into_iter().collect()
    }

    fn tracked_object_ids(&self) -> HashSet<ObjectId> {
        let mut ids = HashSet::new();
        for sample in self.samples.values() {
            ids.insert(sample.object_id);
        }
        for bucket in self.broadphase_buckets.values() {
            ids.extend(bucket.iter().copied());
        }
        ids
    }
}

#[derive(Resource, Debug, Default)]
pub struct ObjectWorld {
    next_id: ObjectId,
    objects: Vec<ObjectData>,
    reaction_impulses: HashMap<ObjectId, Vec2>,
}

impl ObjectWorld {
    pub fn clear(&mut self) {
        self.objects.clear();
        self.reaction_impulses.clear();
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

    pub fn create_object(
        &mut self,
        particle_indices: Vec<usize>,
        particle_pos: &[Vec2],
        particle_mass: &[f32],
        shape_stiffness_alpha: f32,
        shape_iters: usize,
    ) -> Option<ObjectId> {
        if particle_indices.is_empty() {
            return None;
        }

        let mut mass_sum = 0.0;
        let mut com = Vec2::ZERO;
        for &index in &particle_indices {
            let mass = particle_mass[index];
            mass_sum += mass;
            com += particle_pos[index] * mass;
        }
        if mass_sum <= 1e-6 {
            return None;
        }
        com /= mass_sum;

        let mut rest_local = Vec::with_capacity(particle_indices.len());
        for &index in &particle_indices {
            rest_local.push(particle_pos[index] - com);
        }
        recenter_rest_local(&mut rest_local, &particle_indices, particle_mass);
        let local_sdf = build_local_sdf(&rest_local);

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.objects.push(ObjectData {
            id,
            particle_indices,
            rest_local,
            mass_sum,
            shape_stiffness_alpha: shape_stiffness_alpha.clamp(0.0, 1.0),
            shape_iters: shape_iters.max(1),
            shape_dirty: true,
            local_sdf,
            pose_center: Vec2::ZERO,
            pose_theta: 0.0,
            prev_pose_center: Vec2::ZERO,
            prev_pose_theta: 0.0,
            aabb_world: Aabb2::default(),
            prev_aabb_world: Aabb2::default(),
            pose_initialized: false,
        });
        Some(id)
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
            true
        });
    }

    pub fn update_physics_field(
        &mut self,
        particle_pos: &[Vec2],
        particle_mass: &[f32],
        field: &mut ObjectPhysicsField,
    ) {
        self.clear_reaction_impulses();
        let mut dirty_ids = HashSet::new();
        let previous_ids = field.tracked_object_ids();
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
            if moved || object.shape_dirty {
                dirty_ids.insert(object.id);
            }
        }

        let current_ids: HashSet<ObjectId> = self.objects.iter().map(|object| object.id).collect();
        let has_removed_object = previous_ids.iter().any(|id| !current_ids.contains(id));
        if has_removed_object {
            field.clear();
            dirty_ids.extend(current_ids.iter().copied());
        }

        field.rebuild_broadphase(&self.objects);
        field.reproject_dirty_objects(&self.objects, &dirty_ids);
        for object in &mut self.objects {
            if dirty_ids.contains(&object.id) {
                object.shape_dirty = false;
            }
        }
    }
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

        let Some((distance, _normal, sampled_id)) =
            field.sample_signed_distance_and_normal(Vec2::ZERO)
        else {
            panic!("no sample returned for object center");
        };
        assert_eq!(sampled_id, object_id);
        assert!(distance < 0.0);
    }
}
