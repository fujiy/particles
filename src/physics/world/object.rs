use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::physics::solver::params_defaults::DEFAULT_SOLVER_PARAMS;

pub const OBJECT_SHAPE_STIFFNESS_ALPHA: f32 = DEFAULT_SOLVER_PARAMS.object_shape_stiffness_alpha;
pub const OBJECT_SHAPE_ITERS: usize = DEFAULT_SOLVER_PARAMS.object_shape_iters;

pub type ObjectId = u32;

#[derive(Debug, Clone)]
pub struct ObjectData {
    pub id: ObjectId,
    pub particle_indices: Vec<usize>,
    pub rest_local: Vec<Vec2>,
    pub mass_sum: f32,
    pub shape_stiffness_alpha: f32,
    pub shape_iters: usize,
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

#[derive(Resource, Debug, Default)]
pub struct ObjectWorld {
    next_id: ObjectId,
    objects: Vec<ObjectData>,
    particle_owner: Vec<Option<ObjectId>>,
    reaction_impulses: HashMap<ObjectId, Vec2>,
}

impl ObjectWorld {
    pub fn clear(&mut self) {
        self.next_id = 0;
        self.objects.clear();
        self.particle_owner.clear();
        self.reaction_impulses.clear();
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
        *self.reaction_impulses.entry(object_id).or_insert(Vec2::ZERO) += impulse;
    }

    pub fn reaction_impulse_of(&self, object_id: ObjectId) -> Vec2 {
        self.reaction_impulses
            .get(&object_id)
            .copied()
            .unwrap_or(Vec2::ZERO)
    }

    pub fn object_of_particle(&self, particle_index: usize) -> Option<ObjectId> {
        self.particle_owner
            .get(particle_index)
            .copied()
            .flatten()
    }

    pub fn evaluate_object_sdf(
        &self,
        object_id: ObjectId,
        _world_pos: Vec2,
    ) -> Option<ObjectPhysicsSdfSample> {
        self.objects.iter().any(|object| object.id == object_id).then_some(
            ObjectPhysicsSdfSample {
                distance_m: 1.0,
                normal_world: Vec2::Y,
            },
        )
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
        let mut center = Vec2::ZERO;
        for &index in &particle_indices {
            if index >= particle_pos.len() {
                continue;
            }
            let mass = particle_mass.get(index).copied().unwrap_or(1.0).max(0.0);
            mass_sum += mass;
            center += particle_pos[index] * mass;
        }
        if mass_sum > 1e-6 {
            center /= mass_sum;
        }

        let rest_local = particle_indices
            .iter()
            .map(|&index| {
                particle_pos
                    .get(index)
                    .copied()
                    .unwrap_or(Vec2::ZERO)
                    - center
            })
            .collect();

        let object_id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.objects.push(ObjectData {
            id: object_id,
            particle_indices: particle_indices.clone(),
            rest_local,
            mass_sum,
            shape_stiffness_alpha,
            shape_iters,
        });

        for &index in &particle_indices {
            if index >= self.particle_owner.len() {
                self.particle_owner.resize(index + 1, None);
            }
            self.particle_owner[index] = Some(object_id);
        }

        Some(object_id)
    }

    pub fn split_object_after_detach(
        &mut self,
        object_id: ObjectId,
        detached: &HashSet<usize>,
        _particle_pos: &[Vec2],
        _particle_mass: &[f32],
    ) {
        let Some(object) = self.objects.iter_mut().find(|object| object.id == object_id) else {
            return;
        };

        let mut kept_indices = Vec::with_capacity(object.particle_indices.len());
        let mut kept_rest_local = Vec::with_capacity(object.rest_local.len());
        for (slot, &particle_index) in object.particle_indices.iter().enumerate() {
            if detached.contains(&particle_index) {
                if particle_index < self.particle_owner.len() {
                    self.particle_owner[particle_index] = None;
                }
                continue;
            }
            kept_indices.push(particle_index);
            kept_rest_local.push(*object.rest_local.get(slot).unwrap_or(&Vec2::ZERO));
        }
        object.particle_indices = kept_indices;
        object.rest_local = kept_rest_local;
    }

    pub fn split_objects_after_detach(
        &mut self,
        detached_particles: &HashSet<usize>,
        particle_pos: &[Vec2],
        particle_mass: &[f32],
    ) {
        let ids: Vec<ObjectId> = self.objects.iter().map(|object| object.id).collect();
        for object_id in ids {
            self.split_object_after_detach(object_id, detached_particles, particle_pos, particle_mass);
        }
        self.objects.retain(|object| !object.particle_indices.is_empty());
    }

    pub fn split_all_disconnected_objects(
        &mut self,
        _particle_pos: &[Vec2],
        _particle_mass: &[f32],
    ) {
    }

    pub fn remove_objects_by_ids(&mut self, ids: &HashSet<ObjectId>) {
        self.objects.retain(|object| !ids.contains(&object.id));
        for owner in &mut self.particle_owner {
            if let Some(id) = owner {
                if ids.contains(id) {
                    *owner = None;
                }
            }
        }
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
        for snapshot in snapshots {
            let id = snapshot.id;
            self.next_id = self.next_id.max(id.saturating_add(1));
            self.objects.push(ObjectData {
                id,
                particle_indices: snapshot.particle_indices.clone(),
                rest_local: snapshot.rest_local.clone(),
                mass_sum: 0.0,
                shape_stiffness_alpha: snapshot.shape_stiffness_alpha,
                shape_iters: snapshot.shape_iters,
            });
        }
        self.refresh_particle_owner(particle_pos.len(), particle_mass.len());
        Ok(())
    }

    pub fn apply_particle_remap(&mut self, old_to_new: &[Option<usize>], _particle_mass: &[f32]) {
        for object in &mut self.objects {
            let mut remapped_indices = Vec::with_capacity(object.particle_indices.len());
            let mut remapped_rest = Vec::with_capacity(object.rest_local.len());
            for (slot, &old_index) in object.particle_indices.iter().enumerate() {
                let Some(Some(new_index)) = old_to_new.get(old_index) else {
                    continue;
                };
                remapped_indices.push(*new_index);
                remapped_rest.push(*object.rest_local.get(slot).unwrap_or(&Vec2::ZERO));
            }
            object.particle_indices = remapped_indices;
            object.rest_local = remapped_rest;
        }
        self.objects.retain(|object| !object.particle_indices.is_empty());

        let new_len = old_to_new
            .iter()
            .filter_map(|entry| *entry)
            .max()
            .map(|index| index + 1)
            .unwrap_or(0);
        let mut new_owner = vec![None; new_len];
        for object in &self.objects {
            for &index in &object.particle_indices {
                if index < new_owner.len() {
                    new_owner[index] = Some(object.id);
                }
            }
        }
        self.particle_owner = new_owner;
    }

    fn refresh_particle_owner(&mut self, particle_pos_len: usize, particle_mass_len: usize) {
        let max_len = particle_pos_len.max(particle_mass_len);
        self.particle_owner.clear();
        self.particle_owner.resize(max_len, None);
        for object in &self.objects {
            for &particle_index in &object.particle_indices {
                if particle_index < self.particle_owner.len() {
                    self.particle_owner[particle_index] = Some(object.id);
                }
            }
        }
    }
}
