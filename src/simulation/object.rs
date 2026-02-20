use bevy::prelude::*;

pub type ObjectId = u32;

pub const OBJECT_SHAPE_STIFFNESS_ALPHA: f32 = 0.92;
pub const OBJECT_SHAPE_ITERS: usize = 3;

#[derive(Debug, Clone)]
pub struct ObjectData {
    #[allow(dead_code)]
    pub id: ObjectId,
    pub particle_indices: Vec<usize>,
    pub rest_local: Vec<Vec2>,
    pub mass_sum: f32,
    pub shape_stiffness_alpha: f32,
    pub shape_iters: usize,
}

#[derive(Resource, Debug, Default)]
pub struct ObjectWorld {
    next_id: ObjectId,
    objects: Vec<ObjectData>,
}

impl ObjectWorld {
    pub fn clear(&mut self) {
        self.objects.clear();
        self.next_id = 0;
    }

    pub fn objects_mut(&mut self) -> &mut [ObjectData] {
        &mut self.objects
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

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.objects.push(ObjectData {
            id,
            particle_indices,
            rest_local,
            mass_sum,
            shape_stiffness_alpha: shape_stiffness_alpha.clamp(0.0, 1.0),
            shape_iters: shape_iters.max(1),
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
            object.mass_sum = object
                .particle_indices
                .iter()
                .map(|&index| particle_mass[index])
                .sum::<f32>()
                .max(1e-6);
            true
        });
    }
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
}
