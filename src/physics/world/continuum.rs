use bevy::prelude::*;

pub const MATERIAL_ID_WATER: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuumMaterial {
    Water,
}

impl ContinuumMaterial {
    pub const fn id(self) -> u8 {
        match self {
            Self::Water => MATERIAL_ID_WATER,
        }
    }
}

#[derive(Resource, Debug, Default)]
pub struct ContinuumParticleWorld {
    pub x: Vec<Vec2>,
    pub v: Vec<Vec2>,
    pub m: Vec<f32>,
    pub v0: Vec<f32>,
    pub f: Vec<Mat2>,
    pub c: Vec<Mat2>,
    pub material_id: Vec<u8>,
    pub owner_block_id: Vec<usize>,
}

impl ContinuumParticleWorld {
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn clear(&mut self) {
        self.x.clear();
        self.v.clear();
        self.m.clear();
        self.v0.clear();
        self.f.clear();
        self.c.clear();
        self.material_id.clear();
        self.owner_block_id.clear();
    }

    pub fn spawn_particle(
        &mut self,
        position: Vec2,
        velocity: Vec2,
        mass: f32,
        rest_volume: f32,
        material: ContinuumMaterial,
    ) -> usize {
        let index = self.len();
        self.x.push(position);
        self.v.push(velocity);
        self.m.push(mass);
        self.v0.push(rest_volume);
        self.f.push(Mat2::IDENTITY);
        self.c.push(Mat2::ZERO);
        self.material_id.push(material.id());
        self.owner_block_id.push(0);
        index
    }

    pub fn spawn_water_particle(
        &mut self,
        position: Vec2,
        velocity: Vec2,
        mass: f32,
        rest_volume: f32,
    ) -> usize {
        self.spawn_particle(
            position,
            velocity,
            mass,
            rest_volume,
            ContinuumMaterial::Water,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_one_water(mut world: ResMut<ContinuumParticleWorld>) {
        if world.is_empty() {
            world.spawn_water_particle(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0), 1.25, 0.5);
        }
    }

    #[test]
    fn continuum_world_can_be_updated_via_ecs_resource_system() {
        let mut app = App::new();
        app.init_resource::<ContinuumParticleWorld>();
        app.add_systems(Update, spawn_one_water);
        app.update();

        let world = app.world().resource::<ContinuumParticleWorld>();
        assert_eq!(world.len(), 1);
        assert_eq!(world.material_id[0], MATERIAL_ID_WATER);
        assert_eq!(world.f[0], Mat2::IDENTITY);
        assert_eq!(world.c[0], Mat2::ZERO);
    }
}
