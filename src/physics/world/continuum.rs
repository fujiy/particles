use bevy::prelude::*;

pub const MATERIAL_ID_WATER: u8 = 0;
pub const MATERIAL_ID_GRANULAR_SOIL: u8 = 1;
pub const MATERIAL_ID_GRANULAR_SAND: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuumPhase {
    Water,
    GranularSoil,
    GranularSand,
}

impl ContinuumPhase {
    pub const fn id(self) -> u8 {
        match self {
            Self::Water => MATERIAL_ID_WATER,
            Self::GranularSoil => MATERIAL_ID_GRANULAR_SOIL,
            Self::GranularSand => MATERIAL_ID_GRANULAR_SAND,
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
    /// Plastic volume correction scalar (diff_log_J). Granular-only state, water keeps 0.0.
    pub v_vol: Vec<f32>,
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
        self.v_vol.clear();
        self.material_id.clear();
        self.owner_block_id.clear();
    }

    pub fn spawn_particle(
        &mut self,
        position: Vec2,
        velocity: Vec2,
        mass: f32,
        rest_volume: f32,
        phase: ContinuumPhase,
        v_vol: f32,
    ) -> usize {
        let index = self.len();
        self.x.push(position);
        self.v.push(velocity);
        self.m.push(mass);
        self.v0.push(rest_volume);
        self.f.push(Mat2::IDENTITY);
        self.c.push(Mat2::ZERO);
        self.v_vol.push(v_vol);
        self.material_id.push(phase.id());
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
            ContinuumPhase::Water,
            0.0,
        )
    }

    pub fn spawn_granular_soil_particle(
        &mut self,
        position: Vec2,
        velocity: Vec2,
        mass: f32,
        rest_volume: f32,
        v_vol: f32,
    ) -> usize {
        self.spawn_particle(
            position,
            velocity,
            mass,
            rest_volume,
            ContinuumPhase::GranularSoil,
            v_vol,
        )
    }

    pub fn spawn_granular_sand_particle(
        &mut self,
        position: Vec2,
        velocity: Vec2,
        mass: f32,
        rest_volume: f32,
        v_vol: f32,
    ) -> usize {
        self.spawn_particle(
            position,
            velocity,
            mass,
            rest_volume,
            ContinuumPhase::GranularSand,
            v_vol,
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
        assert_eq!(world.v_vol[0], 0.0);
    }
}
