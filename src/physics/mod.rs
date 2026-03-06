pub mod connectivity;
mod controls;
pub mod gpu_mpm;
pub mod material;
pub mod profiler;
pub mod save_load;
pub mod scenario;
pub mod state;
pub mod world;

pub use world::terrain::cell_to_world_center;

use bevy::prelude::*;

use self::controls::{
    apply_save_load_requests, apply_sim_reset, handle_replay_requests, handle_sim_controls,
    initialize_default_world,
};
use self::state::{
    LoadDefaultWorldRequest, LoadMapRequest, ReplayLoadScenarioRequest, ReplaySaveArtifactRequest,
    ReplayState, ResetSimulationRequest, SaveMapRequest, SimUpdateSet, SimulationState,
};
use self::world::terrain::TerrainWorld;
use crate::physics::material::MaterialParams;

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Update,
            (
                SimUpdateSet::Controls,
                SimUpdateSet::Interaction,
                SimUpdateSet::Rendering,
                SimUpdateSet::Ui,
                SimUpdateSet::Overlay,
                SimUpdateSet::Finalize,
            )
                .chain(),
        )
        .init_resource::<TerrainWorld>()
        .init_resource::<SimulationState>()
        .init_resource::<ReplayState>()
        .init_resource::<MaterialParams>()
        .add_message::<ResetSimulationRequest>()
        .add_message::<LoadDefaultWorldRequest>()
        .add_message::<SaveMapRequest>()
        .add_message::<LoadMapRequest>()
        .add_message::<ReplayLoadScenarioRequest>()
        .add_message::<ReplaySaveArtifactRequest>()
        .add_systems(Startup, initialize_default_world)
        .add_systems(
            Update,
            (
                handle_sim_controls,
                apply_sim_reset,
                handle_replay_requests,
                apply_save_load_requests,
            )
                .chain()
                .in_set(SimUpdateSet::Controls),
        );
    }
}
