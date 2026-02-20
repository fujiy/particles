use bevy::prelude::*;

#[derive(Resource, Debug, Default)]
pub struct SimulationState {
    pub running: bool,
}

#[derive(Resource, Debug, Default)]
pub struct SimulationPerfMetrics {
    pub physics_time_this_frame_secs: f64,
    pub potential_max_fps: f64,
}

#[derive(Message)]
pub struct ResetSimulationRequest;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum SimFixedSet {
    Physics,
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum SimUpdateSet {
    Controls,
    Interaction,
    Rendering,
    Ui,
    Overlay,
    Finalize,
}
