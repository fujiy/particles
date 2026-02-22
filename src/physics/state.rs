use bevy::prelude::*;

#[derive(Resource, Debug, Default)]
pub struct SimulationState {
    pub running: bool,
    pub step_once: bool,
}

#[derive(Resource, Debug)]
pub struct SimulationParallelSettings {
    pub enabled: bool,
}

impl Default for SimulationParallelSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Resource, Debug, Default)]
pub struct SimulationPerfMetrics {
    pub physics_time_this_frame_secs: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PhysicsStepProfileSegment {
    pub name: String,
    pub wall_duration_ms: f64,
    pub cpu_duration_ms: f64,
}

#[derive(Resource, Debug, Default)]
pub struct PhysicsStepProfiler {
    pub total_duration_ms: f64,
    pub segments: Vec<PhysicsStepProfileSegment>,
}

#[derive(Message)]
pub struct ResetSimulationRequest;

#[derive(Message)]
pub struct LoadDefaultWorldRequest;

#[derive(Message)]
pub struct SaveMapRequest {
    pub slot_name: String,
}

#[derive(Message)]
pub struct LoadMapRequest {
    pub slot_name: String,
}

#[derive(Message)]
pub struct ReplayLoadScenarioRequest {
    pub scenario_name: String,
}

#[derive(Message)]
pub struct ReplaySaveArtifactRequest {
    pub save_final: bool,
}

#[derive(Resource, Debug, Default)]
pub struct ReplayState {
    pub enabled: bool,
    pub scenario_name: Option<String>,
    pub scenario_total_steps: usize,
    pub current_step: usize,
    pub loop_enabled: bool,
    pub status_message: String,
    pub baseline_particle_count: usize,
    pub baseline_solid_cell_count: usize,
}

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
