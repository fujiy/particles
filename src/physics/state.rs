use bevy::prelude::*;

#[derive(Resource, Debug)]
pub struct SimulationState {
    pub running: bool,
    pub step_once: bool,
    /// Global kill-switch for MLS-MPM stepping (CPU and GPU).
    pub mpm_enabled: bool,
    /// When true, GPU compute path owns the continuum particle update.
    /// CPU MLS-MPM step is skipped.
    pub gpu_mpm_active: bool,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            running: false,
            step_once: false,
            // Enable MPM stepping by default.
            mpm_enabled: true,
            gpu_mpm_active: false,
        }
    }
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
pub enum SimUpdateSet {
    Controls,
    Interaction,
    Rendering,
    Ui,
    Overlay,
    Finalize,
}
