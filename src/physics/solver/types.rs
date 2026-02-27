use crate::physics::world::particle::ParticleStepBreakdown;

#[derive(Clone, Debug, Default)]
pub(crate) struct MpmPhase {
    pub(crate) name: &'static str,
    pub(crate) wall_secs: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StepSimulationTiming {
    pub(crate) particle_step_secs: f64,
    pub(crate) particle_step_cpu_secs: f64,
    pub(crate) particle_breakdown: ParticleStepBreakdown,
    pub(crate) terrain_fracture_commit_secs: f64,
    pub(crate) terrain_fracture_commit_cpu_secs: f64,
    /// Per-phase timing breakdown for MPM water steps (empty for non-water frames).
    pub(crate) mpm_phases: Vec<MpmPhase>,
}
