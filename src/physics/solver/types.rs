#[derive(Clone, Debug, Default)]
pub(crate) struct MpmPhase {
    pub(crate) name: &'static str,
    pub(crate) wall_secs: f64,
    pub(crate) cpu_secs: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct StepSimulationTiming {
    /// Per-phase timing breakdown for MPM water steps (empty for non-water frames).
    pub(crate) mpm_phases: Vec<MpmPhase>,
}
