use particles::physics::scenario::run_default_scenarios;

#[test]
fn physics_scenarios_pass_thresholds() {
    let outputs = run_default_scenarios().expect("scenario execution should succeed");

    let failures: Vec<String> = outputs
        .iter()
        .filter(|output| !output.violations.is_empty())
        .map(|output| {
            format!(
                "scenario={} artifact_dir={} violations={} metrics={{penetration:{:.6}, max_speed:{:.6}, sleep_ratio:{:.6}}}",
                output.metrics.scenario,
                output.artifact_dir.display(),
                output.violations.join("; "),
                output.metrics.combined_penetration_rate,
                output.metrics.max_speed_mps,
                output.metrics.sleeping_ratio,
            )
        })
        .collect();

    assert!(
        failures.is_empty(),
        "physics scenario thresholds failed:\n{}",
        failures.join("\n")
    );
}
