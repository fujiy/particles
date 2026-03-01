use super::*;
use crate::physics::material::{DEFAULT_MATERIAL_PARAMS, terrain_boundary_radius_m};
use crate::physics::solver::granular;
use crate::physics::solver::params_defaults::DEFAULT_SOLVER_PARAMS;
use crate::physics::world::object::{OBJECT_SHAPE_ITERS, OBJECT_SHAPE_STIFFNESS_ALPHA};
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_WORLD_SIZE_M, TerrainWorld, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y,
    WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y, surface_y_for_world_x,
};

fn clear_particles(particles: &mut ParticleWorld) {
    particles.pos.clear();
    particles.prev_pos.clear();
    particles.vel.clear();
    particles.mass.clear();
    particles.material.clear();
    particles.density.clear();
    particles.lambda.clear();
    particles.delta_pos.clear();
    particles.viscosity_work.clear();
    particles.resize_work_buffers();
}

fn clear_fixed_terrain(terrain: &mut TerrainWorld) {
    let min_cell = IVec2::new(
        WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32,
        WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32,
    );
    let max_cell = IVec2::new(
        (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1,
        (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1,
    );
    terrain.fill_rect(min_cell, max_cell, TerrainCell::Empty);
}

#[test]
fn particle_transitions_to_sleep_after_threshold_frames() {
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let _ = particles.spawn_material_particles_from_cells(
        &[IVec2::new(0, 4)],
        ParticleMaterial::SandSolid,
        Vec2::ZERO,
    );
    let index = 0usize;
    particles.activity_state[index] = ParticleActivityState::Active;
    particles.active_hold_frames[index] = 0;
    particles.sleep_candidate_frames[index] = 0;
    particles.pending_wake[index] = false;
    particles.vel[index] = Vec2::ZERO;
    particles.prev_pos[index] = particles.pos[index];
    let object_world = ObjectWorld::default();

    for _ in 0..SLEEP_FRAMES {
        particles.update_sleep_states(&object_world);
    }

    assert_eq!(
        particles.activity_state[index],
        ParticleActivityState::Sleeping
    );
}

#[test]
fn wake_event_reactivates_sleeping_particle() {
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let _ = particles.spawn_material_particles_from_cells(
        &[IVec2::new(0, 4)],
        ParticleMaterial::SandSolid,
        Vec2::ZERO,
    );
    let index = 0usize;
    particles.activity_state[index] = ParticleActivityState::Sleeping;
    particles.vel[index] = Vec2::ZERO;
    particles.prev_pos[index] = particles.pos[index];

    particles.wake_particles_in_radius(particles.pos[index], WAKE_RADIUS * 0.5);
    particles.neighbor_grid.rebuild(&particles.pos);
    particles.propagate_and_apply_wake_requests();

    assert_eq!(
        particles.activity_state[index],
        ParticleActivityState::Active
    );
    assert!(particles.active_hold_frames[index] > 0);
}

#[test]
fn sleeping_particle_skips_gravity_integration() {
    let terrain = TerrainWorld::default();
    let mut object_field = ObjectPhysicsField::default();
    let mut objects = ObjectWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let _ = particles.spawn_material_particles_from_cells(
        &[IVec2::new(0, 4)],
        ParticleMaterial::SandSolid,
        Vec2::ZERO,
    );
    let index = 0usize;
    particles.activity_state[index] = ParticleActivityState::Sleeping;
    particles.vel[index] = Vec2::ZERO;
    particles.prev_pos[index] = particles.pos[index];
    particles.pending_wake[index] = false;
    let before = particles.pos[index];

    for _ in 0..6 {
        objects.update_physics_field(particles.positions(), particles.masses(), &mut object_field);
        particles.step_if_running(&terrain, &object_field, &mut objects, true);
        if objects.objects().is_empty() {
            break;
        }
    }

    assert!(
        particles.pos[index].distance(before) < 1e-6,
        "sleeping particle should not move while sleeping"
    );
}

#[test]
fn terrain_breaks_under_sustained_load() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    clear_fixed_terrain(&mut terrain);
    let load_cell = IVec2::new(0, 0);
    assert!(terrain.set_cell(load_cell, TerrainCell::stone()));

    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let mut objects = ObjectWorld::default();
    let samples_needed = (TERRAIN_LOAD_BREAK_DURATION_SECONDS
        / ((FIXED_DT / SUBSTEPS as f32) * TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS as f32))
        .ceil() as usize
        + 2;
    for _ in 0..samples_needed {
        particles.terrain_load_substep_counter += TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS as u64;
        let mut metric = HashMap::new();
        metric.insert(load_cell, TERRAIN_LOAD_STRAIN_THRESHOLD + 0.2);
        particles.evaluate_terrain_persistent_load(&terrain, &metric);
        if particles.apply_pending_terrain_fractures(&mut terrain, &mut objects) {
            break;
        }
    }

    assert!(
        matches!(
            terrain.get_loaded_cell_or_empty(load_cell),
            TerrainCell::Empty
        ),
        "terrain should fracture after sustained load duration"
    );
}

#[test]
fn particles_fall_under_gravity() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    terrain.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));
    let mut particles = ParticleWorld::default();
    particles.pos = vec![Vec2::new(0.0, 3.0)];
    particles.prev_pos = particles.pos.clone();
    particles.vel = vec![Vec2::ZERO];
    particles.mass = vec![default_particle_mass()];
    particles.material = vec![ParticleMaterial::WaterLiquid];
    let mut solver_params = DEFAULT_SOLVER_PARAMS;
    solver_params.fixed_dt = 1.0 / 60.0;
    particles.set_solver_params(solver_params);
    particles.resize_work_buffers();
    let mut object_world = ObjectWorld::default();
    let object_field = ObjectPhysicsField::default();
    let before_avg_y =
        particles.positions().iter().map(|p| p.y).sum::<f32>() / particles.positions().len() as f32;

    for _ in 0..10 {
        particles.step_if_running(&terrain, &object_field, &mut object_world, true);
    }
    let after_avg_y =
        particles.positions().iter().map(|p| p.y).sum::<f32>() / particles.positions().len() as f32;

    assert!(after_avg_y < before_avg_y - 0.01);
}

#[test]
fn long_run_stability_has_no_nan() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    terrain.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));
    let mut particles = ParticleWorld::default();
    let mut object_world = ObjectWorld::default();
    let object_field = ObjectPhysicsField::default();

    for _ in 0..300 {
        particles.step_if_running(&terrain, &object_field, &mut object_world, true);
    }

    assert!(particles.pos.iter().all(|p| p.is_finite()));
    assert!(particles.vel.iter().all(|v| v.is_finite()));
}

#[test]
fn static_terrain_boundary_push_affects_motion() {
    let mut terrain_static = TerrainWorld::default();
    terrain_static.reset_fixed_world();
    terrain_static
        .rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));
    let terrain_empty = TerrainWorld::default();
    let surface_y = surface_y_for_world_x(0) as f32 * CELL_SIZE_M;

    let mut with_terrain = ParticleWorld::default();
    with_terrain.pos = vec![Vec2::new(0.1, surface_y + 0.45)];
    with_terrain.prev_pos = with_terrain.pos.clone();
    with_terrain.vel = vec![Vec2::new(0.0, -8.0)];
    with_terrain.mass = vec![default_particle_mass()];
    with_terrain.material = vec![ParticleMaterial::WaterLiquid];
    let mut solver_params = DEFAULT_SOLVER_PARAMS;
    solver_params.fixed_dt = 1.0 / 60.0;
    with_terrain.set_solver_params(solver_params);
    with_terrain.resize_work_buffers();

    let mut without_terrain = ParticleWorld::default();
    without_terrain.pos = with_terrain.pos.clone();
    without_terrain.prev_pos = without_terrain.pos.clone();
    without_terrain.vel = with_terrain.vel.clone();
    without_terrain.mass = with_terrain.mass.clone();
    without_terrain.material = with_terrain.material.clone();
    without_terrain.set_solver_params(solver_params);
    without_terrain.resize_work_buffers();
    let mut object_world = ObjectWorld::default();
    let object_field = ObjectPhysicsField::default();

    for _ in 0..12 {
        with_terrain.step_if_running(&terrain_static, &object_field, &mut object_world, true);
        without_terrain.step_if_running(&terrain_empty, &object_field, &mut object_world, true);
    }

    let y_with = with_terrain.pos[0].y;
    let y_without = without_terrain.pos[0].y;
    let d_with = terrain_static
        .sample_signed_distance_and_normal(with_terrain.pos[0])
        .map(|(d, _)| d)
        .unwrap_or(0.0);
    let d_without = terrain_static
        .sample_signed_distance_and_normal(without_terrain.pos[0])
        .map(|(d, _)| d)
        .unwrap_or(0.0);
    assert!(
        d_with > d_without + 0.05 && y_with > y_without,
        "boundary push did not keep particle away from terrain (with={y_with}, without={y_without}, d_with={d_with}, d_without={d_without})"
    );
}

#[test]
fn particles_do_not_penetrate_frozen_terrain() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    terrain.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));
    let surface_y = surface_y_for_world_x(0) as f32 * CELL_SIZE_M;
    let mut particles = ParticleWorld::default();
    particles.pos = vec![Vec2::new(0.1, surface_y + 1.0)];
    particles.prev_pos = particles.pos.clone();
    particles.vel = vec![Vec2::new(0.0, -8.0)];
    particles.mass = vec![default_particle_mass()];
    particles.material = vec![ParticleMaterial::WaterLiquid];
    particles.resize_work_buffers();
    let mut object_world = ObjectWorld::default();
    let object_field = ObjectPhysicsField::default();

    for _ in 0..12 {
        particles.step_if_running(&terrain, &object_field, &mut object_world, true);
    }

    let p = particles.pos[0];
    let d = terrain
        .sample_signed_distance_and_normal(p)
        .map(|(distance, _)| distance)
        .unwrap_or(0.0);
    assert!(
        d > -0.08,
        "particle tunneled too deep into terrain: y={}, signed_distance={}",
        p.y,
        d
    );
}

#[test]
fn paused_simulation_keeps_particle_positions() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    terrain.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));
    let mut particles = ParticleWorld::default();
    let mut object_world = ObjectWorld::default();
    let object_field = ObjectPhysicsField::default();
    let before = particles.pos.clone();

    particles.step_if_running(&terrain, &object_field, &mut object_world, false);

    assert_eq!(before, particles.pos);
}

#[test]
fn reset_restores_initial_configuration() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    terrain.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));
    let mut particles = ParticleWorld::default();
    let mut object_world = ObjectWorld::default();
    let object_field = ObjectPhysicsField::default();
    let initial_pos = particles.pos.clone();
    let initial_vel = particles.vel.clone();

    particles.step_if_running(&terrain, &object_field, &mut object_world, true);
    particles.reset_to_initial();

    assert_eq!(particles.pos, initial_pos);
    assert_eq!(particles.vel, initial_vel);
}

#[test]
fn boundary_overcrowding_remains_stable() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    terrain.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));
    let mut particles = ParticleWorld::default();
    let left_wall_x = WORLD_MIN_CHUNK_X as f32 * CHUNK_WORLD_SIZE_M;

    for (i, pos) in particles.pos.iter_mut().enumerate() {
        let row = (i / 20) as f32;
        let col = (i % 20) as f32;
        *pos = Vec2::new(left_wall_x + 0.25 + col * 0.03, 0.4 + row * 0.03);
    }
    particles.prev_pos.clone_from(&particles.pos);
    particles.vel.fill(Vec2::ZERO);
    let mut object_world = ObjectWorld::default();
    let object_field = ObjectPhysicsField::default();

    for _ in 0..30 {
        particles.step_if_running(&terrain, &object_field, &mut object_world, true);
    }

    for pos in &particles.pos {
        assert!(pos.is_finite());
    }
    for vel in &particles.vel {
        assert!(vel.is_finite());
        assert!(vel.length() <= PARTICLE_SPEED_LIMIT_MPS + 1e-4);
    }
}

#[test]
fn shape_matching_keeps_stone_object_cohesive() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    particles.pos.clear();
    particles.prev_pos.clear();
    particles.vel.clear();
    particles.mass.clear();
    particles.material.clear();
    particles.density.clear();
    particles.lambda.clear();
    particles.delta_pos.clear();
    particles.viscosity_work.clear();
    particles.resize_work_buffers();

    let cells = [IVec2::new(0, 6), IVec2::new(1, 6), IVec2::new(2, 6)];
    let indices = particles.spawn_stone_particles_from_cells(&cells, Vec2::ZERO);
    let initial_span = particles.pos[indices[2]].x - particles.pos[indices[0]].x;

    let mut objects = ObjectWorld::default();
    let _ = objects.create_object(
        indices,
        particles.positions(),
        particles.masses(),
        OBJECT_SHAPE_STIFFNESS_ALPHA,
        OBJECT_SHAPE_ITERS,
    );
    let object_field = ObjectPhysicsField::default();

    for _ in 0..30 {
        particles.step_if_running(&terrain, &object_field, &mut objects, true);
    }

    let span = particles.pos[2].x - particles.pos[0].x;
    assert!(
        (span - initial_span).abs() < 0.05,
        "shape matching drifted too much: initial={initial_span}, current={span}"
    );
}

#[test]
fn water_object_contact_pushes_object_motion() {
    let terrain = TerrainWorld::default();
    let stone_cell = IVec2::new(0, 6);
    let water_pos = cell_to_world_center(stone_cell);

    let mut with_water = ParticleWorld::default();
    with_water.pos.clear();
    with_water.prev_pos.clear();
    with_water.vel.clear();
    with_water.mass.clear();
    with_water.material.clear();
    with_water.density.clear();
    with_water.lambda.clear();
    with_water.delta_pos.clear();
    with_water.viscosity_work.clear();
    with_water.resize_work_buffers();
    let stone_indices = with_water.spawn_stone_particles_from_cells(&[stone_cell], Vec2::ZERO);
    with_water.pos.push(water_pos);
    with_water.prev_pos.push(water_pos);
    with_water.vel.push(Vec2::ZERO);
    with_water.mass.push(default_particle_mass());
    with_water.material.push(ParticleMaterial::WaterLiquid);
    with_water.resize_work_buffers();

    let mut with_water_objects = ObjectWorld::default();
    let _ = with_water_objects
        .create_object(
            stone_indices,
            with_water.positions(),
            with_water.masses(),
            OBJECT_SHAPE_STIFFNESS_ALPHA,
            OBJECT_SHAPE_ITERS,
        )
        .expect("object should be created");
    let mut with_water_field = ObjectPhysicsField::default();
    with_water_objects.update_physics_field(
        with_water.positions(),
        with_water.masses(),
        &mut with_water_field,
    );
    with_water.step_if_running(&terrain, &with_water_field, &mut with_water_objects, true);

    let mut without_water = ParticleWorld::default();
    without_water.pos.clear();
    without_water.prev_pos.clear();
    without_water.vel.clear();
    without_water.mass.clear();
    without_water.material.clear();
    without_water.density.clear();
    without_water.lambda.clear();
    without_water.delta_pos.clear();
    without_water.viscosity_work.clear();
    without_water.resize_work_buffers();
    let stone_indices_no_water =
        without_water.spawn_stone_particles_from_cells(&[stone_cell], Vec2::ZERO);

    let mut without_water_objects = ObjectWorld::default();
    let _ = without_water_objects
        .create_object(
            stone_indices_no_water,
            without_water.positions(),
            without_water.masses(),
            OBJECT_SHAPE_STIFFNESS_ALPHA,
            OBJECT_SHAPE_ITERS,
        )
        .expect("object should be created");
    let mut without_water_field = ObjectPhysicsField::default();
    without_water_objects.update_physics_field(
        without_water.positions(),
        without_water.masses(),
        &mut without_water_field,
    );
    without_water.step_if_running(
        &terrain,
        &without_water_field,
        &mut without_water_objects,
        true,
    );

    let with_vel = with_water.vel[0];
    let without_vel = without_water.vel[0];
    assert!(
        (with_vel - without_vel).length() > 1e-3,
        "expected water contact to alter object velocity (with={with_vel:?}, without={without_vel:?})"
    );
}

#[test]
fn solid_object_breaks_on_explicit_fracture_command() {
    let mut particles = ParticleWorld::default();
    particles.pos.clear();
    particles.prev_pos.clear();
    particles.vel.clear();
    particles.mass.clear();
    particles.material.clear();
    particles.density.clear();
    particles.lambda.clear();
    particles.delta_pos.clear();
    particles.viscosity_work.clear();
    particles.resize_work_buffers();

    let indices = particles.spawn_material_particles_from_cells(
        &[IVec2::new(0, 1)],
        ParticleMaterial::SandSolid,
        Vec2::ZERO,
    );
    let index = indices[0];
    let center = particles.pos[index];

    let mut objects = ObjectWorld::default();
    let _ = objects
        .create_object(
            indices,
            particles.positions(),
            particles.masses(),
            OBJECT_SHAPE_STIFFNESS_ALPHA,
            OBJECT_SHAPE_ITERS,
        )
        .expect("object should be created");
    let detached = particles.fracture_solid_particles_in_radius(center, CELL_SIZE_M);
    objects.split_objects_after_detach(&detached, particles.positions(), particles.masses());

    assert!(
        objects.objects().is_empty(),
        "object should detach completely after explicit fracture"
    );
    assert!(
        particles
            .materials()
            .iter()
            .all(|&m| !matches!(m, ParticleMaterial::SandSolid)),
        "fractured particles should no longer remain solid"
    );
}

#[test]
fn object_fracture_detaches_only_target_particles() {
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);

    let indices = particles.spawn_material_particles_from_cells(
        &[IVec2::new(0, 5), IVec2::new(1, 5), IVec2::new(2, 5)],
        ParticleMaterial::SandSolid,
        Vec2::ZERO,
    );
    let left = indices[0];
    let middle = indices[1];
    let right = indices[2];

    let mut objects = ObjectWorld::default();
    let object_id = objects
        .create_object(
            indices.clone(),
            particles.positions(),
            particles.masses(),
            OBJECT_SHAPE_STIFFNESS_ALPHA,
            OBJECT_SHAPE_ITERS,
        )
        .expect("object should be created");

    particles.pending_object_fractures.insert(object_id);
    particles
        .pending_object_fracture_particles
        .entry(object_id)
        .or_default()
        .insert(middle);
    particles.apply_object_fractures(&mut objects);

    assert_eq!(particles.material[left], ParticleMaterial::SandGranular);
    assert_eq!(particles.material[right], ParticleMaterial::SandGranular);
    assert_eq!(particles.material[middle], ParticleMaterial::SandGranular);
    assert!(objects.objects().is_empty());
}

#[test]
fn terrain_small_component_detaches_into_object() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    clear_fixed_terrain(&mut terrain);

    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let mut objects = ObjectWorld::default();

    let fracture_cell = IVec2::new(0, 20);
    let detached_cell = IVec2::new(1, 20);
    assert!(terrain.set_cell(fracture_cell, TerrainCell::stone()));
    assert!(terrain.set_cell(detached_cell, TerrainCell::stone()));
    let impact_velocity = Vec2::new(2.0, 0.5);
    let impact_pos = cell_to_world_center(fracture_cell);

    particles.pending_terrain_fractures.insert(fracture_cell);
    particles.pending_terrain_fracture_seeds.insert(
        fracture_cell,
        TerrainFractureSeed {
            impulse_sum: 10.0,
            velocity_weighted_sum: impact_velocity * 10.0,
            contact_weighted_sum: impact_pos * 10.0,
        },
    );

    let changed = particles.apply_pending_terrain_fractures(&mut terrain, &mut objects);
    assert!(changed);
    assert!(matches!(
        terrain.get_loaded_cell_or_empty(fracture_cell),
        TerrainCell::Empty
    ));
    assert!(matches!(
        terrain.get_loaded_cell_or_empty(detached_cell),
        TerrainCell::Empty
    ));
    assert!(
        objects.objects().is_empty(),
        "single-cell detached object should be auto-fractured"
    );
    assert!(
        particles
            .material
            .iter()
            .any(|&m| matches!(m, ParticleMaterial::StoneGranular)),
        "fracture cell should spawn granular particles"
    );
}

#[test]
fn terrain_large_component_stays_in_terrain() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    clear_fixed_terrain(&mut terrain);

    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let mut objects = ObjectWorld::default();

    let base = IVec2::new(-6, 30);
    let width = 12;
    let height = 11;
    for y in 0..height {
        for x in 0..width {
            let cell = base + IVec2::new(x, y);
            assert!(terrain.set_cell(cell, TerrainCell::stone()));
        }
    }

    let fracture_cell = base + IVec2::new(0, 0);
    particles.pending_terrain_fractures.insert(fracture_cell);
    particles.pending_terrain_fracture_seeds.insert(
        fracture_cell,
        TerrainFractureSeed {
            impulse_sum: 8.0,
            velocity_weighted_sum: Vec2::new(1.0, 0.0) * 8.0,
            contact_weighted_sum: cell_to_world_center(fracture_cell) * 8.0,
        },
    );

    let changed = particles.apply_pending_terrain_fractures(&mut terrain, &mut objects);
    assert!(changed);
    assert!(matches!(
        terrain.get_loaded_cell_or_empty(fracture_cell),
        TerrainCell::Empty
    ));
    let remaining_cell = base + IVec2::new(width - 1, height - 1);
    assert!(matches!(
        terrain.get_loaded_cell_or_empty(remaining_cell),
        TerrainCell::Solid { .. }
    ));
    assert!(objects.objects().is_empty());
    assert!(
        particles
            .material
            .iter()
            .all(|&m| !matches!(m, ParticleMaterial::StoneSolid)),
        "large terrain component should not be detached into object particles"
    );
}

#[test]
fn edit_removed_cells_detach_small_terrain_component_into_object() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    clear_fixed_terrain(&mut terrain);

    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let mut objects = ObjectWorld::default();

    let removed_cell = IVec2::new(0, 20);
    let detached_cells = [IVec2::new(1, 20), IVec2::new(2, 20)];
    for &cell in &detached_cells {
        assert!(terrain.set_cell(cell, TerrainCell::stone()));
    }

    let removed = HashSet::from([removed_cell]);
    let changed = particles.detach_terrain_components_after_cell_removal(
        &mut terrain,
        &mut objects,
        &removed,
    );
    assert!(changed);
    for &cell in &detached_cells {
        assert!(matches!(
            terrain.get_loaded_cell_or_empty(cell),
            TerrainCell::Empty
        ));
    }
    assert_eq!(objects.objects().len(), 1);
    assert_eq!(
        objects.objects()[0].particle_indices.len(),
        detached_cells.len()
    );
}

#[test]
fn single_cell_object_is_auto_fractured_on_postprocess() {
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let indices = particles.spawn_material_particles_from_cells(
        &[IVec2::new(0, 4)],
        ParticleMaterial::SandSolid,
        Vec2::ZERO,
    );
    let mut objects = ObjectWorld::default();
    let _ = objects
        .create_object(
            indices,
            particles.positions(),
            particles.masses(),
            OBJECT_SHAPE_STIFFNESS_ALPHA,
            OBJECT_SHAPE_ITERS,
        )
        .expect("object should be created");

    particles.postprocess_objects_after_topology_edit(&mut objects);

    assert!(objects.objects().is_empty());
    assert!(
        particles
            .materials()
            .iter()
            .any(|&m| matches!(m, ParticleMaterial::SandGranular)),
        "single-cell object should be converted to granular particles"
    );
}

#[test]
fn granular_solver_does_not_modify_water_only_particles() {
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.pos.push(Vec2::new(-0.1, 1.0));
    particles.prev_pos.push(Vec2::new(-0.1, 1.0));
    particles.vel.push(Vec2::new(0.0, -1.0));
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let before_pos = particles.pos.clone();
    let before_vel = particles.vel.clone();
    let terrain = TerrainWorld::default();
    let object_field = ObjectPhysicsField::default();
    let object_world = ObjectWorld::default();

    let reaction = granular::solve_contacts(
        &mut particles,
        &terrain,
        &object_field,
        &object_world,
        FIXED_DT / SUBSTEPS as f32,
    );

    assert!(reaction.is_empty());
    assert_eq!(particles.pos, before_pos);
    assert_eq!(particles.vel, before_vel);
}

#[test]
fn granular_solver_pushes_particle_out_of_terrain() {
    let mut terrain = TerrainWorld::default();
    terrain.reset_fixed_world();
    terrain.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));

    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let indices = particles.spawn_material_particles_from_cells(
        &[IVec2::new(0, 1)],
        ParticleMaterial::SandGranular,
        Vec2::new(0.0, -2.0),
    );
    let index = indices[0];
    particles.pos[index] = Vec2::new(0.0, 0.01);
    particles.prev_pos[index] = particles.pos[index];
    particles.vel[index] = Vec2::new(0.0, -2.0);
    particles.resize_work_buffers();

    let object_field = ObjectPhysicsField::default();
    let object_world = ObjectWorld::default();
    let before_y = particles.pos[index].y;

    let _ = granular::solve_contacts(
        &mut particles,
        &terrain,
        &object_field,
        &object_world,
        FIXED_DT / SUBSTEPS as f32,
    );

    assert!(
        particles.pos[index].y >= before_y,
        "expected granular XPBD contact to resolve terrain penetration"
    );
}

#[test]
fn fractured_granular_particles_remain_awake_during_sleep_lock() {
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let _ = particles.spawn_material_particles_from_cells(
        &[IVec2::new(0, 3)],
        ParticleMaterial::SandSolid,
        Vec2::ZERO,
    );
    particles.resize_work_buffers();

    let center = particles.pos[0];
    let detached = particles.fracture_solid_particles_in_radius(center, CELL_SIZE_M);
    assert_eq!(detached.len(), 1);
    assert!(matches!(
        particles.material[0],
        ParticleMaterial::SandGranular
    ));

    particles.vel[0] = Vec2::ZERO;
    particles.prev_pos[0] = particles.pos[0];
    let object_world = ObjectWorld::default();
    for _ in 0..(SLEEP_FRAMES + 4) {
        particles.update_sleep_states(&object_world);
        assert!(
            particles.activity_state[0] == ParticleActivityState::Active,
            "fracture sleep lock should keep fresh granular particle active"
        );
    }
}

#[test]
fn sub_block_promotes_to_high_rate_when_debt_exceeds_threshold() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.pos.push(Vec2::new(0.0, 2.0));
    particles.prev_pos.push(Vec2::new(0.0, 2.0));
    particles.vel.push(Vec2::ZERO);
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let sub_block = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    let debt_threshold = particles.solver_params.sub_block_debt_promote_threshold;
    let slow_divisor = rate_divisor_from_level(particles.solver_params.sub_block_max_level);
    let state = particles.sub_block_states.entry(sub_block).or_default();
    state.rate_divisor = slow_divisor;
    state.boundary_debt_impulse = Vec2::new(debt_threshold * 1.1, 0.0);

    particles.prepare_sub_block_rate_scheduler(&terrain);

    let promoted = particles
        .sub_block_states
        .get(&sub_block)
        .expect("sub-block state should exist");
    assert_eq!(promoted.rate_divisor, RATE_DIVISOR_MIN);
}

#[test]
fn sub_block_slow_intrusion_does_not_force_high_rate() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.pos.push(Vec2::new(2.10, 4.0));
    particles.prev_pos.push(Vec2::new(1.90, 4.0));
    particles.vel.push(Vec2::new(0.05, 0.0));
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let sub_block = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    let level_one_divisor = rate_divisor_from_level(1);
    let state = particles.sub_block_states.entry(sub_block).or_default();
    state.rate_divisor = level_one_divisor;

    particles.prepare_sub_block_rate_scheduler(&terrain);

    let refreshed = particles
        .sub_block_states
        .get(&sub_block)
        .expect("sub-block state should exist");
    assert_eq!(refreshed.rate_divisor, level_one_divisor);
}

#[test]
fn sub_block_fast_intrusion_forces_high_rate() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.solver_params.sub_block_level0_max_norm_disp = 0.03;
    particles.pos.push(Vec2::new(2.10, 4.0));
    particles.prev_pos.push(Vec2::new(1.90, 4.0));
    particles.vel.push(Vec2::new(0.90, 0.0));
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let sub_block = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    let level_one_divisor = rate_divisor_from_level(1);
    let state = particles.sub_block_states.entry(sub_block).or_default();
    state.rate_divisor = level_one_divisor;

    particles.prepare_sub_block_rate_scheduler(&terrain);

    let refreshed = particles
        .sub_block_states
        .get(&sub_block)
        .expect("sub-block state should exist");
    assert_eq!(refreshed.rate_divisor, RATE_DIVISOR_MIN);
}

#[test]
fn sub_block_gravity_accumulates_with_effective_step_time() {
    let terrain = TerrainWorld::default();
    let object_field = ObjectPhysicsField::default();
    let mut object_world = ObjectWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.solver_params.substeps = 1;
    particles.solver_params.fixed_dt = 1.0 / 60.0;
    particles.solver_params.sub_block_demote_frames = u16::MAX;
    particles.solver_params.sub_block_promote_frames = u16::MAX;
    particles.solver_params.sub_block_max_level = 8;
    particles.pos.push(Vec2::new(0.0, 4.0));
    particles.prev_pos.push(Vec2::new(0.0, 4.0));
    particles.vel.push(Vec2::ZERO);
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let sub_block = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    let state = particles.sub_block_states.entry(sub_block).or_default();
    state.rate_divisor = rate_divisor_from_level(2);

    for _ in 0..4 {
        particles.step_if_running(&terrain, &object_field, &mut object_world, true);
    }

    let total_time = 4.0 * particles.solver_params.fixed_dt;
    let expected_vy = particles.solver_params.gravity_mps2.y * total_time;
    assert!(
        (particles.vel[0].y - expected_vy).abs() < 1e-4,
        "expected vy={expected_vy}, got {}",
        particles.vel[0].y
    );
}

#[test]
fn sub_block_level_reaches_configured_max_for_tiny_displacement() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.solver_params.sub_block_demote_frames = 1;
    particles.solver_params.sub_block_max_level = 6;
    particles.pos.push(Vec2::new(0.0, 4.0));
    particles.prev_pos.push(Vec2::new(0.0, 4.0));
    particles.vel.push(Vec2::ZERO);
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let sub_block = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    particles.prepare_sub_block_rate_scheduler(&terrain);

    let refreshed = particles
        .sub_block_states
        .get(&sub_block)
        .expect("sub-block state should exist");
    assert_eq!(refreshed.rate_divisor, rate_divisor_from_level(6));
}

#[test]
fn sub_block_neighbor_level_difference_is_limited_to_one() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.solver_params.sub_block_demote_frames = 1;
    particles.solver_params.sub_block_promote_frames = 1;
    particles.solver_params.sub_block_max_level = 6;

    particles.pos.push(Vec2::new(0.1, 4.0));
    particles.prev_pos.push(Vec2::new(0.1, 4.0));
    particles.vel.push(Vec2::new(1.0, 0.0));
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);

    particles.pos.push(Vec2::new(2.1, 4.0));
    particles.prev_pos.push(Vec2::new(2.1, 4.0));
    particles.vel.push(Vec2::ZERO);
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let left_block = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    let right_block = world_pos_to_sub_block(
        particles.pos[1],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    assert_eq!(right_block, left_block + IVec2::X);

    particles.prepare_sub_block_rate_scheduler(&terrain);

    let left = particles
        .sub_block_states
        .get(&left_block)
        .expect("left sub-block state should exist");
    let right = particles
        .sub_block_states
        .get(&right_block)
        .expect("right sub-block state should exist");
    let left_level = rate_level_from_divisor(
        left.rate_divisor,
        particles.solver_params.sub_block_max_level,
    );
    let right_level = rate_level_from_divisor(
        right.rate_divisor,
        particles.solver_params.sub_block_max_level,
    );
    assert!((left_level as i16 - right_level as i16).abs() <= 1);
}

#[test]
fn sub_block_crossing_particle_is_forced_to_update_this_substep() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.solver_params.sub_block_max_level = 6;

    particles.pos.push(Vec2::new(2.10, 4.0));
    particles.prev_pos.push(Vec2::new(1.90, 4.0));
    particles.vel.push(Vec2::new(0.10, 0.0));
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let dst_block = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    let state = particles.sub_block_states.entry(dst_block).or_default();
    state.rate_divisor = rate_divisor_from_level(1);

    particles.prepare_sub_block_rate_scheduler(&terrain);

    let refreshed = particles
        .sub_block_states
        .get(&dst_block)
        .expect("sub-block state should exist");
    assert!(
        !refreshed.scheduled_this_substep,
        "destination block should remain unscheduled on odd substep for divisor=2"
    );
    assert!(
        particles.particle_sub_block_update_mask[0],
        "crossing particle must still be updated this substep"
    );
}

#[test]
fn overlay_includes_granular_only_sub_block() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.solver_params.sub_block_max_level = 6;

    particles.pos.push(Vec2::new(0.0, 4.0));
    particles.prev_pos.push(Vec2::new(0.0, 4.0));
    particles.vel.push(Vec2::ZERO);
    particles
        .mass
        .push(particle_properties(ParticleMaterial::SandGranular).mass);
    particles.material.push(ParticleMaterial::SandGranular);
    particles.resize_work_buffers();

    let coord = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    particles.prepare_sub_block_rate_scheduler(&terrain);

    let state = particles
        .sub_block_states
        .get(&coord)
        .expect("granular-only block state should exist");
    let sample = particles
        .sub_block_overlay_samples()
        .iter()
        .find(|sample| sample.coord == coord)
        .expect("granular-only block should be visible in overlay samples");
    assert_eq!(sample.rate_divisor, state.rate_divisor);
}

#[test]
fn granular_particle_uses_sub_block_schedule_mask() {
    let terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.solver_params.sub_block_max_level = 6;

    particles.pos.push(Vec2::new(2.0, 4.0));
    particles.prev_pos.push(Vec2::new(2.0, 4.0));
    particles.vel.push(Vec2::ZERO);
    particles
        .mass
        .push(particle_properties(ParticleMaterial::SandGranular).mass);
    particles.material.push(ParticleMaterial::SandGranular);
    particles.resize_work_buffers();

    let coord = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    let state = particles.sub_block_states.entry(coord).or_default();
    state.rate_divisor = rate_divisor_from_level(1);

    particles.prepare_sub_block_rate_scheduler(&terrain);

    let refreshed = particles
        .sub_block_states
        .get(&coord)
        .expect("sub-block state should exist");
    assert!(
        !refreshed.scheduled_this_substep,
        "divisor=2 should be unscheduled on first scheduler substep"
    );
    assert!(
        !particles.is_particle_scheduled_in_sub_block(0),
        "granular particle should follow sub-block schedule mask"
    );
}

#[test]
fn sub_block_debt_apply_updates_velocity_before_constraints() {
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    particles.pos.push(Vec2::new(0.0, 2.0));
    particles.prev_pos.push(Vec2::new(0.0, 2.0));
    particles.vel.push(Vec2::ZERO);
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();

    let sub_block = world_pos_to_sub_block(
        particles.pos[0],
        particles.solver_params.sub_block_size_cells.max(1),
    );
    particles.particle_sub_block_coords[0] = sub_block;
    particles.particle_sub_block_update_mask[0] = true;
    let state = particles.sub_block_states.entry(sub_block).or_default();
    state.scheduled_this_substep = true;
    state.boundary_debt_impulse = Vec2::new(0.2, 0.0);
    state.rate_divisor = rate_divisor_from_level(1);

    let dt_sub = particles.solver_params.fixed_dt / particles.solver_params.substeps as f32;
    particles.apply_sub_block_debt_before_constraints(dt_sub);

    assert!(particles.vel[0].x > 0.0);
    assert!(particles.pos[0].x > 0.0);
    assert!(
        particles
            .sub_block_states
            .get(&sub_block)
            .expect("sub-block state should remain")
            .boundary_debt_impulse
            .length()
            <= 1e-6
    );
}

#[test]
fn near_field_boundary_crossing_skips_far_field_boundary_buffer() {
    let mut particles = ParticleWorld::default();
    clear_particles(&mut particles);
    let center_chunk = IVec2::ZERO;
    let prev_pos = Vec2::new(CHUNK_WORLD_SIZE_M * 1.2, 0.5);
    let now_pos = Vec2::new(CHUNK_WORLD_SIZE_M * 2.2, 0.5);
    particles.pos.push(now_pos);
    particles.prev_pos.push(prev_pos);
    particles.vel.push(Vec2::new(0.5, 0.0));
    particles
        .mass
        .push(particle_properties(ParticleMaterial::WaterLiquid).mass);
    particles.material.push(ParticleMaterial::WaterLiquid);
    particles.resize_work_buffers();
    particles.configure_far_field_queue(Some(center_chunk), 3, 1, 0, 0.0, 0);
    particles.active_halo_chunks = 1;

    let mut object_world = ObjectWorld::default();
    particles.capture_far_field_intrusions(&mut object_world);

    assert_eq!(particles.particle_count(), 1);
    assert!(particles.deferred_boundary_particles.is_empty());
}
