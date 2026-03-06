use crate::params::ActiveCameraParams;
use crate::physics::cell_to_world_center;
use crate::physics::state::SimUpdateSet;
use bevy::camera::ScalingMode;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_main_camera).add_systems(
            Update,
            (apply_camera_params, control_main_camera)
                .chain()
                .in_set(SimUpdateSet::Controls),
        );
    }
}

#[derive(Component)]
pub struct MainCamera;

#[derive(Component)]
struct CameraController {
    pan_speed_mps: f32,
    zoom_sensitivity: f32,
    min_zoom: f32,
    max_zoom: f32,
    viewport_height_m: f32,
}

fn setup_main_camera(mut commands: Commands, camera_params: Res<ActiveCameraParams>) {
    let p = &camera_params.0;
    let start_xy = cell_to_world_center(IVec2::new(p.start_cell_x, p.start_cell_y));
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: p.viewport_height_m,
            },
            scale: 1.0,
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_xyz(start_xy.x, start_xy.y, p.start_z),
        MainCamera,
        CameraController {
            pan_speed_mps: p.pan_speed_mps,
            zoom_sensitivity: p.zoom_sensitivity,
            min_zoom: p.min_zoom,
            max_zoom: p.max_zoom,
            viewport_height_m: p.viewport_height_m,
        },
    ));
}

fn apply_camera_params(
    camera_params: Res<ActiveCameraParams>,
    mut cameras: Query<(&mut Projection, &mut CameraController), With<MainCamera>>,
) {
    if !camera_params.is_changed() {
        return;
    }
    let p = &camera_params.0;
    let min_zoom = p.min_zoom.min(p.max_zoom);
    let max_zoom = p.max_zoom.max(p.min_zoom);
    for (mut projection, mut controller) in &mut cameras {
        controller.pan_speed_mps = p.pan_speed_mps;
        controller.zoom_sensitivity = p.zoom_sensitivity;
        controller.min_zoom = min_zoom;
        controller.max_zoom = max_zoom;
        controller.viewport_height_m = p.viewport_height_m;
        if let Projection::Orthographic(orthographic) = projection.as_mut() {
            orthographic.scaling_mode = ScalingMode::FixedVertical {
                viewport_height: p.viewport_height_m,
            };
            orthographic.scale = orthographic.scale.clamp(min_zoom, max_zoom);
        }
    }
}

fn control_main_camera(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<(&mut Transform, &mut Projection, &CameraController), With<MainCamera>>,
) {
    let Some(window) = windows.iter().next() else {
        return;
    };
    let Some((mut transform, mut projection, controller)) = cameras.iter_mut().next() else {
        return;
    };

    let Projection::Orthographic(orthographic) = projection.as_mut() else {
        return;
    };

    let window_size = Vec2::new(window.width().max(1.0), window.height().max(1.0));

    if mouse_scroll.delta.y != 0.0 {
        let old_scale = orthographic.scale;
        let zoom_delta = -mouse_scroll.delta.y * controller.zoom_sensitivity;
        let new_scale =
            (old_scale * (1.0 + zoom_delta)).clamp(controller.min_zoom, controller.max_zoom);

        if let Some(cursor_pos) = window.cursor_position() {
            let world_before = screen_to_world_2d(
                transform.translation.truncate(),
                cursor_pos,
                window_size,
                controller.viewport_height_m,
                old_scale,
            );
            let world_after = screen_to_world_2d(
                transform.translation.truncate(),
                cursor_pos,
                window_size,
                controller.viewport_height_m,
                new_scale,
            );
            transform.translation += (world_before - world_after).extend(0.0);
        }

        orthographic.scale = new_scale;
    }

    if !window.ime_enabled {
        let mut keyboard_pan = Vec2::ZERO;
        if keyboard.pressed(KeyCode::ArrowLeft) || keyboard.pressed(KeyCode::KeyA) {
            keyboard_pan.x -= 1.0;
        }
        if keyboard.pressed(KeyCode::ArrowRight) || keyboard.pressed(KeyCode::KeyD) {
            keyboard_pan.x += 1.0;
        }
        if keyboard.pressed(KeyCode::ArrowDown) || keyboard.pressed(KeyCode::KeyS) {
            keyboard_pan.y -= 1.0;
        }
        if keyboard.pressed(KeyCode::ArrowUp) || keyboard.pressed(KeyCode::KeyW) {
            keyboard_pan.y += 1.0;
        }
        if keyboard_pan.length_squared() > 0.0 {
            let delta = keyboard_pan.normalize()
                * controller.pan_speed_mps
                * orthographic.scale
                * time.delta_secs();
            transform.translation += delta.extend(0.0);
        }
    }

    let alt_pressed = keyboard.pressed(KeyCode::AltLeft) || keyboard.pressed(KeyCode::AltRight);
    let drag_active = mouse_buttons.pressed(MouseButton::Middle)
        || (alt_pressed && mouse_buttons.pressed(MouseButton::Left));

    if drag_active && mouse_motion.delta.length_squared() > 0.0 {
        let world_per_pixel = controller.viewport_height_m * orthographic.scale / window_size.y;
        transform.translation.x -= mouse_motion.delta.x * world_per_pixel;
        transform.translation.y += mouse_motion.delta.y * world_per_pixel;
    }
}

fn screen_to_world_2d(
    camera_world: Vec2,
    cursor_px: Vec2,
    window_size_px: Vec2,
    viewport_height_m: f32,
    zoom_scale: f32,
) -> Vec2 {
    let viewport_world_h = viewport_height_m * zoom_scale;
    let viewport_world_w = viewport_world_h * (window_size_px.x / window_size_px.y);
    let ndc_x = (cursor_px.x / window_size_px.x) - 0.5;
    let ndc_y = 0.5 - (cursor_px.y / window_size_px.y);

    camera_world + Vec2::new(ndc_x * viewport_world_w, ndc_y * viewport_world_h)
}
