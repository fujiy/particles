use crate::physics::cell_to_world_center;
use crate::physics::state::SimUpdateSet;
use bevy::camera::ScalingMode;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

const CAMERA_VIEWPORT_HEIGHT_M: f32 = 14.0;
const CAMERA_PAN_SPEED_MPS: f32 = 10.0;
const CAMERA_ZOOM_SENSITIVITY: f32 = 0.06;
const CAMERA_MIN_ZOOM: f32 = 0.1;
const CAMERA_MAX_ZOOM: f32 = 1000.0;
const CAMERA_START_CELL: IVec2 = IVec2::new(0, 8);
const CAMERA_START_Z: f32 = 100.0;

pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_main_camera)
            .add_systems(Update, control_main_camera.in_set(SimUpdateSet::Controls));
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

fn setup_main_camera(mut commands: Commands) {
    let start_xy = cell_to_world_center(CAMERA_START_CELL);
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: CAMERA_VIEWPORT_HEIGHT_M,
            },
            scale: 1.0,
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_xyz(start_xy.x, start_xy.y, CAMERA_START_Z),
        MainCamera,
        CameraController {
            pan_speed_mps: CAMERA_PAN_SPEED_MPS,
            zoom_sensitivity: CAMERA_ZOOM_SENSITIVITY,
            min_zoom: CAMERA_MIN_ZOOM,
            max_zoom: CAMERA_MAX_ZOOM,
            viewport_height_m: CAMERA_VIEWPORT_HEIGHT_M,
        },
    ));
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
