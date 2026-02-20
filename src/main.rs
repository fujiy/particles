mod camera_controller;
mod interface;
mod overlay;
mod physics;
mod render;

use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use camera_controller::CameraControllerPlugin;
use interface::InterfacePlugin;
use overlay::OverlayPlugin;
use physics::PhysicsPlugin;
use render::RenderPlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.1)))
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins((
            PhysicsPlugin,
            InterfacePlugin,
            OverlayPlugin,
            RenderPlugin,
            CameraControllerPlugin,
        ))
        .run();
}
