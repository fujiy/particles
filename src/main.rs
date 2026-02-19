mod camera_controller;
mod simulation;

use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use camera_controller::CameraControllerPlugin;
use simulation::SimulationPlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.1)))
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins((SimulationPlugin, CameraControllerPlugin))
        .run();
}
