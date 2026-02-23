use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::log::{BoxedLayer, LogPlugin};
use bevy::prelude::*;
use bevy_inspector_egui::bevy_egui::EguiPlugin;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use particles::camera_controller::CameraControllerPlugin;
use particles::interface::InterfacePlugin;
use particles::overlay::OverlayPlugin;
use particles::physics::PhysicsPlugin;
use particles::render::RenderPlugin;

fn tracy_layer(_app: &mut App) -> Option<BoxedLayer> {
    #[cfg(feature = "tracy")]
    {
        Some(Box::new(tracing_tracy::TracyLayer::default()))
    }
    #[cfg(not(feature = "tracy"))]
    {
        None
    }
}

fn main() {
    let default_plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(LogPlugin {
            custom_layer: tracy_layer,
            ..default()
        });
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.1)))
        .add_plugins(default_plugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(EguiPlugin::default())
        .add_plugins(WorldInspectorPlugin::new());
    app.add_plugins((
        PhysicsPlugin,
        InterfacePlugin,
        OverlayPlugin,
        RenderPlugin,
        CameraControllerPlugin,
    ))
    .run();
}
