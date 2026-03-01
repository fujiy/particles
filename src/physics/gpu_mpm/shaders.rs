// Shader asset handles for GPU MPM compute passes.

use bevy::prelude::*;

#[derive(Resource)]
pub struct MpmShaders {
    pub types: Handle<Shader>,
    pub clear: Handle<Shader>,
    pub p2g: Handle<Shader>,
    pub grid_update: Handle<Shader>,
    pub g2p: Handle<Shader>,
    pub drift: Handle<Shader>,
}

impl FromWorld for MpmShaders {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        Self {
            types: asset_server.load("shaders/mpm_types.wgsl"),
            clear: asset_server.load("shaders/mpm_clear.wgsl"),
            p2g: asset_server.load("shaders/mpm_p2g.wgsl"),
            grid_update: asset_server.load("shaders/mpm_grid_update.wgsl"),
            g2p: asset_server.load("shaders/mpm_g2p.wgsl"),
            drift: asset_server.load("shaders/mpm_drift.wgsl"),
        }
    }
}
