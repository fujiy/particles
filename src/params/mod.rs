// パラメータ資産管理モジュール (PARAM-01)
//
// `assets/params/` 配下の RON ファイルを Bevy Asset として管理し、
// hot reload で実行中に値を更新する。
//
// 3 ファイル構成:
//   physics.ron    — GPU MPM 物性・境界・連成
//   render.ron     — 水ドット・地形ドット描画
//   palette.ron    — 水ドット材質パレット
pub mod palette;
pub mod physics;
pub mod render;

use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;

use palette::{PaletteParams, PaletteParamsLoader};
use physics::{PhysicsParams, PhysicsParamsLoader};
use render::{RenderParams, RenderParamsLoader};

// ---------------------------------------------------------------------------
// Handle resources (アセットロード追跡用)
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct PhysicsParamsHandle(pub Handle<PhysicsParams>);
#[derive(Resource, Default)]
pub struct RenderParamsHandle(pub Handle<RenderParams>);
#[derive(Resource, Default)]
pub struct PaletteParamsHandle(pub Handle<PaletteParams>);
// ---------------------------------------------------------------------------
// Active resources (最後に検証済みのスナップショット)
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct ActivePhysicsParams(pub PhysicsParams);
#[derive(Resource, Clone, ExtractResource)]
pub struct ActiveRenderParams(pub RenderParams);
#[derive(Resource, Clone, ExtractResource)]
pub struct ActivePaletteParams(pub PaletteParams);
impl Default for ActivePhysicsParams {
    fn default() -> Self {
        Self(PhysicsParams::default())
    }
}
impl Default for ActiveRenderParams {
    fn default() -> Self {
        Self(RenderParams::default())
    }
}
impl Default for ActivePaletteParams {
    fn default() -> Self {
        Self(PaletteParams::default())
    }
}
// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct ParamsPlugin;

impl Plugin for ParamsPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<PhysicsParams>()
            .init_asset::<RenderParams>()
            .init_asset::<PaletteParams>()
            .register_asset_loader(PhysicsParamsLoader)
            .register_asset_loader(RenderParamsLoader)
            .register_asset_loader(PaletteParamsLoader)
            .init_resource::<PhysicsParamsHandle>()
            .init_resource::<RenderParamsHandle>()
            .init_resource::<PaletteParamsHandle>()
            .init_resource::<ActivePhysicsParams>()
            .init_resource::<ActiveRenderParams>()
            .init_resource::<ActivePaletteParams>()
            .add_systems(Startup, load_all_params)
            .add_systems(
                Update,
                (hot_reload_physics, hot_reload_render, hot_reload_palette),
            );
    }
}

// ---------------------------------------------------------------------------
// Startup system
// ---------------------------------------------------------------------------

fn load_all_params(
    asset_server: Res<AssetServer>,
    mut physics_h: ResMut<PhysicsParamsHandle>,
    mut render_h: ResMut<RenderParamsHandle>,
    mut palette_h: ResMut<PaletteParamsHandle>,
) {
    physics_h.0 = asset_server.load("params/physics.ron");
    render_h.0 = asset_server.load("params/render.ron");
    palette_h.0 = asset_server.load("params/palette.ron");
}

// ---------------------------------------------------------------------------
// Hot reload systems
// ---------------------------------------------------------------------------

fn hot_reload_physics(
    handle: Res<PhysicsParamsHandle>,
    assets: Res<Assets<PhysicsParams>>,
    mut active: ResMut<ActivePhysicsParams>,
    mut events: MessageReader<AssetEvent<PhysicsParams>>,
) {
    apply_if_valid(
        &handle.0,
        &assets,
        &mut active.0,
        &mut events,
        "physics.ron",
    );
}

fn hot_reload_render(
    handle: Res<RenderParamsHandle>,
    assets: Res<Assets<RenderParams>>,
    mut active: ResMut<ActiveRenderParams>,
    mut events: MessageReader<AssetEvent<RenderParams>>,
) {
    apply_if_valid(&handle.0, &assets, &mut active.0, &mut events, "render.ron");
}

fn hot_reload_palette(
    handle: Res<PaletteParamsHandle>,
    assets: Res<Assets<PaletteParams>>,
    mut active: ResMut<ActivePaletteParams>,
    mut events: MessageReader<AssetEvent<PaletteParams>>,
) {
    apply_if_valid(
        &handle.0,
        &assets,
        &mut active.0,
        &mut events,
        "palette.ron",
    );
}

// ---------------------------------------------------------------------------
// 共通ヘルパー
// ---------------------------------------------------------------------------

trait Validatable: Clone {
    fn validate(&self) -> Result<(), String>;
}

impl Validatable for PhysicsParams {
    fn validate(&self) -> Result<(), String> {
        self.validate()
    }
}
impl Validatable for RenderParams {
    fn validate(&self) -> Result<(), String> {
        self.validate()
    }
}
impl Validatable for PaletteParams {
    fn validate(&self) -> Result<(), String> {
        self.validate()
    }
}
fn apply_if_valid<A>(
    handle: &Handle<A>,
    assets: &Assets<A>,
    active: &mut A,
    events: &mut MessageReader<AssetEvent<A>>,
    label: &str,
) where
    A: Asset + Validatable,
{
    for event in events.read() {
        let id = match event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => *id,
            _ => continue,
        };
        if id != handle.id() {
            continue;
        }
        let Some(params) = assets.get(id) else {
            continue;
        };
        match params.validate() {
            Ok(()) => {
                *active = params.clone();
                info!("[params] {label} を再ロード: 有効");
            }
            Err(msg) => {
                warn!("[params] {label} の値が不正 — 直前の有効値を維持: {msg}");
            }
        }
    }
}
