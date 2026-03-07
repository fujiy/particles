// パラメータ資産管理モジュール (PARAM-01)
//
// `assets/params/` 配下の RON ファイルを Bevy Asset として管理し、
// hot reload で実行中に値を更新する。
//
// 4 ファイル構成:
//   camera.ron     — カメラ初期位置・操作感度
//   interface.ron  — UI/操作/配色
//   physics.ron    — GPU MPM 物性・境界・連成
//   render.ron     — 水ドット・地形ドット描画
//   palette.ron    — 水ドット材質パレット
//   overlay.ron    — Overlay UI / SDF 描画パラメータ
pub mod camera;
pub mod interface;
pub mod overlay;
pub mod palette;
pub mod physics;
pub mod render;

use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;

use camera::{CameraParams, CameraParamsLoader};
use interface::{InterfaceParams, InterfaceParamsLoader};
use overlay::{OverlayParams, OverlayParamsLoader};
use palette::{PaletteParams, PaletteParamsLoader};
use physics::{PhysicsParams, PhysicsParamsLoader};
use render::{RenderParams, RenderParamsLoader};

// ---------------------------------------------------------------------------
// Handle resources (アセットロード追跡用)
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct CameraParamsHandle(pub Handle<CameraParams>);
#[derive(Resource, Default)]
pub struct InterfaceParamsHandle(pub Handle<InterfaceParams>);
#[derive(Resource, Default)]
pub struct PhysicsParamsHandle(pub Handle<PhysicsParams>);
#[derive(Resource, Default)]
pub struct RenderParamsHandle(pub Handle<RenderParams>);
#[derive(Resource, Default)]
pub struct PaletteParamsHandle(pub Handle<PaletteParams>);
#[derive(Resource, Default)]
pub struct OverlayParamsHandle(pub Handle<OverlayParams>);
// ---------------------------------------------------------------------------
// Active resources (最後に検証済みのスナップショット)
// ---------------------------------------------------------------------------

#[derive(Resource, Clone)]
pub struct ActiveCameraParams(pub CameraParams);
#[derive(Resource, Clone)]
pub struct ActiveInterfaceParams(pub InterfaceParams);
#[derive(Resource)]
pub struct ActivePhysicsParams(pub PhysicsParams);
#[derive(Resource, Clone, ExtractResource)]
pub struct ActiveRenderParams(pub RenderParams);
#[derive(Resource, Clone, ExtractResource)]
pub struct ActivePaletteParams(pub PaletteParams);
#[derive(Resource, Clone, ExtractResource)]
pub struct ActiveOverlayParams(pub OverlayParams);
impl Default for ActiveCameraParams {
    fn default() -> Self {
        Self(CameraParams::default())
    }
}
impl Default for ActiveInterfaceParams {
    fn default() -> Self {
        Self(InterfaceParams::default())
    }
}
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
impl Default for ActiveOverlayParams {
    fn default() -> Self {
        Self(OverlayParams::default())
    }
}
// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct ParamsPlugin;

impl Plugin for ParamsPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<CameraParams>()
            .init_asset::<InterfaceParams>()
            .init_asset::<PhysicsParams>()
            .init_asset::<RenderParams>()
            .init_asset::<PaletteParams>()
            .init_asset::<OverlayParams>()
            .register_asset_loader(CameraParamsLoader)
            .register_asset_loader(InterfaceParamsLoader)
            .register_asset_loader(PhysicsParamsLoader)
            .register_asset_loader(RenderParamsLoader)
            .register_asset_loader(PaletteParamsLoader)
            .register_asset_loader(OverlayParamsLoader)
            .init_resource::<CameraParamsHandle>()
            .init_resource::<InterfaceParamsHandle>()
            .init_resource::<PhysicsParamsHandle>()
            .init_resource::<RenderParamsHandle>()
            .init_resource::<PaletteParamsHandle>()
            .init_resource::<OverlayParamsHandle>()
            .init_resource::<ActiveCameraParams>()
            .init_resource::<ActiveInterfaceParams>()
            .init_resource::<ActivePhysicsParams>()
            .init_resource::<ActiveRenderParams>()
            .init_resource::<ActivePaletteParams>()
            .init_resource::<ActiveOverlayParams>()
            .add_systems(Startup, load_all_params)
            .add_systems(
                Update,
                (
                    hot_reload_camera,
                    hot_reload_interface,
                    hot_reload_physics,
                    hot_reload_render,
                    hot_reload_palette,
                    hot_reload_overlay,
                ),
            );
    }
}

// ---------------------------------------------------------------------------
// Startup system
// ---------------------------------------------------------------------------

fn load_all_params(
    asset_server: Res<AssetServer>,
    mut camera_h: ResMut<CameraParamsHandle>,
    mut interface_h: ResMut<InterfaceParamsHandle>,
    mut physics_h: ResMut<PhysicsParamsHandle>,
    mut render_h: ResMut<RenderParamsHandle>,
    mut palette_h: ResMut<PaletteParamsHandle>,
    mut overlay_h: ResMut<OverlayParamsHandle>,
) {
    camera_h.0 = asset_server.load("params/camera.ron");
    interface_h.0 = asset_server.load("params/interface.ron");
    physics_h.0 = asset_server.load("params/physics.ron");
    render_h.0 = asset_server.load("params/render.ron");
    palette_h.0 = asset_server.load("params/palette.ron");
    overlay_h.0 = asset_server.load("params/overlay.ron");
}

// ---------------------------------------------------------------------------
// Hot reload systems
// ---------------------------------------------------------------------------

fn hot_reload_camera(
    handle: Res<CameraParamsHandle>,
    assets: Res<Assets<CameraParams>>,
    mut active: ResMut<ActiveCameraParams>,
    mut events: MessageReader<AssetEvent<CameraParams>>,
) {
    apply_if_valid(&handle.0, &assets, &mut active.0, &mut events, "camera.ron");
}

fn hot_reload_interface(
    handle: Res<InterfaceParamsHandle>,
    assets: Res<Assets<InterfaceParams>>,
    mut active: ResMut<ActiveInterfaceParams>,
    mut events: MessageReader<AssetEvent<InterfaceParams>>,
) {
    apply_if_valid(
        &handle.0,
        &assets,
        &mut active.0,
        &mut events,
        "interface.ron",
    );
}

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

fn hot_reload_overlay(
    handle: Res<OverlayParamsHandle>,
    assets: Res<Assets<OverlayParams>>,
    mut active: ResMut<ActiveOverlayParams>,
    mut events: MessageReader<AssetEvent<OverlayParams>>,
) {
    apply_if_valid(
        &handle.0,
        &assets,
        &mut active.0,
        &mut events,
        "overlay.ron",
    );
}

// ---------------------------------------------------------------------------
// 共通ヘルパー
// ---------------------------------------------------------------------------

trait Validatable: Clone {
    fn validate(&self) -> Result<(), String>;
}

impl Validatable for CameraParams {
    fn validate(&self) -> Result<(), String> {
        self.validate()
    }
}
impl Validatable for InterfaceParams {
    fn validate(&self) -> Result<(), String> {
        self.validate()
    }
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
impl Validatable for OverlayParams {
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
