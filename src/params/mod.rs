// パラメータ資産管理モジュール (PARAM-01)
//
// `assets/params/` 配下の RON ファイルを Bevy Asset として管理し、
// hot reload で実行中に値を更新する。
//
// 5 ファイル構成:
//   physics.ron    — GPU MPM 物性・境界・連成
//   render.ron     — 水ドット・地形ドット描画
//   palette.ron    — 水ドット材質パレット
//   overlay.ron    — SDF・グリッドオーバーレイ閾値
//   material.ron   — 材料種ごとの摩擦・反発・破壊閾値
//   generation.ron — 地形生成・ノイズ・分布

pub mod generation;
pub mod material;
pub mod overlay;
pub mod palette;
pub mod physics;
pub mod render;

use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;

use generation::{GenerationParams, GenerationParamsLoader};
use material::{MaterialAssetParams, MaterialAssetParamsLoader};
use overlay::{OverlayParams, OverlayParamsLoader};
use palette::{PaletteParams, PaletteParamsLoader};
use physics::{PhysicsParams, PhysicsParamsLoader};
use render::{RenderParams, RenderParamsLoader};

// ---------------------------------------------------------------------------
// Handle resources (アセットロード追跡用)
// ---------------------------------------------------------------------------

#[derive(Resource, Default)] pub struct PhysicsParamsHandle(pub Handle<PhysicsParams>);
#[derive(Resource, Default)] pub struct RenderParamsHandle(pub Handle<RenderParams>);
#[derive(Resource, Default)] pub struct PaletteParamsHandle(pub Handle<PaletteParams>);
#[derive(Resource, Default)] pub struct OverlayParamsHandle(pub Handle<OverlayParams>);
#[derive(Resource, Default)] pub struct MaterialAssetParamsHandle(pub Handle<MaterialAssetParams>);
#[derive(Resource, Default)] pub struct GenerationParamsHandle(pub Handle<GenerationParams>);

// ---------------------------------------------------------------------------
// Active resources (最後に検証済みのスナップショット)
// ---------------------------------------------------------------------------

#[derive(Resource)] pub struct ActivePhysicsParams(pub PhysicsParams);
#[derive(Resource)] pub struct ActiveRenderParams(pub RenderParams);
#[derive(Resource, Clone, ExtractResource)] pub struct ActivePaletteParams(pub PaletteParams);
#[derive(Resource)] pub struct ActiveOverlayParams(pub OverlayParams);
#[derive(Resource)] pub struct ActiveMaterialAssetParams(pub MaterialAssetParams);
#[derive(Resource)] pub struct ActiveGenerationParams(pub GenerationParams);

impl Default for ActivePhysicsParams       { fn default() -> Self { Self(PhysicsParams::default()) } }
impl Default for ActiveRenderParams        { fn default() -> Self { Self(RenderParams::default()) } }
impl Default for ActivePaletteParams       { fn default() -> Self { Self(PaletteParams::default()) } }
impl Default for ActiveOverlayParams       { fn default() -> Self { Self(OverlayParams::default()) } }
impl Default for ActiveMaterialAssetParams { fn default() -> Self { Self(MaterialAssetParams::default()) } }
impl Default for ActiveGenerationParams    { fn default() -> Self { Self(GenerationParams::default()) } }

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct ParamsPlugin;

impl Plugin for ParamsPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_asset::<PhysicsParams>()
            .init_asset::<RenderParams>()
            .init_asset::<PaletteParams>()
            .init_asset::<OverlayParams>()
            .init_asset::<MaterialAssetParams>()
            .init_asset::<GenerationParams>()
            .register_asset_loader(PhysicsParamsLoader)
            .register_asset_loader(RenderParamsLoader)
            .register_asset_loader(PaletteParamsLoader)
            .register_asset_loader(OverlayParamsLoader)
            .register_asset_loader(MaterialAssetParamsLoader)
            .register_asset_loader(GenerationParamsLoader)
            .init_resource::<PhysicsParamsHandle>()
            .init_resource::<RenderParamsHandle>()
            .init_resource::<PaletteParamsHandle>()
            .init_resource::<OverlayParamsHandle>()
            .init_resource::<MaterialAssetParamsHandle>()
            .init_resource::<GenerationParamsHandle>()
            .init_resource::<ActivePhysicsParams>()
            .init_resource::<ActiveRenderParams>()
            .init_resource::<ActivePaletteParams>()
            .init_resource::<ActiveOverlayParams>()
            .init_resource::<ActiveMaterialAssetParams>()
            .init_resource::<ActiveGenerationParams>()
            .add_systems(Startup, load_all_params)
            .add_systems(Update, (
                hot_reload_physics,
                hot_reload_render,
                hot_reload_palette,
                hot_reload_overlay,
                hot_reload_material,
                hot_reload_generation,
            ));
    }
}

// ---------------------------------------------------------------------------
// Startup system
// ---------------------------------------------------------------------------

fn load_all_params(
    asset_server: Res<AssetServer>,
    mut physics_h:    ResMut<PhysicsParamsHandle>,
    mut render_h:     ResMut<RenderParamsHandle>,
    mut palette_h:    ResMut<PaletteParamsHandle>,
    mut overlay_h:    ResMut<OverlayParamsHandle>,
    mut material_h:   ResMut<MaterialAssetParamsHandle>,
    mut generation_h: ResMut<GenerationParamsHandle>,
) {
    physics_h.0    = asset_server.load("params/physics.ron");
    render_h.0     = asset_server.load("params/render.ron");
    palette_h.0    = asset_server.load("params/palette.ron");
    overlay_h.0    = asset_server.load("params/overlay.ron");
    material_h.0   = asset_server.load("params/material.ron");
    generation_h.0 = asset_server.load("params/generation.ron");
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
    apply_if_valid(&handle.0, &assets, &mut active.0, &mut events, "physics.ron");
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
    apply_if_valid(&handle.0, &assets, &mut active.0, &mut events, "palette.ron");
}

fn hot_reload_overlay(
    handle: Res<OverlayParamsHandle>,
    assets: Res<Assets<OverlayParams>>,
    mut active: ResMut<ActiveOverlayParams>,
    mut events: MessageReader<AssetEvent<OverlayParams>>,
) {
    apply_if_valid(&handle.0, &assets, &mut active.0, &mut events, "overlay.ron");
}

fn hot_reload_material(
    handle: Res<MaterialAssetParamsHandle>,
    assets: Res<Assets<MaterialAssetParams>>,
    mut active: ResMut<ActiveMaterialAssetParams>,
    mut events: MessageReader<AssetEvent<MaterialAssetParams>>,
) {
    apply_if_valid(&handle.0, &assets, &mut active.0, &mut events, "material.ron");
}

fn hot_reload_generation(
    handle: Res<GenerationParamsHandle>,
    assets: Res<Assets<GenerationParams>>,
    mut active: ResMut<ActiveGenerationParams>,
    mut events: MessageReader<AssetEvent<GenerationParams>>,
) {
    apply_if_valid(&handle.0, &assets, &mut active.0, &mut events, "generation.ron");
}

// ---------------------------------------------------------------------------
// 共通ヘルパー
// ---------------------------------------------------------------------------

trait Validatable: Clone {
    fn validate(&self) -> Result<(), String>;
}

impl Validatable for PhysicsParams       { fn validate(&self) -> Result<(), String> { self.validate() } }
impl Validatable for RenderParams        { fn validate(&self) -> Result<(), String> { self.validate() } }
impl Validatable for PaletteParams       { fn validate(&self) -> Result<(), String> { self.validate() } }
impl Validatable for OverlayParams       { fn validate(&self) -> Result<(), String> { self.validate() } }
impl Validatable for MaterialAssetParams { fn validate(&self) -> Result<(), String> { self.validate() } }
impl Validatable for GenerationParams    { fn validate(&self) -> Result<(), String> { self.validate() } }

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
