// assets/params/render.ron の Asset 型定義とローダー。
//
// 水ドット・地形ドットの描画パラメータを管理する。
// バッファサイズやワークグループサイズ等レイアウト依存の定数は対象外。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// サブ構造体
// ---------------------------------------------------------------------------

/// 水ドット前処理の密度スプラット・ブラーパラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterSplatParams {
    /// 密度スプラットの半径 [ドット単位]. 許容: 0.5 – 10.0
    pub radius_dots: f32,
    /// ガウスブラーのシグマ [ドット単位]. 許容: 0.5 – 20.0
    pub blur_sigma_dots: f32,
    /// ブラーカーネルの半径 [ドット]. 許容: 1 – 30
    pub blur_radius_dots: u32,
    /// 密度アトミック蓄積スケール（精度調整用）. 許容: 16.0 – 4096.0
    pub atomic_scale: f32,
    /// 水面判定の密度閾値. 許容: 0.01 – 1.0
    pub density_threshold: f32,
}

/// 水ドット描画パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterRenderParams {
    /// 前処理（スプラット・ブラー）パラメータ
    pub splat: WaterSplatParams,
    /// 水ドット側のハッシュシード。許容: 任意
    pub palette_seed: u32,
}

/// 地形 Near/Far LOD 設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainLodRenderParams {
    /// Near キャッシュの画面余白倍率. 許容: 1.0 – 4.0
    pub near_margin_factor: f32,
    /// Near 品質分母（大きいほど解像度を下げる）. 許容: 1.0 – 8.0
    pub near_quality_divisor: f32,
    /// Far キャッシュの画面余白倍率. 許容: 1.0 – 6.0
    pub far_margin_factor: f32,
    /// Far 品質分母（大きいほど解像度を下げる）. 許容: 1.0 – 8.0
    pub far_quality_divisor: f32,
    /// Far 解像度の追加ヘッドルーム係数. 許容: 1.0 – 2.0
    pub far_quality_headroom: f32,
    /// Far downsample の最小値. 許容: 1 – 64
    pub far_min_downsample: u32,
}

/// 地形 Back レイヤ設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainBackRenderParams {
    /// Back が維持する最小画面解像度の分母（2.0 なら screen_px / 2 以上）。許容: 1.0 – 8.0
    pub min_screen_resolution_divisor: f32,
    /// Back の視差スケール係数（大きいほど遠景の縮尺差が強くなる）。許容: 0.25 – 8.0
    pub scale_multiplier: f32,
    /// 空気遠近ティント量. 許容: 0.0 – 1.0
    pub atmosphere_tint: f32,
}

/// sRGB 8bit 相当の色指定（0–255）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorSrgb8Params {
    /// 赤チャンネル. 許容: 0.0 – 255.0
    pub r: f32,
    /// 緑チャンネル. 許容: 0.0 – 255.0
    pub g: f32,
    /// 青チャンネル. 許容: 0.0 – 255.0
    pub b: f32,
}

/// 地形ドット描画パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainRenderParams {
    /// 地形パレットシード（デバッグ用色の再現性）. 許容: 任意
    pub palette_seed: u32,
    /// Near/Far LOD 設定
    pub lod: TerrainLodRenderParams,
    /// Back レイヤ設定
    pub back: TerrainBackRenderParams,
    /// 全体背景の空色
    pub sky_color: ColorSrgb8Params,
}

// ---------------------------------------------------------------------------
// トップレベル Asset 型
// ---------------------------------------------------------------------------

/// 描画パラメータ。`assets/params/render.ron` から読み込まれる。
#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct RenderParams {
    /// 水ドット描画パラメータ
    pub water: WaterRenderParams,
    /// 地形ドット描画パラメータ
    pub terrain: TerrainRenderParams,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            water: WaterRenderParams {
                splat: WaterSplatParams {
                    radius_dots: 2.5,
                    blur_sigma_dots: 5.0,
                    blur_radius_dots: 10,
                    atomic_scale: 256.0,
                    density_threshold: 0.50,
                },
                palette_seed: 0xA53C_9E4D,
            },
            terrain: TerrainRenderParams {
                palette_seed: 0x5EED_7163,
                lod: TerrainLodRenderParams {
                    near_margin_factor: 1.35,
                    near_quality_divisor: 1.0,
                    far_margin_factor: 2.5,
                    far_quality_divisor: 2.0,
                    far_quality_headroom: 1.1,
                    far_min_downsample: 1,
                },
                back: TerrainBackRenderParams {
                    min_screen_resolution_divisor: 2.0,
                    scale_multiplier: 2.0,
                    atmosphere_tint: 0.35,
                },
                sky_color: ColorSrgb8Params {
                    r: 164.0,
                    g: 208.0,
                    b: 255.0,
                },
            },
        }
    }
}

impl RenderParams {
    pub fn validate(&self) -> Result<(), String> {
        macro_rules! check {
            ($val:expr, $name:expr, $lo:expr, $hi:expr) => {
                if !($val >= $lo && $val <= $hi) {
                    return Err(format!(
                        "{}: {} は [{}, {}] の範囲外",
                        $name, $val, $lo, $hi
                    ));
                }
            };
        }
        let s = &self.water.splat;
        check!(s.radius_dots, "water.splat.radius_dots", 0.5, 10.0);
        check!(s.blur_sigma_dots, "water.splat.blur_sigma_dots", 0.5, 20.0);
        check!(s.blur_radius_dots, "water.splat.blur_radius_dots", 1, 30);
        check!(s.atomic_scale, "water.splat.atomic_scale", 16.0, 4096.0);
        check!(
            s.density_threshold,
            "water.splat.density_threshold",
            0.01,
            1.0
        );
        let lod = &self.terrain.lod;
        check!(
            lod.near_margin_factor,
            "terrain.lod.near_margin_factor",
            1.0,
            4.0
        );
        check!(
            lod.near_quality_divisor,
            "terrain.lod.near_quality_divisor",
            1.0,
            8.0
        );
        check!(
            lod.far_margin_factor,
            "terrain.lod.far_margin_factor",
            1.0,
            6.0
        );
        check!(
            lod.far_quality_divisor,
            "terrain.lod.far_quality_divisor",
            1.0,
            8.0
        );
        check!(
            lod.far_quality_headroom,
            "terrain.lod.far_quality_headroom",
            1.0,
            2.0
        );
        check!(
            lod.far_min_downsample,
            "terrain.lod.far_min_downsample",
            1,
            64
        );
        let back = &self.terrain.back;
        check!(
            back.min_screen_resolution_divisor,
            "terrain.back.min_screen_resolution_divisor",
            1.0,
            8.0
        );
        check!(
            back.scale_multiplier,
            "terrain.back.scale_multiplier",
            0.25,
            8.0
        );
        check!(
            back.atmosphere_tint,
            "terrain.back.atmosphere_tint",
            0.0,
            1.0
        );
        check!(self.terrain.sky_color.r, "terrain.sky_color.r", 0.0, 255.0);
        check!(self.terrain.sky_color.g, "terrain.sky_color.g", 0.0, 255.0);
        check!(self.terrain.sky_color.b, "terrain.sky_color.b", 0.0, 255.0);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Asset loader
// ---------------------------------------------------------------------------

#[derive(Default, bevy::reflect::TypePath)]
pub struct RenderParamsLoader;

impl AssetLoader for RenderParamsLoader {
    type Asset = RenderParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<RenderParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: RenderParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
