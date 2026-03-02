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
}

/// 地形ドット描画パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainRenderParams {
    // 現在調整可能なパラメータなし（dots_per_cell はバッファサイズ依存で対象外）
    // 将来的な拡張のためセクションを確保
    /// 地形パレットシード（デバッグ用色の再現性）. 許容: 任意
    pub palette_seed: u32,
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
            },
            terrain: TerrainRenderParams {
                palette_seed: 0x5EED_7163,
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
        check!(s.density_threshold, "water.splat.density_threshold", 0.01, 1.0);
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
