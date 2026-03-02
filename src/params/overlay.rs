// assets/params/overlay.ron の Asset 型定義とローダー。
//
// オーバーレイ描画の閾値・スケールパラメータを管理する。
// UI 色やボタン位置等レイアウト依存の定数は対象外。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// サブ構造体
// ---------------------------------------------------------------------------

/// SDF オーバーレイの描画スケールパラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdfOverlayParams {
    /// SDF 等値線のステップ幅 [セル単位の倍率]. 許容: 0.1 – 10.0
    pub step_cell_scale: f32,
    /// SDF 等値線の円半径 [セル単位の倍率]. 許容: 0.01 – 1.0
    pub cell_radius_scale: f32,
    /// SDF 表示範囲 [セル単位の倍率]. 許容: 1.0 – 30.0
    pub range_cell_scale: f32,
    /// 地形内部塗り潰しの Z オフセット. 許容: 0.0 – 1.0
    pub negative_fill_z: f32,
}

/// グリッドオーバーレイのスケールパラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridOverlayParams {
    /// オブジェクト軸線の長さ [セル単位の倍率]. 許容: 0.1 – 5.0
    pub object_axis_length_cell_scale: f32,
    /// オブジェクト中心点の半径 [セル単位の倍率]. 許容: 0.01 – 0.5
    pub object_center_radius_cell_scale: f32,
}

// ---------------------------------------------------------------------------
// トップレベル Asset 型
// ---------------------------------------------------------------------------

/// オーバーレイ描画パラメータ。`assets/params/overlay.ron` から読み込まれる。
#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct OverlayParams {
    /// SDF オーバーレイパラメータ
    pub sdf: SdfOverlayParams,
    /// グリッドオーバーレイパラメータ
    pub grid: GridOverlayParams,
}

impl Default for OverlayParams {
    fn default() -> Self {
        Self {
            sdf: SdfOverlayParams {
                step_cell_scale: 1.0,
                cell_radius_scale: 0.45,
                range_cell_scale: 6.0,
                negative_fill_z: 0.2,
            },
            grid: GridOverlayParams {
                object_axis_length_cell_scale: 1.6,
                object_center_radius_cell_scale: 0.12,
            },
        }
    }
}

impl OverlayParams {
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
        check!(self.sdf.step_cell_scale, "sdf.step_cell_scale", 0.1, 10.0);
        check!(self.sdf.cell_radius_scale, "sdf.cell_radius_scale", 0.01, 1.0);
        check!(self.sdf.range_cell_scale, "sdf.range_cell_scale", 1.0, 30.0);
        check!(self.sdf.negative_fill_z, "sdf.negative_fill_z", 0.0, 1.0);
        check!(self.grid.object_axis_length_cell_scale, "grid.object_axis_length_cell_scale", 0.1, 5.0);
        check!(self.grid.object_center_radius_cell_scale, "grid.object_center_radius_cell_scale", 0.01, 0.5);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Asset loader
// ---------------------------------------------------------------------------

#[derive(Default, bevy::reflect::TypePath)]
pub struct OverlayParamsLoader;

impl AssetLoader for OverlayParamsLoader {
    type Asset = OverlayParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<OverlayParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: OverlayParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
