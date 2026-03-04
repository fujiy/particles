// assets/params/generation.ron の Asset 型定義とローダー。
//
// 地形生成・ノイズ・分布パラメータを管理する。
// TERRAIN_GENERATOR_VERSION はセーブデータ互換に直結するため対象外。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// サブ構造体
// ---------------------------------------------------------------------------

/// 地形表面高さノイズパラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeightNoiseParams {
    /// 低周波ノイズの周波数 [セル⁻¹]. 許容: 1e-5 – 1.0
    pub freq: f64,
    /// 低周波ノイズの振幅 [セル]. 許容: 0 – 500
    pub amp_cells: i32,
    /// 高周波ディテールの周波数倍率 (freq に掛ける). 許容: 1.0 – 20.0
    pub detail_freq_mult: f64,
    /// 高周波ディテールの振幅 [セル]. 許容: 0.0 – 20.0
    pub detail_amp_cells: f32,
    /// 高周波ディテールの fBm オクターブ数. 許容: 1 – 8
    pub detail_octaves: u32,
}

/// fBm（フラクショナルブラウン運動）パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FbmParams {
    /// オクターブ数. 許容: 1 – 10
    pub octaves: u32,
    /// オクターブ間の周波数倍率. 許容: 1.1 – 4.0
    pub lacunarity: f64,
    /// オクターブ間の振幅倍率. 許容: 0.1 – 0.9
    pub gain: f64,
    /// フットプリント低域通過フィルタの減衰強度. 許容: 0.5 – 10.0
    pub attenuation_scale: f64,
}

/// 地形表面ブレンドバンドパラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceBlendParams {
    /// バンド基本半幅 [セル]. 許容: 0.0 – 5.0
    pub base_cells: f32,
    /// フットプリント 1 セルあたりの追加幅 [セル]. 許容: 0.0 – 5.0
    pub per_footprint: f32,
}

/// 土壌層パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoilLayerParams {
    /// 土壌層の平均深さ [セル]. 許容: 0 – 50
    pub depth_cells: i32,
    /// 土壌深さ変動の振幅 [セル]. 許容: 0.0 – 20.0
    pub depth_variation_cells: f32,
    /// 土壌深さ変動ノイズの周波数倍率 (height_noise.freq に掛ける). 許容: 0.1 – 2.0
    pub depth_noise_freq_mult: f64,
}

// ---------------------------------------------------------------------------
// トップレベル Asset 型
// ---------------------------------------------------------------------------

/// 地形生成パラメータ。`assets/params/generation.ron` から読み込まれる。
#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    /// グローバルワールドシード. 許容: 任意
    pub world_seed: u32,
    /// ベースライン地表面 Y 座標 [セル]. 許容: -1000 – 1000
    pub base_surface_y: i32,
    /// 地形高さノイズ
    pub height_noise: HeightNoiseParams,
    /// fBm パラメータ
    pub fbm: FbmParams,
    /// 地表ブレンドバンド
    pub surface_blend: SurfaceBlendParams,
    /// 土壌層
    pub soil: SoilLayerParams,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            world_seed: 13_370,
            base_surface_y: 0,
            height_noise: HeightNoiseParams {
                freq: 0.01,
                amp_cells: 50,
                detail_freq_mult: 6.0,
                detail_amp_cells: 2.0,
                detail_octaves: 3,
            },
            fbm: FbmParams {
                octaves: 4,
                lacunarity: 2.0,
                gain: 0.5,
                attenuation_scale: 2.2,
            },
            surface_blend: SurfaceBlendParams {
                base_cells: 0.35,
                per_footprint: 0.75,
            },
            soil: SoilLayerParams {
                depth_cells: 4,
                depth_variation_cells: 3.0,
                depth_noise_freq_mult: 0.65,
            },
        }
    }
}

impl GenerationParams {
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
        check!(self.base_surface_y, "base_surface_y", -1000, 1000);
        let h = &self.height_noise;
        check!(h.freq, "height_noise.freq", 1e-5, 1.0);
        check!(h.amp_cells, "height_noise.amp_cells", 0, 500);
        check!(
            h.detail_freq_mult,
            "height_noise.detail_freq_mult",
            1.0,
            20.0
        );
        check!(
            h.detail_amp_cells,
            "height_noise.detail_amp_cells",
            0.0,
            20.0
        );
        check!(h.detail_octaves, "height_noise.detail_octaves", 1, 8);
        let f = &self.fbm;
        check!(f.octaves, "fbm.octaves", 1, 10);
        check!(f.lacunarity, "fbm.lacunarity", 1.1, 4.0);
        check!(f.gain, "fbm.gain", 0.1, 0.9);
        check!(f.attenuation_scale, "fbm.attenuation_scale", 0.5, 10.0);
        let b = &self.surface_blend;
        check!(b.base_cells, "surface_blend.base_cells", 0.0, 5.0);
        check!(b.per_footprint, "surface_blend.per_footprint", 0.0, 5.0);
        let s = &self.soil;
        check!(s.depth_cells, "soil.depth_cells", 0, 50);
        check!(
            s.depth_variation_cells,
            "soil.depth_variation_cells",
            0.0,
            20.0
        );
        check!(
            s.depth_noise_freq_mult,
            "soil.depth_noise_freq_mult",
            0.1,
            2.0
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Asset loader
// ---------------------------------------------------------------------------

#[derive(Default, bevy::reflect::TypePath)]
pub struct GenerationParamsLoader;

impl AssetLoader for GenerationParamsLoader {
    type Asset = GenerationParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<GenerationParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: GenerationParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
