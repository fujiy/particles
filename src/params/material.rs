// assets/params/material.ron の Asset 型定義とローダー。
//
// 材料種ごとの摩擦・反発・粘性・破壊閾値を管理する。
// 粒子質量・半径等 const fn 計算に依存する値は対象外（Rust 側 defaults.rs で管理）。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// サブ構造体
// ---------------------------------------------------------------------------

/// 液体相の接触・摩擦パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidContactParams {
    /// 地形反発スティフネス. 許容: 0.0 – 5.0
    pub terrain_repulsion_stiffness: f32,
    /// オブジェクト反発スティフネス. 許容: 0.0 – 5.0
    pub object_repulsion_stiffness: f32,
    /// 静止摩擦係数. 許容: 0.0 – 1.0
    pub friction_static: f32,
    /// 動摩擦係数. 許容: 0.0 – 1.0
    pub friction_dynamic: f32,
    /// XSPH 粘性係数. 許容: 0.0 – 1.0
    pub xsph_viscosity: f32,
}

/// 固体/粒状相の接触・摩擦パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolidContactParams {
    /// 地形反発スティフネス. 許容: 0.0 – 5.0
    pub terrain_repulsion_stiffness: f32,
    /// オブジェクト反発スティフネス. 許容: 0.0 – 5.0
    pub object_repulsion_stiffness: f32,
    /// 静止摩擦係数. 許容: 0.0 – 50.0
    pub friction_static: f32,
    /// 動摩擦係数. 許容: 0.0 – 50.0
    pub friction_dynamic: f32,
}

/// 固体破壊閾値パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakParams {
    /// 衝突インパルス破壊閾値 [N·s]. 許容: 0.0 – 1e8
    pub collision_impulse_threshold: f32,
    /// ひずみ破壊閾値 [-]. 許容: 0.0 – 10.0
    pub strain_threshold: f32,
}

/// 材料種の調整可能パラメータセット
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialKindParams {
    /// 液体相接触パラメータ（水のみ）
    pub liquid: Option<LiquidContactParams>,
    /// 固体相接触パラメータ
    pub solid: Option<SolidContactParams>,
    /// 粒状相接触パラメータ
    pub granular: Option<SolidContactParams>,
    /// 固体破壊閾値（液体相を持たない材料のみ）
    pub solid_break: Option<BreakParams>,
}

// ---------------------------------------------------------------------------
// トップレベル Asset 型
// ---------------------------------------------------------------------------

/// 材料パラメータ。`assets/params/material.ron` から読み込まれる。
#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct MaterialAssetParams {
    /// 水
    pub water: MaterialKindParams,
    /// 石
    pub stone: MaterialKindParams,
    /// 土壌
    pub soil: MaterialKindParams,
    /// 砂
    pub sand: MaterialKindParams,
}

impl Default for MaterialAssetParams {
    fn default() -> Self {
        Self {
            water: MaterialKindParams {
                liquid: Some(LiquidContactParams {
                    terrain_repulsion_stiffness: 1.0,
                    object_repulsion_stiffness: 0.70,
                    friction_static: 0.01,
                    friction_dynamic: 0.01,
                    xsph_viscosity: 0.01,
                }),
                solid: None,
                granular: None,
                solid_break: None,
            },
            stone: MaterialKindParams {
                liquid: None,
                solid: Some(SolidContactParams {
                    terrain_repulsion_stiffness: 0.85,
                    object_repulsion_stiffness: 0.35,
                    friction_static: 3.0,
                    friction_dynamic: 2.0,
                }),
                granular: Some(SolidContactParams {
                    terrain_repulsion_stiffness: 0.28,
                    object_repulsion_stiffness: 0.22,
                    friction_static: 10.0,
                    friction_dynamic: 8.0,
                }),
                solid_break: Some(BreakParams {
                    collision_impulse_threshold: 12000.0,
                    strain_threshold: 1.6,
                }),
            },
            soil: MaterialKindParams {
                liquid: None,
                solid: Some(SolidContactParams {
                    terrain_repulsion_stiffness: 0.33,
                    object_repulsion_stiffness: 0.26,
                    friction_static: 10.0,
                    friction_dynamic: 8.0,
                }),
                granular: Some(SolidContactParams {
                    terrain_repulsion_stiffness: 0.26,
                    object_repulsion_stiffness: 0.20,
                    friction_static: 20.0,
                    friction_dynamic: 16.0,
                }),
                solid_break: Some(BreakParams {
                    collision_impulse_threshold: 50.0,
                    strain_threshold: 0.10,
                }),
            },
            sand: MaterialKindParams {
                liquid: None,
                solid: Some(SolidContactParams {
                    terrain_repulsion_stiffness: 0.31,
                    object_repulsion_stiffness: 0.24,
                    friction_static: 10.0,
                    friction_dynamic: 8.0,
                }),
                granular: Some(SolidContactParams {
                    terrain_repulsion_stiffness: 0.24,
                    object_repulsion_stiffness: 0.18,
                    friction_static: 10.0,
                    friction_dynamic: 8.0,
                }),
                solid_break: Some(BreakParams {
                    collision_impulse_threshold: 10.0,
                    strain_threshold: 0.05,
                }),
            },
        }
    }
}

impl MaterialAssetParams {
    pub fn validate(&self) -> Result<(), String> {
        Self::validate_kind(&self.water, "water")?;
        Self::validate_kind(&self.stone, "stone")?;
        Self::validate_kind(&self.soil, "soil")?;
        Self::validate_kind(&self.sand, "sand")?;
        Ok(())
    }

    fn validate_kind(k: &MaterialKindParams, name: &str) -> Result<(), String> {
        macro_rules! check {
            ($val:expr, $field:expr, $lo:expr, $hi:expr) => {
                if !($val >= $lo && $val <= $hi) {
                    return Err(format!(
                        "{}.{}: {} は [{}, {}] の範囲外",
                        name, $field, $val, $lo, $hi
                    ));
                }
            };
        }
        if let Some(liq) = &k.liquid {
            check!(liq.terrain_repulsion_stiffness, "liquid.terrain_repulsion_stiffness", 0.0, 5.0);
            check!(liq.object_repulsion_stiffness, "liquid.object_repulsion_stiffness", 0.0, 5.0);
            check!(liq.friction_static, "liquid.friction_static", 0.0, 1.0);
            check!(liq.friction_dynamic, "liquid.friction_dynamic", 0.0, 1.0);
            check!(liq.xsph_viscosity, "liquid.xsph_viscosity", 0.0, 1.0);
        }
        for (phase_params, prefix) in [k.solid.as_ref().map(|p| (p, "solid")), k.granular.as_ref().map(|p| (p, "granular"))].into_iter().flatten() {
            check!(phase_params.terrain_repulsion_stiffness, &format!("{prefix}.terrain_repulsion_stiffness"), 0.0, 5.0);
            check!(phase_params.object_repulsion_stiffness, &format!("{prefix}.object_repulsion_stiffness"), 0.0, 5.0);
            check!(phase_params.friction_static, &format!("{prefix}.friction_static"), 0.0, 50.0);
            check!(phase_params.friction_dynamic, &format!("{prefix}.friction_dynamic"), 0.0, 50.0);
        }
        if let Some(brk) = &k.solid_break {
            check!(brk.collision_impulse_threshold, "solid_break.collision_impulse_threshold", 0.0, 1e8);
            check!(brk.strain_threshold, "solid_break.strain_threshold", 0.0, 10.0);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Asset loader
// ---------------------------------------------------------------------------

#[derive(Default, bevy::reflect::TypePath)]
pub struct MaterialAssetParamsLoader;

impl AssetLoader for MaterialAssetParamsLoader {
    type Asset = MaterialAssetParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<MaterialAssetParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: MaterialAssetParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
