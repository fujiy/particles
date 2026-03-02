// assets/params/physics.ron の Asset 型定義とローダー。
//
// GPU MPM 物性・境界・連成パラメータを責務別セクションに分けて管理する。
// 実行中に hot reload で値を更新でき、検証失敗時は直前の有効値を維持する。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// サブ構造体
// ---------------------------------------------------------------------------

/// 水の状態方程式パラメータ [Eq.7, physics.md]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterEosParams {
    /// 基準密度 ρ₀ [kg/m³]. 許容: 100.0 – 10000.0
    pub rho0: f32,
    /// 音速 c₀ [m/s]. bulk_modulus = ρ₀·c₀² で計算される. 許容: 1.0 – 1000.0
    pub sound_speed_mps: f32,
}

/// 変形勾配クランプパラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeformationClampParams {
    /// J = det(F) の下限クランプ. 許容: 0.01 – 1.0
    pub j_min: f32,
    /// J の上限クランプ. 許容: 1.0 – 10.0
    pub j_max: f32,
    /// C マトリクスのノルム上限クランプ. 許容: 1.0 – 1000.0
    pub c_max_norm: f32,
}

/// APIC ↔ PIC ブレンドパラメータ [Eq.32, physics.md]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApicBlendParams {
    /// 水の APIC ブレンド係数 α. 1.0 = 完全 APIC, 0.0 = 完全 PIC. 許容: 0.0 – 1.0
    pub water: f32,
    /// 粉体の APIC ブレンド係数 α. 許容: 0.0 – 1.0
    pub granular: f32,
}

/// 地形境界 Coulomb 摩擦パラメータ [Eq.29, physics.md]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryFrictionParams {
    /// 水の Coulomb 摩擦係数 μ_b. 許容: 0.0 – 2.0
    pub water: f32,
    /// 粉体の Coulomb 摩擦係数 μ_b. 許容: 0.0 – 2.0
    pub granular: f32,
}

/// Drucker-Prager 粉体パラメータ (soil または sand)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DruckerPragerParams {
    /// ヤング率 E [Pa]. 許容: 1.0 – 1e8
    pub youngs_modulus_pa: f32,
    /// ポアソン比 ν. 許容: 0.0 – 0.45
    pub poisson_ratio: f32,
    /// 内部摩擦角 φ [度]. 許容: 0.0 – 60.0
    pub friction_deg: f32,
    /// 凝着力 c [Pa]. 許容: 0.0 – 1e6
    pub cohesion_pa: f32,
    /// 硬化係数 h. 許容: 0.0 – 10.0
    pub hardening: f32,
}

/// 水-粉体連成パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingParams {
    /// 法線方向スティフネス. 許容: 0.0 – 10.0
    pub normal_stiffness: f32,
    /// 接線方向ドラッグ. 許容: 0.0 – 10.0
    pub tangent_drag: f32,
    /// 連成摩擦係数. 許容: 0.0 – 2.0
    pub friction: f32,
    /// 最大インパルス比. 許容: 0.0 – 1.0
    pub max_impulse_ratio: f32,
}

// ---------------------------------------------------------------------------
// トップレベル Asset 型
// ---------------------------------------------------------------------------

/// GPU MPM 物性パラメータ。`assets/params/physics.ron` から読み込まれる。
#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParams {
    /// 水の状態方程式 [Eq.7, physics.md]
    pub water: WaterEosParams,
    /// 変形勾配クランプ
    pub deformation: DeformationClampParams,
    /// APIC ↔ PIC ブレンド [Eq.32, physics.md]
    pub apic: ApicBlendParams,
    /// 地形境界 Coulomb 摩擦 [Eq.29, physics.md]
    pub boundary: BoundaryFrictionParams,
    /// Drucker-Prager 土壌パラメータ
    pub soil: DruckerPragerParams,
    /// Drucker-Prager 砂パラメータ
    pub sand: DruckerPragerParams,
    /// 粉体の引張クランプ [Pa]. 0.0 = 無効. 許容: 0.0 – 1e5
    pub granular_tensile_clamp: f32,
    /// 水-粉体連成パラメータ
    pub coupling: CouplingParams,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            water: WaterEosParams {
                rho0: 1_000.0,
                sound_speed_mps: 16.0,
            },
            deformation: DeformationClampParams {
                j_min: 0.6,
                j_max: 1.4,
                c_max_norm: 80.0,
            },
            apic: ApicBlendParams {
                water: 0.95,
                granular: 0.78,
            },
            boundary: BoundaryFrictionParams {
                water: 0.3,
                granular: 0.6,
            },
            soil: DruckerPragerParams {
                youngs_modulus_pa: 8.0e2,
                poisson_ratio: 0.28,
                friction_deg: 36.0,
                cohesion_pa: 0.0,
                hardening: 0.0,
            },
            sand: DruckerPragerParams {
                youngs_modulus_pa: 7.0e2,
                poisson_ratio: 0.25,
                friction_deg: 34.0,
                cohesion_pa: 0.0,
                hardening: 0.0,
            },
            granular_tensile_clamp: 0.0,
            coupling: CouplingParams {
                normal_stiffness: 0.55,
                tangent_drag: 0.30,
                friction: 0.45,
                max_impulse_ratio: 0.50,
            },
        }
    }
}

impl PhysicsParams {
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
        check!(self.water.rho0, "water.rho0", 100.0, 10_000.0);
        check!(self.water.sound_speed_mps, "water.sound_speed_mps", 1.0, 1_000.0);
        check!(self.deformation.j_min, "deformation.j_min", 0.01, 1.0);
        check!(self.deformation.j_max, "deformation.j_max", 1.0, 10.0);
        check!(self.deformation.c_max_norm, "deformation.c_max_norm", 1.0, 1_000.0);
        if self.deformation.j_min >= self.deformation.j_max {
            return Err(format!(
                "deformation.j_min ({}) >= j_max ({})",
                self.deformation.j_min, self.deformation.j_max
            ));
        }
        check!(self.apic.water, "apic.water", 0.0, 1.0);
        check!(self.apic.granular, "apic.granular", 0.0, 1.0);
        check!(self.boundary.water, "boundary.water", 0.0, 2.0);
        check!(self.boundary.granular, "boundary.granular", 0.0, 2.0);
        Self::validate_dp(&self.soil, "soil")?;
        Self::validate_dp(&self.sand, "sand")?;
        check!(self.granular_tensile_clamp, "granular_tensile_clamp", 0.0, 1e5);
        check!(self.coupling.normal_stiffness, "coupling.normal_stiffness", 0.0, 10.0);
        check!(self.coupling.tangent_drag, "coupling.tangent_drag", 0.0, 10.0);
        check!(self.coupling.friction, "coupling.friction", 0.0, 2.0);
        check!(self.coupling.max_impulse_ratio, "coupling.max_impulse_ratio", 0.0, 1.0);
        Ok(())
    }

    fn validate_dp(dp: &DruckerPragerParams, prefix: &str) -> Result<(), String> {
        macro_rules! check {
            ($val:expr, $name:expr, $lo:expr, $hi:expr) => {
                if !($val >= $lo && $val <= $hi) {
                    return Err(format!(
                        "{}.{}: {} は [{}, {}] の範囲外",
                        prefix, $name, $val, $lo, $hi
                    ));
                }
            };
        }
        check!(dp.youngs_modulus_pa, "youngs_modulus_pa", 1.0, 1e8);
        check!(dp.poisson_ratio, "poisson_ratio", 0.0, 0.45);
        check!(dp.friction_deg, "friction_deg", 0.0, 60.0);
        check!(dp.cohesion_pa, "cohesion_pa", 0.0, 1e6);
        check!(dp.hardening, "hardening", 0.0, 10.0);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Asset loader
// ---------------------------------------------------------------------------

#[derive(Default, bevy::reflect::TypePath)]
pub struct PhysicsParamsLoader;

impl AssetLoader for PhysicsParamsLoader {
    type Asset = PhysicsParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<PhysicsParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: PhysicsParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
