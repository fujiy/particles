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
    /// 対称ドラッグ率 γ [1/s]. 許容: 0.0 – 100.0
    pub drag_gamma: f32,
    /// 連成摩擦係数. 許容: 0.0 – 2.0
    pub friction: f32,
    /// 界面法線計算を有効化する最小 |∇phi_g|. 許容: 0.0 – 1000.0
    pub interface_min_grad: f32,
    /// 界面法線正規化の分母ε. 許容: 1e-9 – 1.0
    pub interface_normal_eps: f32,
}

/// 実行制御・検証向けのランタイム調整パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeTuningParams {
    /// 境界速度補正で使う SDF 閾値（h倍率）. 許容: 0.0 – 4.0
    pub boundary_velocity_sdf_threshold_h: f32,
    /// 境界の法線内向き速度をどれだけ除去するか. 1.0=完全投影, 0.0=無効. 許容: 0.0 – 1.0
    pub boundary_velocity_normal_projection_scale: f32,
    /// 統計の侵入判定ε [m]. 許容: 1e-6 – 0.1
    pub stats_penetration_epsilon_m: f32,
    /// 1フレームの最大catch-up substeps. 許容: 1 – 64
    pub max_substeps_per_frame: u32,
}

// ---------------------------------------------------------------------------
// トップレベル Asset 型
// ---------------------------------------------------------------------------

/// GPU MPM 物性パラメータ。`assets/params/physics.ron` から読み込まれる。
#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParams {
    /// 固定サブステップ dt [s]. 許容: 1e-5 – 1.0
    pub fixed_dt: f32,
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
    /// 実行制御・検証向けランタイム調整
    pub runtime: RuntimeTuningParams,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            fixed_dt: 1.0 / 240.0,
            water: WaterEosParams {
                rho0: 1_000.0,
                sound_speed_mps: 16.0,
            },
            deformation: DeformationClampParams {
                j_min: 0.88,
                j_max: 1.4,
                c_max_norm: 80.0,
            },
            apic: ApicBlendParams {
                water: 0.95,
                granular: 0.78,
            },
            boundary: BoundaryFrictionParams {
                water: 0.3,
                granular: 1.6,
            },
            soil: DruckerPragerParams {
                youngs_modulus_pa: 2.0e4,
                poisson_ratio: 0.28,
                friction_deg: 52.0,
                cohesion_pa: 400.0,
                hardening: 1.0,
            },
            sand: DruckerPragerParams {
                youngs_modulus_pa: 1.8e4,
                poisson_ratio: 0.25,
                friction_deg: 48.0,
                cohesion_pa: 120.0,
                hardening: 0.5,
            },
            granular_tensile_clamp: 0.0,
            coupling: CouplingParams {
                drag_gamma: 1.0,
                friction: 0.45,
                interface_min_grad: 0.02,
                interface_normal_eps: 1.0e-6,
            },
            runtime: RuntimeTuningParams {
                boundary_velocity_sdf_threshold_h: 0.5,
                boundary_velocity_normal_projection_scale: 1.0,
                stats_penetration_epsilon_m: 1.0e-3,
                max_substeps_per_frame: 8,
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
        check!(self.fixed_dt, "fixed_dt", 1e-5, 1.0);
        check!(self.water.rho0, "water.rho0", 100.0, 10_000.0);
        check!(
            self.water.sound_speed_mps,
            "water.sound_speed_mps",
            1.0,
            1_000.0
        );
        check!(self.deformation.j_min, "deformation.j_min", 0.01, 1.0);
        check!(self.deformation.j_max, "deformation.j_max", 1.0, 10.0);
        check!(
            self.deformation.c_max_norm,
            "deformation.c_max_norm",
            1.0,
            1_000.0
        );
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
        check!(
            self.granular_tensile_clamp,
            "granular_tensile_clamp",
            0.0,
            1e5
        );
        check!(self.coupling.drag_gamma, "coupling.drag_gamma", 0.0, 100.0);
        check!(self.coupling.friction, "coupling.friction", 0.0, 2.0);
        check!(
            self.coupling.interface_min_grad,
            "coupling.interface_min_grad",
            0.0,
            1000.0
        );
        check!(
            self.coupling.interface_normal_eps,
            "coupling.interface_normal_eps",
            1e-9,
            1.0
        );
        check!(
            self.runtime.boundary_velocity_sdf_threshold_h,
            "runtime.boundary_velocity_sdf_threshold_h",
            0.0,
            4.0
        );
        check!(
            self.runtime.boundary_velocity_normal_projection_scale,
            "runtime.boundary_velocity_normal_projection_scale",
            0.0,
            1.0
        );
        check!(
            self.runtime.stats_penetration_epsilon_m,
            "runtime.stats_penetration_epsilon_m",
            1e-6,
            0.1
        );
        if self.runtime.max_substeps_per_frame < 1 || self.runtime.max_substeps_per_frame > 64 {
            return Err(format!(
                "runtime.max_substeps_per_frame: {} は [1, 64] の範囲外",
                self.runtime.max_substeps_per_frame
            ));
        }
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
