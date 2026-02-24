# Physics Report: Water MLS-MPM (Explicit)

## 1. 目的

本書は、水（単相）を対象とした明示 MLS-MPM ソルバの実装仕様を、連続体方程式から離散化、計算手順、安定条件まで一貫して定義する。

対象外:
- 剛体ソルバ本体
- 粉体系（granular）ソルバ本体
- 陰解法ベースの圧力投影

## 2. 連続体モデル

### 2.1 支配方程式

連続体の質量保存・運動量保存を以下で扱う。

- 質量保存:
$$
\frac{D\rho}{Dt} + \rho\,\nabla\cdot\mathbf{u} = 0
$$

- 運動量保存:
$$
\rho\frac{D\mathbf{u}}{Dt} = \nabla\cdot\boldsymbol{\sigma} + \rho\mathbf{g}
$$

ここで
- $\rho$: 密度
- $\mathbf{u}$: 速度
- $\boldsymbol{\sigma}$: Cauchy 応力
- $\mathbf{g}$: 重力加速度

### 2.2 水の構成則（弱圧縮）

水は等方圧力 + 粘性で近似する。

$$
\boldsymbol{\sigma} = -p\mathbf{I} + 2\mu\mathbf{D}
$$
$$
\mathbf{D} = \frac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^T)
$$

圧力は密度圧縮（EOS系）を標準とし、

$$
p = K\max\left(\frac{\rho}{\rho_0}-1,\,0\right)
$$

を基本形として使う（弱圧縮、引張側は0で打ち切る）。

体積比 $J = \det(\mathbf{F})$ を使う

$$
p = K(J-1)
$$

は簡易近似（比較・デバッグ用途）として扱う。

## 3. 離散状態（粒子・格子）

### 3.1 粒子状態

粒子 $p$ は以下を持つ。

- 位置 $\mathbf{x}_p$
- 速度 $\mathbf{v}_p$
- 質量 $m_p$
- 初期体積 $V^0_p$
- 変形勾配 $\mathbf{F}_p$
- APIC affine 行列 $\mathbf{C}_p$

### 3.2 格子状態

格子ノード $i$ は以下を持つ。

- 質量 $m_i$
- 運動量 $\mathbf{p}_i$
- 速度 $\mathbf{v}_i = \mathbf{p}_i/m_i$

## 4. MLS-MPM 更新式

### 4.1 補間カーネル

粒子からノードへの重みを $w_{ip} = N(\mathbf{x}_i-\mathbf{x}_p)$ とする。

- 一貫して同じ次数（例: quadratic B-spline）を P2G/G2P で使用する。
- 勾配は $\nabla w_{ip}$ として事前計算またはオンザフライ評価する。

### 4.2 P2G（質量・運動量転送）

ノード質量:
$$
m_i = \sum_p w_{ip} m_p
$$

APIC を含むノード運動量:
$$
\mathbf{p}_i = \sum_p w_{ip} m_p\left(\mathbf{v}_p + \mathbf{C}_p(\mathbf{x}_i-\mathbf{x}_p)\right)
$$

内部力寄与を明示時間積分で組み込む形は
$$
\Delta \mathbf{p}^{int}_i = -\Delta t\sum_p V^0_p\,\mathbf{P}_p\mathbf{F}_p^T\nabla w_{ip}
$$

ここで $\mathbf{P}_p$ は第一Piola応力。

本実装の標準（密度圧縮）では、まず格子質量から粒子密度を推定する。
$$
\rho_p \approx \sum_i w_{ip}\,m_i / h^d
$$
$$
p_p = K\max\left(\frac{\rho_p}{\rho_0}-1,\,0\right)
$$

その上で等方圧力応力を使う。
$$
\boldsymbol{\sigma}_p \approx -p_p\mathbf{I}
$$

Jベース圧力を使う簡易近似では
$$
\mathbf{P}_p \approx -p_p\,J_p\,\mathbf{F}_p^{-T}
$$
とできる。

### 4.3 格子更新

ノード速度更新:
$$
\mathbf{v}_i^* = \frac{\mathbf{p}_i + \Delta \mathbf{p}^{int}_i}{m_i}
$$
$$
\mathbf{v}_i^{n+1} = \mathbf{v}_i^* + \Delta t\,\mathbf{g}
$$

`m_i` が閾値未満のノードは無効としてスキップする。

### 4.4 境界条件（地形SDF）

地形SDFを $\phi(\mathbf{x})$、法線を
$$
\mathbf{n} = \frac{\nabla\phi}{\|\nabla\phi\|}
$$
とする。

$\phi(\mathbf{x}_i)<0$ のノードで

1. 法線成分を非貫通化
$$
\mathbf{v}_i \leftarrow \mathbf{v}_i - \min(\mathbf{v}_i\cdot\mathbf{n},0)\mathbf{n}
$$

2. 接線減衰（摩擦係数 $\mu_b$）
$$
\mathbf{v}_{t} = \mathbf{v}_i - (\mathbf{v}_i\cdot\mathbf{n})\mathbf{n}
$$
$$
\mathbf{v}_{t} \leftarrow \max(0,1-\mu_b)\,\mathbf{v}_{t}
$$

### 4.5 G2P（速度・位置・内部状態更新）

粒子速度:
$$
\mathbf{v}_p^{n+1} = \sum_i w_{ip}\mathbf{v}_i^{n+1}
$$

APIC affine:
$$
\mathbf{C}_p^{n+1} = \frac{4}{h^2}\sum_i w_{ip}\,\mathbf{v}_i^{n+1}(\mathbf{x}_i-\mathbf{x}_p)^T
$$

粒子位置:
$$
\mathbf{x}_p^{n+1} = \mathbf{x}_p^n + \Delta t\,\mathbf{v}_p^{n+1}
$$

変形勾配:
$$
\mathbf{F}_p^{n+1} = (\mathbf{I}+\Delta t\,\mathbf{C}_p^{n+1})\mathbf{F}_p^n
$$

安定化のため $J_p = \det(\mathbf{F}_p)$ に上下限クランプを設ける。

実装上は、数値ドリフト抑制のために
$$
\mathbf{F}_p \leftarrow (1-\alpha_F)\mathbf{F}_p + \alpha_F\mathbf{I}
$$
の緩和（`f_relaxation = \alpha_F`）を任意で入れてよい。これは物理モデルではなく数値安定化項である。

## 5. 時間刻み制御（CFL）

block $b$ の時間刻みは

$$
\Delta t_b = \min\left(
C_u\frac{h_b}{u_{\max,b}},
C_c\frac{h_b}{c_b+u_{\max,b}},
C_a\sqrt{\frac{h_b}{a_{\max,b}}}
\right)
$$

- $u_{\max,b}$: block内最大速度
- $c_b$: EOSから得る見かけ音速
- $a_{\max,b}$: block内最大加速度

実装では `dt_b <= dt_frame` を満たす整数 subcycling 回数に丸める。

## 6. LoD と subcycling

### 6.1 空間LoD

- 細格子 block: 境界や高せん断領域
- 粗格子 block: 低活動・遠方領域

### 6.2 時間LoD

- 各 block は独自 `dt_b` で更新する。
- フレーム内で更新回数が異なる block 同士は、境界フラックス交換を同期点で解決する。

### 6.3 境界交換ポリシー

- LoD 境界で交換する量:
  - 質量フラックス
  - 運動量フラックス
- 対称更新を保証し、片側のみ更新を禁止する。

## 7. 最小実装アルゴリズム（水優先）

1. `active blocks` を確定する。  
2. 各 block で `dt_b` を算出する。  
3. フレーム区間を subcycle タイムラインへ分割する。  
4. サイクルごとに対象 block のみ `P2G -> Grid -> Boundary -> G2P` を実行する。  
5. 同期点で LoD 境界フラックスを双方向に調停する。  
6. フレーム末で保存量誤差（質量・運動量）と CFL 余裕を記録する。  

## 8. 検証指標

- 質量誤差率:
$$
\epsilon_m = \frac{|M(t)-M(0)|}{M(0)}
$$

- 運動量誤差率:
$$
\epsilon_p = \frac{\|\mathbf{P}(t)-\mathbf{P}(0)-\int_0^t M\mathbf{g}\,dt\|}{\|\mathbf{P}(0)\|+\epsilon}
$$

- 最大CFL比:
$$
r_{CFL} = \max_b \frac{\Delta t_b}{\Delta t_{limit,b}}
$$

- 境界侵入率: `phi(x_p) < 0` 粒子比率

受け入れ基準は `docs/tasks.md` の Work Unit 完了条件で定義する。

## 9. 実装上の注意

- カーネル重みと勾配評価は P2G/G2P で同一実装を使う。
- 無効ノード判定閾値を固定し、フレームごとに変えない（決定論性維持）。
- 並列化時はノード加算をスレッドローカル集計 + reduce で実装する。
- 乱数や非決定順序和を避け、headlessテストの再現性を優先する。
