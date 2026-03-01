# Physics Report: Water MLS-MPM (Single Grid, GPU-First)

## 1. 目的

本書は、水（単相）を対象とした明示 MLS-MPM ソルバの実装仕様を、
単一解像度グリッドとGPU実行前提で定義する。

対象外:
- 可変空間LoD block
- block境界フラックス交換
- ghost粒子ベースの境界連成

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

比較・検証用途として体積比 $J = \det(\mathbf{F})$ ベースの

$$
p = K(J-1)
$$

も利用可能とする。

## 3. 離散状態（粒子・格子）

### 3.1 粒子状態

粒子 $p$ は以下を持つ。

- 位置 $\mathbf{x}_p$
- 速度 $\mathbf{v}_p$
- 質量 $m_p$
- 初期体積 $V^0_p$
- 変形勾配 $\mathbf{F}_p$
- APIC affine 行列 $\mathbf{C}_p$

### 3.2 格子状態（単一解像度）

格子ノード $i$ は以下を持つ。

- 質量 $m_i$
- 運動量 $\mathbf{p}_i$
- 速度 $\mathbf{v}_i = \mathbf{p}_i/m_i$

グリッド間隔は全領域で一定の `h` を使う。

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

$\phi(\mathbf{x}_i)<\phi_{th}$ のノードで

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

3. 深部侵入（$\phi < \phi_{th}$）では押し戻し速度を加算する。

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

## 5. 時間刻み制御（CFL）

単一解像度グリッドのため、刻み幅はグローバル `dt` とする。

$$
\Delta t \le C \frac{h}{u_{max}+c}
$$

- $u_{max}$: 粒子速度の最大値
- $c$: 見かけ音速
- $C$: safety factor

必要時のみ一様 substep（全粒子同一分割）を導入する。

## 6. GPU実行アーキテクチャ

### 6.1 データ常駐ポリシー

- 粒子バッファと格子バッファはGPU常駐とする。
- CPUとの同期は以下に限定する。
  - パラメータ更新
  - スポーン/削除イベント
  - 最小限メトリクス readback
- 毎ステップの粒子全量 readback は行わない。
- 連続体計算のCPU本番経路は持たない。

### 6.2 1ステップの標準パイプライン

1. `build_active_tiles`
2. `clear_active_grid`
3. `p2g_mass_momentum`
4. `p2g_pressure_or_density`
5. `grid_update_with_boundary`
6. `g2p_update_particles`
7. `compact_or_cull_particles`（必要時）

### 6.3 active tile 制御

- tile は固定サイズノード集合で表現する。
- 粒子から tile index を求め、`tile_count` をatomic加算する。
- `tile_count > 0` を compaction して `active_tile_list` を作る。
- `grid clear` / `grid update` は `active_tile_list` だけを対象に dispatch する。

### 6.4 P2G/G2Pとの関係

- P2G と G2P は粒子並列が主であり、tile化の主効果は grid系パスに出る。
- ただしP2Gで更新されるノード範囲は active tile 近傍に限定されるため、
  active tile 情報はP2Gのメモリアクセス局所化にも寄与する。

### 6.5 境界連成の実装方針

- 地形SDFサンプルもGPUで評価/参照する。
- 境界処理だけCPU実行し毎ステップ同期する方式は、同期待ちで不利なため採用しない。

### 6.6 既存CPUコードの扱い

- 既存CPU実装は、期待される挙動の確認に限定して参照してよい。
- GPU移行後に不要となったCPUコードは即時削除し、残骸を放置しない。

## 7. 連成拡張方針

- 段階1: 水GPU経路を成立（`water_drop` 再現）
- 段階2: 剛体連成（GPU側へ境界速度場入力/反力出力I/Fを固定）
- 段階3: 粉体連成（GPU基盤へ統合）

全段階で「交換量の保存則」と「決定論性」をテストで保証する。

## 8. 描画実装方針

- 第1段階はデバッグオーバーレイで粒子を円表示し、物理挙動を検証する。
- 第2段階で水のドット絵テクスチャ描画をGPUで実装する。
- ドット絵描画はフラグメントシェーダーを第一候補とし、必要に応じてcomputeで補助テクスチャを生成する。

## 9. 受け入れ基準（最小セット）

- `water_drop` シナリオで以下を満たすこと。
  - クラッシュ/NaNなし
  - 質量誤差が閾値以内
  - 地形侵入率が閾値以内
  - 実時間ステップで安定継続
- デバッグ円オーバーレイで粒子分布が連続的に観察できること。
- GPUドット絵描画で視覚破綻（ちらつき、境界欠落、同期遅延）がないこと。
- プロファイルでGPU pass別時間を計測可能であること。
