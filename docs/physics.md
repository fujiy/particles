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

## 6. 現在の Block/LoD/Ghost 実装（MPM-WATER-07）

本章は、可変空間解像度（空間LoD）を持つ block 構成で、粒子と格子ノードの対応を
どのように管理し、境界を跨ぐ寄与を ghost としてどう扱うかを、実装準拠で説明する。

### 6.1 目的

空間LoDを導入すると、同一 world 位置に対して block ごとに異なる `h_b` を使うため、
次の2点が同時に課題になる。

1. 同一物理位置を、粗密 block 間で一意に同定できること
2. owner が異なる block に対しても、境界近傍の粒子寄与（P2G）を欠落させないこと

本実装の目的は、粒子の更新主体を owner block に固定しつつ、ghost 参照を用いて
境界の連続性と決定論性を維持することである。

### 6.2 用語定義

- `Block`: `GridBlock`。`(level, h_b, dt_b, origin_node, node_dims)` を持つ局所格子領域。
- `Level`: 空間解像度指数。`h_b = CELL_SIZE_M * 2^level`。
- `Cell dims` / `Node dims`: block は `N x N cells` を持ち、ノード数は `N+1 x N+1`。
- `World key`: level に依存しないノード同定キー。`world_key = node_coord * 2^level`。
- `Owner particle`: その粒子を G2P で更新する block を持つ粒子。
- `Ghost particle`: ある block の P2G に寄与するが、その block の owner ではない粒子。
- `Active block`: `owner_indices` または `ghost_indices` が非空の block。
- `Due block`: 現在 tick で G2P まで進める owner block。
- `Owner drift time`: subcycling の時刻ずれで生じる予測時間（`owner_block_drift_secs`）。

### 6.3 大まかなデータ構造

空間LoDと ghost 管理は主に次の3構造で成立する。

1. `GridHierarchy`
   - `blocks: Vec<GridBlock>`
   - `node_lookup: HashMap<world_key, GridNodeLocation>`
   - 共有境界ノードの owner 解決を担う。

2. `GridBlock`
   - ノード配列 `nodes` と owner フラグ `owned_nodes` を保持
   - `is_world_key_owned` により「この block が書き込めるノードか」を判定

3. `MpmBlockIndexTable`
   - `owner_indices[b]`: block `b` が更新主体となる粒子
   - `ghost_indices[b]`: block `b` の P2G に寄与する他owner粒子
   - `block_neighbors[b]`: レイアウト変化時に再構築される近傍候補

不変条件:
- 全粒子は owner を1つだけ持つ。
- P2G/pressure のノード書き込みは owner ノード（`owned_nodes=true`）のみに限定する。

### 6.4 大まかなアルゴリズム

フレーム内 subcycling の各 tick で、以下の順序を繰り返す。

1. `refresh_block_index_table` で owner を更新し、必要時に owner/ghost を再構築
2. active block 集合を更新
3. tick 到達済みの due block を決定
4. 全 block に対して `refresh_ghost_indices_for_block` を実行
5. ghost 反映後の active 集合を `grid_blocks_for_step` として再取得
6. `step_block_set_coupled(grid_blocks_for_step, due_blocks, ...)` を実行
   - `grid_blocks_for_step`: P2G, pressure, boundary, grid update 対象
   - `due_blocks`: G2P 対象（owner 粒子更新）

この分離により、「境界に近いが owner 変更はまだ起きない粒子」の寄与欠落を防ぐ。

### 6.5 詳細アルゴリズム

#### 6.5.1 block 生成と共有境界の owner 決定

- level map 指定時は `reset_mpm_grid_hierarchy_with_level_map` が適用される。
- `node_dims = cell_dims + 1` として境界ノード共有を許可する。
- `GridHierarchy::rebuild_node_lookup` で world key を構築し、共有ノード owner を決定する。

owner 優先規則:
1. より細かい level（小さい `level`）
2. 同 level は `origin_node.x` が小さい方
3. さらに同値なら `origin_node.y` が小さい方

#### 6.5.2 `refresh_block_index_table`（owner 再bin）

粒子ごとに予測位置
$$
\mathbf{x}_{pred}=\mathbf{x}+\mathbf{v}\Delta t_{drift}+\frac{1}{2}\mathbf{g}\Delta t_{drift}^{2}
$$
を計算し、`block_index_for_position` で owner 候補を得る。

さらに、候補 block で stencil を張ったときの欠損ノード数
（`missing_stencil_nodes_for_block`）を評価し、近傍候補から欠損最小の block を採用する。
これにより、粗密境界直前での不必要な失速を緩和する。

owner 変更が無い場合は early return し、owner/ghost 全再構築は行わない。
この最適化は計算量を下げる一方、境界近傍 ghost が古くなるため、次節の局所更新を必須とする。

#### 6.5.3 `refresh_ghost_indices_for_block`（局所 ghost 更新）

target block `b` について:

1. target block の world AABB を計算
2. 他 block AABB と `margin = 2.5 * max(h_target, h_other)` で近傍候補を抽出
3. 候補 block の owner 粒子のみを列挙
4. 粒子ごとに owner 側 `h_owner` で stencil を評価
5. stencil node の world AABB と target AABB が交差すれば ghost とみなす
6. `sort+dedup` 後に `ghost_indices[b]` を置換

この更新は tick ごとに全 block に対して行い、owner 不変時の取りこぼしを解消する。

#### 6.5.4 P2G/pressure/G2P における owner と ghost の責務分離

- P2G mass/momentum:
  - owner + ghost を処理
  - ghost も owner `h_b` で stencil を評価
  - 書き込み先は target block の owner ノードに限定
- P2G pressure:
  - 粒子 owner の `h_b` で密度推定・圧力力転送を行う
  - 読み取りは全 grid、書き込みはローカル owner ノードのみ
- G2P:
  - `owner_indices` のみ更新（ghost は更新しない）

従って、ghost は「隣接 block の P2G 入力補完」のための参照集合であり、
状態更新の主語は常に owner 粒子である。

#### 6.5.5 時間LoD（rate）との関係

- `dt_b` は frame を2冪分割した `base_dt_unit` の整数倍に量子化する。
- 空間LoD検証モードでは rate 計算に `h_rate=CELL_SIZE_M` を使用し、
  粗 block が過大な `dt_b` を取り続けることで生じる見かけ減速を抑制している。

## 7. 現行アルゴリズム（水優先）

1. `active_blocks_from_index_table` で active 候補を決める。  
2. 各 block の rate level と `dt_b` を量子化する。  
3. scheduler tick ごとに `refresh_block_index_table` を実行する。  
4. 全 block に対して `refresh_ghost_indices_for_block` を実行する。  
5. `grid_blocks_for_step` に対して `P2G -> Pressure -> Boundary -> Grid Update` を実行する。  
6. `due_blocks` の owner 粒子のみ G2P を実行する。  
7. block ごとの CFL比・境界侵入指標・質量集計を更新する。  

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

## 9. 実装上の注意（block/LoD/ghost）

- カーネル重みと勾配評価は P2G/G2P で同一実装を使う。
- `world_key` を経由しない node 参照を混在させると level 間ズレの原因になる。
- 共有境界ノードへの書き込みは owner block のみ許可する。
- `refresh_block_index_table` の early return を使う場合、
  `refresh_ghost_indices_for_block` の実行順序を必ず維持する。
- 並列化時は block index 重複排除済みであることを前提に raw pointer path を使う。
- 乱数や非決定順序和を避け、headlessテストの再現性を優先する。
