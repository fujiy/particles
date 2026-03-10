# Physics Report: Water + Granular MLS‑MPM（単一グリッド・GPU前提）

> **改訂版**：境界摩擦を「減衰」から **クーロン摩擦（stick/slip）**へ変更／流体の負圧（張力）を **充填率で制御**／APIC↔PICブレンド係数の導入／粉体のDrucker‑Pragerを **SVD+log‑strain射影**（GPU向き）へ変更／水と粉体は **材料別速度場**（同一グリッド上で相ごとに速度を持つ）で連成。

---

## 1. 目的とスコープ

本書は、水と粉体（土/砂）を対象とした **明示** MLS‑MPM ソルバの実装仕様を定義する。  
ターゲットは **Terraria のようなセル単位の2D建築ゲーム**であり、グリッド幅 $h$ はセル幅より小さめ（例：半分程度）を想定する。狙いは **マクロにそれっぽい挙動**であり、現実の粒子間隙を水が流れるような液状化・飽和土等は対象外とする。

**対象外（明示的に導入しない）**
- 可変解像度（LoD）グリッド
- block境界でのフラックス交換
- ghost粒子ベースの境界連成
- 多孔質二相連成（間隙水圧・有効応力・飽和度進化）［Tampubolon2017］

---

## 1.1 記号と定義

空間次元を $d = 2$ とする．

**時間**
- $n$：タイムステップ
- $\Delta t$：時間刻み幅（s）
- $\mathbf{g}$：重力加速度（m/s$^2$）

**グリッド**
- $h$：グリッド間隔（m）
- $i$：グリッドノード index
- $\mathbf{x}_i\in\mathbb{R}^d$：ノード位置

**粒子**
- $p$：粒子 index
- $\mathbf{x}_p,\mathbf{v}_p$：粒子位置・速度
- $m_p$：粒子質量（一定）
- $V^0_p$：初期体積（一定）
- $\mathbf{F}_p$：変形勾配
- $J_p=\det(\mathbf{F}_p)$：体積比
- $\mathbf{C}_p$：APIC affine 行列（局所速度勾配の推定）

**材料（相）**
- $k\in\{\mathrm{w},\mathrm{g}\}$：$\mathrm{w}$=water, $\mathrm{g}$=granular
- $\rho^k_0$：基準密度（rest density）
- $K^k$：体積弾性率（bulk modulus；弱圧縮の硬さ）
- $\alpha^k_{\mathrm{apic}}\in[0,1]$：APIC↔PIC ブレンド係数（Sec. 4.5）
- $\mu^k_{\mathrm{b}}\ge 0$：地形境界の摩擦係数（Sec. 4.4）

---

## 2. 連続体モデル

### 2.1 支配方程式（共通）

水相・粉体相とも、保存則自体は同じで、違いは **構成則（応力 $\boldsymbol{\sigma}$ の決め方）**にある。

質量保存：
$$
\frac{D\rho}{Dt} + \rho\,\nabla\cdot\mathbf{u} = 0
\tag{1}
$$

運動量保存：
$$
\rho\frac{D\mathbf{u}}{Dt} = \nabla\cdot\boldsymbol{\sigma} + \rho\mathbf{g}
\tag{2}
$$

---

### 2.2 水の構成則（弱圧縮・張力制御つき）

#### 2.2.1 応力（等方圧のみ：基準は非粘性）

ゲーム用途（粗い解像度・安定性優先）では、粘性を物理的に厳密に入れるよりも、数値散逸のノブ（APIC↔PIC）で見た目を制御する方が扱いやすい。よってベースラインは **非粘性・等方圧**とし、粘性は将来的な拡張（格子拡散 or 応力粘性）として位置付ける。

Cauchy 応力：
$$
\boldsymbol{\sigma}^{\mathrm{w}} = -p^{\mathrm{w}}\mathbf{I}
\tag{3}
$$

MLS‑MPM の内部力に使うため、第1Piola応力に変換する：
$$
\mathbf{P}^{\mathrm{w}}
= J\,\boldsymbol{\sigma}^{\mathrm{w}}\mathbf{F}^{-T}
= -p^{\mathrm{w}}\,J\,\mathbf{F}^{-T}
\tag{4}
$$

#### 2.2.2 弱圧縮の状態方程式（EOS）

初期化で
$$
m_p = \rho^{\mathrm{w}}_0 V^0_p
\tag{5}
$$
を満たすとき、密度は
$$
\rho^{\mathrm{w}}_p = \frac{m_p}{V^0_p J_p} = \frac{\rho^{\mathrm{w}}_0}{J_p}
\tag{6}
$$
となる。ここから線形 EOS を用いて
$$
p^{\mathrm{w}}_p
= K^{\mathrm{w}}\left(\frac{\rho^{\mathrm{w}}_p}{\rho^{\mathrm{w}}_0}-1\right)
= K^{\mathrm{w}}\left(\frac{1}{J_p}-1\right)
\tag{7}
$$
とする。

> 注：これは “$p=K(J-1)$” のような弾性固体の罰則型とは別で、密度（体積比）に対する弱圧縮 EOS である。流体をNeo‑Hookeanなどの弾性モデルで近似すると「ゴムっぽく」なりやすいことが知られており、本仕様はそれを避ける意図で EOS 形を採用する［NiallTL］。

#### 2.2.3 負圧（張力）の扱い：充填率に応じて滑らかに抑制

Eq. (7) をそのまま使うと、自由表面などで $\rho<\rho_0$ が起きたとき **負圧**が生じ、解像度不足由来の誤差が「人工的な凝集（cohesion）」として見えてしまうことがある。一方で単純に $p\leftarrow\max(p,0)$ とすると、拡張側の復元力が完全に消え、体積（密度）誤差が戻らず薄くなり続ける場合がある。

そこで本仕様では、**局所充填率（volume fraction）**に応じて負圧だけを滑らかに弱める。

ノード $i$ の水充填率：
$$
\phi^{\mathrm{w}}_i = \frac{m^{\mathrm{w}}_i}{\rho^{\mathrm{w}}_0\,h^d}
\tag{8}
$$
粒子 $p$ への補間：
$$
\phi^{\mathrm{w}}_p = \sum_i w_{ip}\phi^{\mathrm{w}}_i
\tag{9}
$$

パラメータ $\phi_{\min}<\phi_{\max}$ を用い、減衰係数 $s(\phi)\in[0,1]$ を
$$
s(\phi)=\mathrm{smoothstep}\!\left(\phi_{\min},\phi_{\max},\phi\right)
\tag{10}
$$
とする。ここで $\mathrm{smoothstep}(a,b,x)$ は $x\le a$ で0、$x\ge b$ で1、間は3次の滑らかな補間。

負圧にのみ適用する：
$$
p^{\mathrm{w}}_p \leftarrow 
\begin{cases}
p^{\mathrm{w}}_p & (p^{\mathrm{w}}_p \ge 0),\\
s(\phi^{\mathrm{w}}_p)\,p^{\mathrm{w}}_p & (p^{\mathrm{w}}_p < 0).
\end{cases}
\tag{11}
$$

将来的に、より単純な tensionless（$p\leftarrow\max(p,0)$）や、下限付き（$p\ge -p_{\min}$）に置換する可能性がある。

#### 2.2.4 （任意）粒子体積の毎ステップ再評価

流体は局所変形が大きく、$\mathbf{F}$ の積分誤差が $J$ に蓄積して破綻しやすい。  
実装オプションとして、グリッドから密度（または近傍質量）を回収して体積 $V_p=m_p/\rho_p$ を毎ステップ再評価し、体積推定を“積分量”から“測定量”へ寄せる手法を用意してよい［NiallTL］。ただし追加のメモリアクセスが増えるため、既定では無効とする。

---

### 2.3 粉体の構成則（Drucker‑Prager：SVD + log‑strain 射影）

粉体（土/砂）は、弾性（Neo‑Hookean）＋Drucker‑Prager 型塑性を用い、**SVD と log‑strain 空間での射影**により GPU で安定に積分する［PhysSimDP］。Newton 反復ベースの return mapping は、GPU で収束事故・分岐が増えやすいので採用しない。

#### 2.3.1 弾性（圧縮可能 Neo‑Hookean）

ヤング率 $E$、ポアソン比 $\nu$ から Lamé 定数：
$$
\mu = \frac{E}{2(1+\nu)},\qquad
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}.
\tag{12}
$$

第1Piola 応力：
$$
\mathbf{P}^{\mathrm{g}}(\mathbf{F})
= \mu(\mathbf{F}-\mathbf{F}^{-T}) + \lambda\ln(J)\,\mathbf{F}^{-T},
\qquad J=\det(\mathbf{F}).
\tag{13}
$$

#### 2.3.2 log‑strain 空間での DP 射影（GPU向き）

試行変形 $\mathbf{F}_{\mathrm{tr}}$ を SVD：
$$
\mathbf{F}_{\mathrm{tr}}=\mathbf{U}\,\mathrm{diag}(\boldsymbol{\sigma})\,\mathbf{V}^T,\quad
\boldsymbol{\sigma}=(\sigma_1,\dots,\sigma_d),\ \sigma_i>0
\tag{14}
$$
とし、数値安定のため $\sigma_i\leftarrow\max(\sigma_i,\sigma_{\min})$ を入れる。

Hencky（log）ひずみベクトル：
$$
\boldsymbol{\epsilon}=\ln(\boldsymbol{\sigma})+\frac{v_{\mathrm{vol}}}{d}\mathbf{1},
\quad \mathbf{1}=(1,\dots,1)
\tag{15}
$$
（$v_{\mathrm{vol}}$ は Sec. 2.3.3 で定義する体積補正スカラー）。

偏差成分とノルム：
$$
\bar{\epsilon}=\frac{1}{d}\mathbf{1}^T\boldsymbol{\epsilon},\quad
\hat{\boldsymbol{\epsilon}}=\boldsymbol{\epsilon}-\bar{\epsilon}\mathbf{1},\quad
\|\hat{\boldsymbol{\epsilon}}\|=\sqrt{\hat{\boldsymbol{\epsilon}}^T\hat{\boldsymbol{\epsilon}}+\varepsilon}
\tag{16}
$$

摩擦角 $\varphi$ から DP 係数：
$$
\alpha_{\mathrm{dp}}=\sqrt{\frac{2}{3}}\frac{2\sin\varphi}{3-\sin\varphi}
\tag{17}
$$

射影（3ケース）［PhysSimDP］：
- **Case II（膨張 / 引張強度なし）**：$\mathbf{1}^T\boldsymbol{\epsilon}\ge 0$ なら $\boldsymbol{\epsilon}\leftarrow\mathbf{0}$
- それ以外では
$$
\Delta\gamma
= \|\hat{\boldsymbol{\epsilon}}\| + \frac{d\lambda+2\mu}{2\mu}\left(\mathbf{1}^T\boldsymbol{\epsilon}\right)\alpha_{\mathrm{dp}}
\tag{18}
$$
  - **Case I（弾性）**：$\Delta\gamma\le 0$ なら変更しない
  - **Case III（せん断降伏）**：それ以外なら
$$
\boldsymbol{\epsilon}\leftarrow \boldsymbol{\epsilon}-\frac{\Delta\gamma}{\|\hat{\boldsymbol{\epsilon}}\|}\hat{\boldsymbol{\epsilon}}
\tag{19}
$$

復元：
$$
\boldsymbol{\sigma}_{\mathrm{out}}=\exp(\boldsymbol{\epsilon}),\quad
\mathbf{F}_{\mathrm{corr}}=\mathbf{U}\,\mathrm{diag}(\boldsymbol{\sigma}_{\mathrm{out}})\,\mathbf{V}^T
\tag{20}
$$

#### 2.3.3 体積補正スカラー $v_{\mathrm{vol}}$

Case II を含む射影では、膨張が“新しい基準形状”として固定される副作用を起こし得るため、体積補正スカラー（`diff_log_J`）を追跡する［PhysSimDP］：
$$
v_{\mathrm{vol}}^{n+1}
= v_{\mathrm{vol}}^{n}-\ln\det(\mathbf{F}_{\mathrm{corr}})+\ln\det(\mathbf{F}_{\mathrm{tr}})
\tag{21}
$$

---

## 3. 離散状態（粒子・格子）

### 3.1 粒子状態

粒子 $p$ は以下を保持する。
- $\mathbf{x}_p,\mathbf{v}_p$
- $m_p,V^0_p$
- $\mathbf{F}_p$, $J_p=\det(\mathbf{F}_p)$
- $\mathbf{C}_p$
- material id $k(p)\in\{\mathrm{w},\mathrm{g}\}$

粉体粒子のみ：
- $v_{\mathrm{vol}}$（Eq. 21）

### 3.2 格子状態（単一グリッド・材料別速度場）

同一グリッド上で、ノードごとに材料別の質量・運動量・速度を持つ。  
ノード $i$、材料 $k$ について
- $m^k_i$：質量
- $\mathbf{p}^k_i$：運動量
- $\mathbf{v}^k_i=\mathbf{p}^k_i/m^k_i$（ただし $m^k_i>m_{\min}$）

これにより、水と粉体が同じセルにいても相対運動できる（単一速度場の“貼り付き”を避ける）。

---

## 4. MLS‑MPM 更新式

### 4.1 補間カーネル

粒子からノードへの重みを $w_{ip}=N(\mathbf{x}_i-\mathbf{x}_p)$、勾配を $\nabla w_{ip}$ とする。  
本仕様の既定は **quadratic B‑spline**（2Dで $3\times 3$ stencil）とする。

> 高次 B‑spline はセル跨ぎのノイズ低減に効くが stencil が広がり、セル表現の“ブロック感”を薄める可能性がある。まずは quadratic を維持し、精度劣化が気になった段階で再検討する［NiallTL］。

### 4.2 P2G（材料別：質量・運動量）

材料 $k$ ごとに

質量：
$$
m^k_i=\sum_{p:\,k(p)=k} w_{ip}m_p
\tag{22}
$$

運動量（APIC）：
$$
\mathbf{p}^k_i
=\sum_{p:\,k(p)=k} w_{ip}m_p\left(\mathbf{v}_p+\mathbf{C}_p(\mathbf{x}_i-\mathbf{x}_p)\right)
\tag{23}
$$

### 4.3 内部力（応力発散：材料別）

材料 $k$ ごとに
$$
\Delta\mathbf{p}^{k,\mathrm{int}}_i
=-\Delta t\sum_{p:\,k(p)=k} V^0_p\,\mathbf{P}_p\mathbf{F}_p^T\nabla w_{ip}
\tag{24}
$$

ここで $\mathbf{P}_p$ は
- 水：Eq. (4)（$p^{\mathrm{w}}$ は Eq. (7)→(11)）
- 粉体：Eq. (13) を $\mathbf{F}_p$（DP射影後）で評価

### 4.4 格子更新と地形境界（材料別）

速度：
$$
\mathbf{v}^{k,*}_i=\frac{\mathbf{p}^k_i+\Delta\mathbf{p}^{k,\mathrm{int}}_i}{m^k_i}
\quad (m^k_i>m_{\min})
\tag{25}
$$
重力：
$$
\mathbf{v}^{k,n+1}_i=\mathbf{v}^{k,*}_i+\Delta t\,\mathbf{g}
\tag{26}
$$

#### 4.4.1 静的地形の node 分類と boundary-aware transfer

静的地形がセル occupancy と SDF から与えられる場合、まず MPM node を
- **fluid node**: 地形外
- **solid node**: 地形内
の 2 値で分類する。現段階の地形はセル単位であり、静的壁境界の一次実装としては 2 値分類を既定とする。

粒子 $p$ の quadratic B-spline stencil を $\mathcal{N}(p)$ とし、そのうち fluid node のみを
$$
\mathcal{N}_{\mathrm{f}}(p)=\{\,i\in\mathcal{N}(p)\mid i\ \text{is fluid}\,\}
\tag{27}
$$
とする。transfer は solid node を含めず、fluid 側だけで重みを再正規化する：
$$
\tilde{w}_{ip}=
\frac{w_{ip}}{\sum_{j\in\mathcal{N}_{\mathrm{f}}(p)} w_{jp}}
\quad (i\in\mathcal{N}_{\mathrm{f}}(p))
\tag{28}
$$

P2G / G2P / APIC affine / 境界近傍の内部力評価は、この fluid-side support に対して整合的に行う。目的は、**壁内 node の速度補正や回復速度を粒子が stencil 経由で拾ってしまう経路を通常系から除去すること**である。

> 注：静的セル地形ではまず 2 値 node mask を採用する。将来、剛体や滑らかな境界へ拡張する際は、cut fraction / face fraction / SDF 値などの連続量へ一般化して、境界近傍の補間をより滑らかにする。

#### 4.4.2 境界 node の速度射影：非貫通 + クーロン摩擦

地形SDF を $\phi(\mathbf{x})$ とし、法線
$$
\mathbf{n}(\mathbf{x})=\frac{\nabla\phi(\mathbf{x})}{\|\nabla\phi(\mathbf{x})\|}
\tag{29}
$$
を用いる。fluid node のうち地形境界に隣接する node で、材料 $k$ ごとに以下を行う。

1) **非貫通（法線成分のみ）**
$$
v_n=\mathbf{v}^k_i\cdot\mathbf{n},\quad
\mathbf{v}^k_i\leftarrow \mathbf{v}^k_i-\min(v_n,0)\mathbf{n}
\tag{30}
$$

2) **接線：クーロン摩擦（stick/slip の近似）**  
接線速度 $\mathbf{v}_t=\mathbf{v}^k_i-(\mathbf{v}^k_i\cdot\mathbf{n})\mathbf{n}$、大きさ $v_t=\|\mathbf{v}_t\|$ とする。摩擦係数 $\mu^k_{\mathrm{b}}$ を用いて
$$
\mathbf{v}_t \leftarrow
\begin{cases}
\mathbf{0} & \left(v_t \le \mu^k_{\mathrm{b}}\,\max(\mathbf{v}^k_i\cdot\mathbf{n},0)\right),\\[4pt]
\left(1-\dfrac{\mu^k_{\mathrm{b}}\,\max(\mathbf{v}^k_i\cdot\mathbf{n},0)}{v_t}\right)\mathbf{v}_t & \text{otherwise}
\end{cases}
\tag{31}
$$
として $\mathbf{v}^k_i$ を更新する。

既定では、**静的地形に対して壁内 node へ一律の外向き回復速度を注入しない**。壁内 node 由来の反発は normal 系で過剰な早期反発や剪断を生みやすく、通常系の境界条件としては採用しない。

> これは速度レベルでの近似的な摩擦円錐射影である。単純な接線減衰よりも、$\mu$ を摩擦係数として解釈しやすく、stick/slip の質感が安定する［PhysSimFriction］。

### 4.5 G2P（材料別：速度・APIC・移流・$\mathbf{F}$）

粒子 $p$ の材料を $k=k(p)$ とする。

粒子速度：
$$
\mathbf{v}^{n+1}_p=\sum_i w_{ip}\mathbf{v}^{k,n+1}_i
\tag{32}
$$

APIC affine（raw）：
$$
\mathbf{C}^{\mathrm{raw}}_p
=\frac{4}{h^2}\sum_i w_{ip}\mathbf{v}^{k,n+1}_i(\mathbf{x}_i-\mathbf{x}_p)^T
\tag{33}
$$

APIC↔PIC ブレンド（数値散逸ノブ）：
$$
\mathbf{C}^{n+1}_p=\alpha^k_{\mathrm{apic}}\,\mathbf{C}^{\mathrm{raw}}_p,
\quad \alpha^k_{\mathrm{apic}}\in[0,1]
\tag{34}
$$

移流：
$$
\mathbf{x}^{n+1}_p=\mathbf{x}^n_p+\Delta t\,\mathbf{v}^{n+1}_p
\tag{35}
$$

変形勾配：
$$
\mathbf{F}^{n+1}_p=(\mathbf{I}+\Delta t\,\mathbf{C}^{n+1}_p)\mathbf{F}^n_p
\tag{36}
$$
$J_p=\det(\mathbf{F}_p)$ は $[J_{\min},J_{\max}]$ にクランプする。

#### 4.5.1 移流後の粒子フェイルセーフ

上記の boundary-aware transfer は通常系の接触を担う。これとは別に、数値誤差・大速度・薄障害物などで粒子が地形内へ侵入した場合のみ、移流後に粒子レベルのフェイルセーフを適用してよい。

粒子位置 $\mathbf{x}^{n+1}_p$ で SDF を評価し、$\phi(\mathbf{x}^{n+1}_p)<0$ なら、法線 $\mathbf{n}$ を用いて
$$
\mathbf{x}^{n+1}_p \leftarrow \mathbf{x}^{n+1}_p - \phi(\mathbf{x}^{n+1}_p)\mathbf{n}
\tag{37}
$$
で地形外へ射影する。あわせて速度は
$$
\mathbf{v}^{n+1}_p \leftarrow \mathbf{v}^{n+1}_p - \min(\mathbf{v}^{n+1}_p\cdot\mathbf{n},0)\mathbf{n}
\tag{38}
$$
として法線内向き成分を除去する。これは**通常系の壁反発を粒子押し返しで代用するものではなく、異常時のみの最後段フェイルセーフ**とする。

#### 4.5.2 水の等方化（せん断の蓄積を捨てる）

水が“弾性体っぽく”見える主要因は、$\mathbf{F}$ にせん断が蓄積することにある。水は等方圧のみを持てばよいので、更新後に
$$
\mathbf{F}^{n+1}_p \leftarrow J_p^{1/d}\mathbf{I}
\quad (k(p)=\mathrm{w})
\tag{39}
$$
として等方化する（体積比は保持）。この意図は、流体を弾性モデルで近似した際の“ゴム化”を避けることにある［NiallTL］。

> 将来、動的剛体との 2-way coupling へ拡張する際は、2 値 node mask だけでは接触が粗くなりやすい。境界近傍の滑らかさのために連続的な cut fraction / face fraction / SDF を導入し、さらに node 射影や粒子フェイルセーフで除去した法線運動量を剛体側へ対称更新して、全体系の運動量保存を明示的に担保する。

### 4.6 粉体の塑性更新（DP射影）

粉体粒子（$k(p)=\mathrm{g}$）では Eq. (36) のあとに
- $\mathbf{F}_{\mathrm{tr}}\leftarrow \mathbf{F}^{n+1}_p$
- Eqs. (14)–(20) で $\mathbf{F}_{\mathrm{corr}}$ を求める
- Eq. (21) で $v_{\mathrm{vol}}$ 更新
- $\mathbf{F}^{n+1}_p\leftarrow \mathbf{F}_{\mathrm{corr}}$
を行う。

---

## 5. 時間刻み制御（目安）

単一解像度のため $\Delta t$ はグローバル。安全側の目安として
$$
\Delta t \le C\frac{h}{u_{\max}+c}
\tag{40}
$$
を用いる（$u_{\max}$：最大速度、$c$：見かけ音速、$C\in(0,1)$：安全率）。必要に応じて一様 substep を入れる。

---

## 6. GPU 実行（概要）

- 粒子・格子バッファは GPU 常駐
- CPU 同期はパラメータ更新／スポーン・削除／最小限のメトリクスのみ

1ステップ（概略）：
1. active tile 構築
2. active grid クリア（材料別配列）
3. P2G（材料別）
4. 内部力（応力）計算→格子運動量更新
5. 格子更新＋地形境界（材料別）
6. **材料間連成（Sec. 7）**
7. G2P（材料別）

---

## 7. 水–粉体連成（ゲーム向け：保存則重視の近似）

同一ノードに水と粉体が存在するとき、材料間の運動量交換を **格子上**で近似する。採用する要素は：
- 法線方向：非貫通（相対速度の接近成分を除去）
- 接線方向：クーロン摩擦（簡易）
- 追加安定化：弱い対称ドラッグ（相対速度の減衰）

多孔質二相（液状化）に必要な項は導入しない［Tampubolon2017］。

### 7.1 対称ドラッグ（“若干の平均化”）

ノード $i$ で水・粉体の質量と速度を $(m^{\mathrm{w}}_i,\mathbf{v}^{\mathrm{w}}_i)$、$(m^{\mathrm{g}}_i,\mathbf{v}^{\mathrm{g}}_i)$ とする。  
縮約質量：
$$
m_{\mathrm{red}}=\frac{m^{\mathrm{w}}_i m^{\mathrm{g}}_i}{m^{\mathrm{w}}_i+m^{\mathrm{g}}_i}
\tag{41}
$$
相対速度：
$$
\Delta\mathbf{v}=\mathbf{v}^{\mathrm{w}}_i-\mathbf{v}^{\mathrm{g}}_i
\tag{42}
$$
ドラッグ率 $\gamma\ge 0$ を用い $\eta=\min(\gamma\Delta t,1)$ として
$$
\mathbf{J}_{\mathrm{drag}}=-\eta\,m_{\mathrm{red}}\Delta\mathbf{v}
\tag{43}
$$
を定義し、
$$
\mathbf{v}^{\mathrm{w}}_i\leftarrow \mathbf{v}^{\mathrm{w}}_i+\frac{\mathbf{J}_{\mathrm{drag}}}{m^{\mathrm{w}}_i},\quad
\mathbf{v}^{\mathrm{g}}_i\leftarrow \mathbf{v}^{\mathrm{g}}_i-\frac{\mathbf{J}_{\mathrm{drag}}}{m^{\mathrm{g}}_i}
\tag{44}
$$
で更新する。これは全運動量を保存しつつ相対速度を減らす。

### 7.2 材料間摩擦（接線 + 非貫通）

摩擦には界面法線が必要なため、簡易に $\phi^{\mathrm{g}}$（粉体充填率）の勾配から求める：
$$
\mathbf{n}^{\mathrm{wg}}_i=\frac{\nabla \phi^{\mathrm{g}}_i}{\|\nabla\phi^{\mathrm{g}}_i\|+\epsilon}
\tag{45}
$$
$\|\nabla\phi^{\mathrm{g}}\|$ が小さい場合は未解像としてスキップしてよい。

ドラッグ後（または前）の相対速度 $\Delta\mathbf{v}$ を
$$
\Delta v_n=\Delta\mathbf{v}\cdot\mathbf{n}^{\mathrm{wg}}_i,\quad
\Delta\mathbf{v}_t=\Delta\mathbf{v}-\Delta v_n\mathbf{n}^{\mathrm{wg}}_i
\tag{46}
$$
に分解し、$\Delta v_n<0$（接近）なら
$$
\mathbf{J}_n=-m_{\mathrm{red}}\Delta v_n\mathbf{n}^{\mathrm{wg}}_i
\tag{47}
$$
で接近成分を除去する。

接線摩擦係数 $\mu_{\mathrm{wg}}$ を用い、速度レベルで摩擦円錐に射影する：
$$
\mathbf{J}_t=
\begin{cases}
-m_{\mathrm{red}}\Delta\mathbf{v}_t
& \left(\|\Delta\mathbf{v}_t\|\le \mu_{\mathrm{wg}}\|\mathbf{J}_n\|/m_{\mathrm{red}}\right),\\[4pt]
-\mu_{\mathrm{wg}}\|\mathbf{J}_n\|\dfrac{\Delta\mathbf{v}_t}{\|\Delta\mathbf{v}_t\|}
& \text{otherwise}
\end{cases}
\tag{48}
$$
$\mathbf{J}=\mathbf{J}_n+\mathbf{J}_t$ を Eq. (44) と同様に対称適用する。

---

## 8. 受け入れ基準（最小）

- water‑only：クラッシュ/NaNなし、地形侵入率が閾値以内、実時間で安定継続
- granular‑only：クラッシュ/NaNなし、$v_{\mathrm{vol}}$ 監視で体積ドリフトが閾値以内
- mixed：材料間インパルスの収支（保存）が閾値以内、非液状化前提で安定継続

---

## 参考文献

- **[Hu2018]** Hu et al., “A Moving Least Squares Material Point Method with Displacement Discontinuity and Two‑Way Rigid Body Coupling”, ACM TOG (SIGGRAPH 2018). DOI: 10.1145/3197517.3201293.（Project/PDF: https://yzhu.io/publication/mpmmls2018siggraph/ ）
- **[APIC]** Jiang et al., “The Affine Particle‑In‑Cell Method”, ACM TOG (SIGGRAPH 2015). DOI: 10.1145/2766996.（PDF: https://www.math.ucdavis.edu/~jteran/papers/JSSTS15.pdf ）
- **[PhysSimFriction]** Physics‑Based Simulation（online book）, “Friction”.（https://phys-sim-book.github.io/lec2.4-friction.html ）
- **[PhysSimDP]** Physics‑Based Simulation（online book）, “Drucker‑Prager Elastoplasticity”.（https://phys-sim-book.github.io/lec30.1-drucker_prager.html ）
- **[NiallTL]** Niall T.L., “mpm guide”.（https://nialltl.neocities.org/articles/mpm_guide ）
- **[Sulsky1999]** Sulsky et al., “The material point method for the simulation of large deformation behavior of materials”, CMAME.（PDF mirror: https://math.unm.edu/~sulsky/papers/CMAME.pdf ）
- **[TaichiMPM]** yuanming‑hu/taichi_mpm（MLS‑MPM参照実装・mpm88例）。（https://github.com/yuanming-hu/taichi_mpm ）
- **[Tampubolon2017]** Tampubolon et al., “Multi‑species simulation of porous sand and water mixtures”.（PDF: https://www.math.ucdavis.edu/~jteran/papers/PGKFTJM17.pdf ）
