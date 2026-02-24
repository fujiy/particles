# Physics Notes (Granular / XPBD)

このドキュメントは、現行実装における**粉体（granular）XPBDソルバ**を、
XPBD未経験者でも追えるように基礎から説明するためのノートである。

対象は「粉体」だけであり、流体（PBD/XPBD）は扱わない。
また、ここでの式は理論一般ではなく、**現在のコード挙動**を説明することを優先する。

参照実装:
- `src/physics/solver/liquid.rs`
- `src/physics/solver/granular.rs`
- `src/physics/world/particle/mod.rs`

---

## 1. まず「XPBDで何をしているか」

### 1.1 PBDの直感

PBD（Position-Based Dynamics）は、
「速度や力を直接解く」よりも先に「位置が満たすべき条件（拘束）」を解き、
最後に位置差分から速度を作る手法である。

例えば接触なら、「粒子同士が重ならない」を拘束
$$
C(\mathbf{x}) \ge 0
$$
として定義し、反復的に位置補正して満たす。

長所は安定性、短所は剛性（硬さ）と時間刻みの依存が強いこと。

### 1.2 XPBDの追加要素

XPBD（Extended PBD）は、拘束にコンプライアンス（柔らかさ）を導入し、
時間刻みを変えても挙動が崩れにくいようにした拡張である。

拘束ごとにラグランジュ乗数 $\lambda$ を持ち、1反復の更新は典型的に
$$
\Delta \lambda =
\frac{-C(\mathbf{x}) - \alpha \lambda}
{\sum_k w_k \lVert \nabla_{\mathbf{x}_k} C \rVert^2 + \alpha}
$$
となる（$w_k=1/m_k$ は逆質量）。

ここで
$$
\alpha = \frac{c}{\Delta t^2}
$$
で、$c$ はコンプライアンス（$c=0$ に近いほど硬い）。

この $\Delta\lambda$ で位置補正を行う：
$$
\Delta \mathbf{x}_k = w_k \, \nabla_{\mathbf{x}_k} C \, \Delta\lambda
$$

---

## 2. この実装のデータと記号

以降で使う記号:

- 粒子 index: $i, j$
- 位置: $\mathbf{x}_i$
- 速度: $\mathbf{v}_i$
- 質量: $m_i$、逆質量: $w_i = 1/m_i$
- 法線: $\mathbf{n}$
- 接線単位ベクトル: $\mathbf{t}$
- 固定サブステップ: $\Delta t_{\text{sub}}$
- 粉体内部サブステップ: $\Delta t_g = \Delta t_{\text{sub}}/N_g$
- XPBD法線コンプライアンス項: $\alpha_n = c_n/\Delta t_g^2$
- XPBD接線コンプライアンス項: $\alpha_t = c_t/\Delta t_g^2$

この実装では、粒子ごとに「この substep で実際に進めた時間」$\Delta t_i$ を持つ。
これは `particle_execution_dt_substep[i]` として保存される。

---

## 3. マルチレートの前提

`Sub-block` ごとに更新レベルがあり、
$$
r = 2^{\text{level}}
$$
を divisor（更新間引き率）とする。

substep index を $s$ とすると、scheduled 条件は
$$
(s+1)\bmod r = 0
$$
である。

このとき有効時間幅は次のように扱う:

- 非マルチレート対象粒子: $\Delta t_i = \Delta t_{\text{sub}}$
- マルチレート対象粒子
  - scheduled: $\Delta t_i = r\,\Delta t_{\text{sub}}$
  - unscheduled: $\Delta t_i = 0$

重要なのは、後段の速度再構成もこの $\Delta t_i$ を使うこと。
つまり「予測で進めた時間」と「速度再構成の分母」が一致する。

---

## 4. 1 fixed substep における粉体の計算順

粉体に関係する大枠:

1. 予測段（`predict_positions`）
2. `Sub-block` スケジューラ更新
3. 粉体XPBD接触（`granular::solve_contacts`）
4. 速度再構成（`update_velocity`）
5. 反発（`granular::apply_restitution`）

この順序の意味は以下。

- 予測段で「拘束を無視した仮位置」を作る
- XPBDで拘束を満たすよう位置を戻す
- 最後に位置差分から速度を作る

---

## 5. 予測段（現行: 粉体は重力を直接積分しない）

現在の実装では、粉体については
「重力で速度を直接更新」しない。

予測段の粉体は
$$
\mathbf{v}_i^* = \mathbf{v}_i
$$
$$
\mathbf{x}_i^* = \mathbf{x}_i + \mathbf{v}_i \Delta t_i
$$
のみである。

unscheduled 粒子は $\Delta t_i=0$ なので、その substep では凍結される。

この設計意図は、外力（少なくとも重力）を接触拘束側に寄せるためである。

---

## 6. 粉体XPBD: 粒子-粒子接触

### 6.1 法線拘束（非貫通）

2粒子の半径を $r_i, r_j$ とし、
$$
C_n = \lVert \mathbf{x}_i-\mathbf{x}_j \rVert - (r_i+r_j)
$$
を拘束とする。重なっていれば $C_n<0$。

法線:
$$
\mathbf{n}_{ij}=
\frac{\mathbf{x}_i-\mathbf{x}_j}
{\lVert \mathbf{x}_i-\mathbf{x}_j \rVert}
$$

XPBD更新:
$$
\Delta\lambda_n =
\frac{-C_n-\alpha_n\lambda_n}
{(w_i+w_j)+\alpha_n}
$$
$$
\lambda_n \leftarrow \lambda_n+\Delta\lambda_n
$$

位置補正:
$$
\Delta\mathbf{x}_i^{(n)} = +w_i\,\mathbf{n}_{ij}\,\Delta\lambda_n
,\quad
\Delta\mathbf{x}_j^{(n)} = -w_j\,\mathbf{n}_{ij}\,\Delta\lambda_n
$$

### 6.2 接線拘束（摩擦）

相対速度:
$$
\mathbf{v}_{rel}=\mathbf{v}_i-\mathbf{v}_j
$$
接線方向（法線成分を除去して正規化）:
$$
\mathbf{t}=
\frac{\mathbf{v}_{rel}-(\mathbf{v}_{rel}\cdot\mathbf{n}_{ij})\mathbf{n}_{ij}}
{\lVert \mathbf{v}_{rel}-(\mathbf{v}_{rel}\cdot\mathbf{n}_{ij})\mathbf{n}_{ij}\rVert}
$$

速度ベース近似の接線拘束:
$$
C_t=(\mathbf{v}_{rel}\cdot\mathbf{t})\Delta t_g
$$

更新:
$$
\Delta\lambda_t =
\frac{-C_t-\alpha_t\lambda_t}
{(w_i+w_j)+\alpha_t}
$$

クーロン上限でクランプ:
$$
\lambda_t^{new} =
\mathrm{clamp}\left(
\lambda_t+\Delta\lambda_t,\,
-\mu_k|\lambda_n|,\,
+\mu_k|\lambda_n|
\right)
$$

位置補正:
$$
\Delta\mathbf{x}_i^{(t)} = +w_i\mathbf{t}(\lambda_t^{new}-\lambda_t)
,\quad
\Delta\mathbf{x}_j^{(t)} = -w_j\mathbf{t}(\lambda_t^{new}-\lambda_t)
$$

---

## 7. 粉体XPBD: 粒子-地形 / 粒子-オブジェクト接触

ここが現行実装の要点で、重力を拘束側へ入れる。

### 7.1 重力バイアス項

粒子 $i$ の有効時間 $\Delta t_i$ から
$$
\mathbf{b}_g=\mathbf{g}\Delta t_i^2
$$
を作る。

これは「その有効時間で自由落下したときの変位スケール」に対応する。

### 7.2 地形（SDF）接触

SDF距離を $d$、押し出し半径を $r$、法線を $\mathbf{n}$ とすると
$$
C_n = d-r + \mathbf{b}_g\cdot\mathbf{n}
$$

更新は片側質量なので
$$
\Delta\lambda_n =
\frac{-C_n-\alpha_n\lambda_n}
{w_i+\alpha_n}
$$
$$
\Delta\mathbf{x}_i^{(n)} = w_i\mathbf{n}\Delta\lambda_n
$$

### 7.3 オブジェクト（SDF）接触

候補オブジェクトのうち最深接触1件を採用し、同様に
$$
C_n = d_{obj}-r_{obj}+\mathbf{b}_g\cdot\mathbf{n}_{obj}
$$
を解く。

接線拘束は地形・オブジェクトとも同型:
$$
\Delta\lambda_t=
\frac{-C_t-\alpha_t\lambda_t}
{w_i+\alpha_t}
$$
を計算し、$\pm \mu_k|\lambda_n|$ でクランプする。

---

## 8. 反復・位置適用・速度再構成

`granular_iters` 回の反復で各拘束補正を蓄積した後、scheduled 粒子のみ
$$
\mathbf{x}_i \leftarrow \mathbf{x}_i + \Delta\mathbf{x}_i
$$
を適用する。

その後、`update_velocity` で
$$
\mathbf{v}_i \leftarrow
\frac{\mathbf{x}_i-\mathbf{x}_i^{prev}}
{\Delta t_i}
$$
と再構成する（$\Delta t_i=0$ の粒子は更新しない）。

---

## 9. 反発（restitution）

接触していて法線相対速度が接近中なら impulse を与える。

$$
v_n = (\mathbf{v}_i-\mathbf{v}_j)\cdot\mathbf{n}_{ij}<0
$$
$$
J = -\frac{(1+e)v_n}{w_i+w_j}
$$
$$
\mathbf{v}_i \leftarrow \mathbf{v}_i + w_iJ\mathbf{n}_{ij}
,\quad
\mathbf{v}_j \leftarrow \mathbf{v}_j - w_jJ\mathbf{n}_{ij}
$$

ここで $e$ は反発係数。

---

## 10. 現行モデルの意味と限界

現行挙動の重要点:

1. unscheduled 粒子は接触解法も反発もスキップされる
2. 粉体の重力は拘束側にのみ入る
3. したがって、**非接触状態の粉体は加速しない**

この 3 はユーザー観点で「空中で重力が効かない」に対応する。
現在の実装は「接触安定化」を優先した中間段階であり、
自由落下と接触安定を同時に満たすには外力モデルの次段設計が必要になる。

---

## 11. 次に設計すべき論点（粉体）

外力をXPBDへ寄せたまま自由落下も表現するための代表的な論点:

- 接触拘束とは別に「運動量更新（外力）」をどこで持つか
- マルチレート粒子での $\Delta t_i$ とコンプライアンス再正規化の整合
- 非接触時の運動と接触時の拘束解法を、同じエネルギー観で接続できるか

本ドキュメントはまず現行実装の説明を目的とし、
上記の改良案そのものは別セクション（将来版）で扱う。
