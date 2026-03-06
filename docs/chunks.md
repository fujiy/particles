# Chunk Residency / Active Tile Design

## 1. 本ドキュメントの役割

- `docs/chunks.md` は、MLS-MPM 計算領域の **chunk residency**, **slot 管理**, **active tile**, **incremental update** の詳細仕様を定義する。
- `docs/design.md` は概要と責務分離のみを持ち、本ドキュメントを正本とする。
- 対象は v1 の GPU 常駐・単一解像度グリッド経路であり、空間 LoD や block 境界フラックスは扱わない。

## 2. 設計方針

- world 全域を dense な grid バッファとして保持しない。
- 論理空間としては無限に近い一様グリッドを持ち、物理メモリ上では **resident chunk slot** のみを確保する。
- 粒子バッファは v1 では毎フレーム全量ソートしない。
- chunk residency は **incremental** に更新し、active tile は **毎ステップ GPU 上で再構築**する。
- CPU は疎な辞書管理と slot 割り当てを担い、GPU は粒子/格子の bulk な計算を担う。

## 3. 用語と定数

### 3.1 幾何

- `CELL_SIZE_M = 0.25`
- `NODE_SPACING_M = 0.125`
- `CHUNK_CELL_SIZE = 16`
- `CHUNK_NODE_SIZE = 32`
- `TILE_NODE_SIZE = 8`
- `TILES_PER_CHUNK_AXIS = 4`
- `TILES_PER_CHUNK = 16`

1 chunk は
- 16x16 cells
- 32x32 owned nodes
- 8m x 8m
を表す。

### 3.2 座標と所有

- `chunk_coord = (cx, cy)` は world 上で一意な `i32, i32` とする。
- `global_node = (nx, ny)` から `chunk_coord` を求めるときは
  - `chunk_coord = floor_div(global_node, 32)`
  - `local_node = mod(global_node, 32)`
  とする。
- 幾何学的な境界は node line 上に置く。
- ノード所有は half-open とし、chunk `(cx, cy)` は `x in [32*cx, 32*cx+32)`, `y in [32*cy, 32*cy+32)` を所有する。
- 右端/上端境界 node は隣接 chunk が所有する。

### 3.3 用語

- **occupied chunk**: `home` 粒子が1つ以上存在する chunk
- **resident chunk**: GPU 上に slot が割り当てられている chunk
- **halo residency**: occupied chunk の stencil / active tile 更新のために近傍 chunk も resident にしておくこと
- **slot id**: resident chunk に対応する GPU 上の物理連番
- **home chunk slot id**: 粒子が現在所属している occupied chunk の slot id
- **tile**: 8x8 nodes の最小 active 単位

## 4. なぜ occupied と resident を分けるか

粒子が chunk 境界近くにいると、P2G の stencil は自 chunk だけでなく隣接 chunk の node にも書き込む。したがって、

- `home` 粒子が存在する chunk だけを確保する

では不十分であり、occupied chunk の近傍も GPU 上で受け皿として resident にする必要がある。

v1 では **occupied chunk の 3x3 近傍**を resident 候補とする。これにより
- P2G の chunk 跨ぎ書き込み
- active tile の近傍 mark
- grid update 時の境界 tile 処理
を単純化する。

## 5. バッファ構成

## 5.1 CPU 側メタデータ

### 5.1.1 Chunk Slot Buffer

index は `slot id`。

各 slot は少なくとも以下を持つ。
- `chunk_coord_x: i32`
- `chunk_coord_y: i32`
- `occupied_particle_count: u32`
- `halo_ref_count: u32`
- `is_allocated: bool`

意味:
- `occupied_particle_count > 0` なら occupied chunk
- `halo_ref_count > 0` なら、occupied chunk の 3x3 近傍として resident である必要がある
- `occupied_particle_count == 0` かつ `halo_ref_count == 0` なら slot を free list に戻せる

### 5.1.2 Chunk Table

- key: `chunk_coord (i32, i32)`
- value: `slot id`

CPU 側の正本辞書。resident chunk のみを保持する。

### 5.1.3 Free List

- 空き `slot id` の再利用キュー
- free list が空なら末尾に新規 slot を追加してもよい

## 5.2 GPU 側メタデータ

### 5.2.1 Chunk Meta Buffer

index は `slot id`。

各 slot は少なくとも以下を持つ。
- `chunk_coord_x`
- `chunk_coord_y`
- `neighbor_slot_id[8]`
- `active_tile_mask: u32`

`neighbor_slot_id[8]` は Moore 近傍の slot id。順序は固定する。
例: `(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)`

無効近傍は `INVALID_SLOT` を入れる。

### 5.2.2 Chunk Meta Diff Buffer

CPU が chunk residency を更新したあと、変更のあった slot だけを GPU に送るための差分バッファ。

v1 では実装簡略化のため、変更 slot 数が小さければ
- 差分 upload
- 必要なら chunk meta 全量 upload
のどちらでもよい。

## 5.3 GPU 格子バッファ

格子実体は `slot id` ごとに SoA で持つ。

例:
- `grid_mass_water[slot_capacity * 1024]`
- `grid_momx_water[slot_capacity * 1024]`
- `grid_momy_water[slot_capacity * 1024]`
- `grid_mass_granular[slot_capacity * 1024]`
- `grid_momx_granular[...]`
- `grid_momy_granular[...]`
- `grid_aux[...]`

`1024 = 32 * 32`。

ローカル node index は
- `local_index = local_y * 32 + local_x`
- `global_index = slot_id * 1024 + local_index`

とする。

## 5.4 GPU 粒子バッファ

index は `particle id`。

各粒子は少なくとも以下を持つ。
- 位置・速度・質量・体積・`F_p`・`C_p` 等の物理量
- `material_id`, `phase_id`
- `home_chunk_slot_id`

v1 では粒子順序の chunk 局所性は保証しない。
必要なら将来、別経路で reorder を検討する。

## 5.5 GPU Movers Buffer

stream compaction 用の append バッファ。

各 record は少なくとも以下を持つ。
- `particle_id`
- `old_home_slot_id`
- `new_chunk_coord_x`
- `new_chunk_coord_y`

別に `mover_count` を持つ。

## 5.6 GPU Movers Result Buffer

CPU が `movers` を処理した結果を GPU に返すためのバッファ。

各 record は少なくとも以下を持つ。
- `particle_id`
- `new_home_slot_id`

必要なら `old_home_slot_id` を再掲して検証に使ってよい。

## 6. 不変条件

- `home_chunk_slot_id` は常に **occupied chunk** を指す。
- resident chunk は occupied chunk を必ず含む。
- occupied chunk が存在する限り、その 3x3 近傍は resident に保つ。
- GPU `neighbor_slot_id` は CPU chunk table と整合している必要がある。
- active tile mask は毎 step 0 初期化して再構築する。

## 7. ステップアルゴリズム

### 7.1 Step 0: 前提

前フレーム終了時点で粒子位置 `x_p` は更新済みであり、次フレーム開始時に `home_chunk_slot_id` が古い可能性がある。したがって step 冒頭で mover 判定を行う。

### 7.2 Step 1: GPU で mover を抽出

粒子並列で以下を行う。

1. `slot = home_chunk_slot_id[p]`
2. `old_coord = gpu_chunk_meta[slot].chunk_coord`
3. `new_coord = floor_div(world_pos_to_global_node(x_p), 32)`
4. `old_coord != new_coord` なら
   - `idx = atomicAdd(mover_count, 1)`
   - `movers[idx] = { particle_id, old_home_slot_id, new_chunk_coord }`

mover 抽出は **atomic append** を既定とする。mover 率が高く競合が問題になる場合のみ scan ベース compaction を検討する。

### 7.3 Step 2: CPU で residency を更新

`movers` を readback し、CPU 側で以下を行う。

#### 7.3.1 occupied 粒子数の更新

- old occupied chunk の `occupied_particle_count` を減算
- new occupied chunk の `occupied_particle_count` を加算

このとき、chunk の occupied/non-occupied 状態が 0↔1 で変化した場合のみ halo residency を更新する。

#### 7.3.2 halo residency の更新

occupied 状態が 0→1 になった chunk `c` について、`c` の 3x3 近傍すべての `halo_ref_count` を増やす。

occupied 状態が 1→0 になった chunk `c` について、`c` の 3x3 近傍すべての `halo_ref_count` を減らす。

これにより、境界粒子の P2G 先となる近傍 chunk が resident に維持される。

#### 7.3.3 slot の確保と解放

- 更新で新たに resident が必要になった chunk は、`chunk_table` を引いて
  - 既存 slot があれば再利用
  - なければ free list から払い出し
  - free list が空なら末尾追加
- `occupied_particle_count == 0` かつ `halo_ref_count == 0` になった slot は解放候補とする

#### 7.3.4 近傍 slot 情報の再計算

変更があった slot と、その Moore 近傍 slot について `neighbor_slot_id[8]` を再計算する。

### 7.4 Step 3: CPU から GPU へ差分反映

GPU に返すもの:
- `movers_result`: `particle_id -> new_home_slot_id`
- 変更された slot の `chunk meta diff`

必要に応じて以下も返してよい。
- 新規確保 slot の初期化リスト
- 解放 slot のクリア指示

### 7.5 Step 4: GPU で粒子の home slot を更新

`movers_result` を粒子並列または record 並列で反映し、`particle[p].home_chunk_slot_id = new_home_slot_id` を更新する。

### 7.6 Step 5: GPU で active tile を mark

まず全 resident slot の `active_tile_mask` を 0 に初期化する。

その後、粒子並列で P2G stencil が触る tile を `atomicOr` で mark する。

重要:
- tile mark は **自 chunk だけでなく、neighbor slot 側の tile** にも行う
- これにより、粒子は存在しないが境界 stencil で書き込まれる halo chunk の tile も active にできる

v1 では 1 chunk の active tile を `u32` 1個で持つ。

## 7.7 Step 6: active tile のみ clear

active tile mask を走査し、対応する node だけを 0 クリアする。

clear 対象:
- 材料別質量
- 材料別運動量
- 材料別補助バッファ
- 必要なら active tile ごとの一時領域

active でない tile は前 step のゴミが残っていてよいが、**次の計算で参照されない**ことが前提。

### 7.8 Step 7: P2G

粒子並列で P2G を行う。

- `home_chunk_slot_id` と `neighbor_slot_id` を使って、stencil が属する slot を解決する
- node 書き込み先は `slot_id * 1024 + local_index`
- 材料別配列へ atomic add する

### 7.9 Step 8: Grid Update

node 並列または tile 並列で、active tile に属する node のみ更新する。

処理内容:
- 内部力
- 外力
- 地形 SDF 境界補正
- 水-粉体連成
- node 速度更新

### 7.10 Step 9: G2P

粒子並列で G2P を行う。

- `home_chunk_slot_id` と `neighbor_slot_id` を用いて必要な node を読む
- 粒子速度・位置・`C_p`・`F_p`・粉体の `v_vol_p` を更新する

### 7.11 Step 10: 統計とデバッグ

収集例:
- `mover_count`
- occupied chunk 数
- resident chunk 数
- active tile 数
- 無効 neighbor 参照数
- slot 再利用数 / 新規割当数 / 解放数

## 8. sort/unique を常用しない理由

v1 は incremental 方式を採用するため、steady-state で毎フレーム
- 全粒子の chunk key を生成して sort/unique
- 全粒子を slot id 順に再整列
を要求しない。

理由:
- world chunk key は `i32 x 2` が正本であり、64bit 相当 key の full sort は避けたい
- 境界を跨ぐ粒子は通常フレームでは少数である
- chunk residency は低カードinalityな疎メタデータであり CPU が扱いやすい

ただし fallback として full rebuild を持つことは許容する。

## 9. full rebuild fallback

以下のようなフレームでは full rebuild を選んでよい。
- `mover_count` が極端に多い
- 大量 spawn / despawn
- セーブデータ読込直後
- パラメータ変更で residency を再初期化したい

full rebuild で行うこと:
- 全粒子から occupied chunk 集合を再構築
- occupied/halo residency を再計算
- chunk table / slot buffer / neighbor slot を再初期化

これは **正しさのための必須経路ではなく、性能と実装簡略化のための fallback** と位置付ける。

## 10. active tile の詳細

### 10.1 タイル番号

chunk 内 `8x8 nodes` tile の `(tx, ty)` に対し
- `tile_id = ty * 4 + tx`

### 10.2 bitmask

- `active_tile_mask & (1 << tile_id) != 0` で active 判定
- 将来 tile 数を増やす場合は `u32xN` か別配列へ拡張する

### 10.3 mark の対象

粒子の P2G stencil が触る node の bounding box を tile に変換し、該当 tile を全部 mark する。

粒子が chunk 境界近くにいる場合は
- 自 chunk
- x 方向近傍
- y 方向近傍
- 対角近傍
の tile を mark し得る。

## 11. Neighbor Slot の運用

### 11.1 近傍解決

chunk 内局所 node から、どの slot に属するかを
- `local_x`, `local_y`
- boundary 越え判定
から決める。

たとえば right/top を half-open 所有にしているため、stencil の一部が
- `local_x >= 32`
- `local_y >= 32`
- `local_x < 0`
- `local_y < 0`
に出た場合は neighbor slot に送る。

### 11.2 更新タイミング

`neighbor_slot_id[8]` は chunk residency 変更時のみ更新する。毎フレーム全量再計算は不要。

## 12. CPU / GPU 責務分離

### CPU が持つもの

- resident chunk の辞書管理
- slot 割り当て / 解放
- occupied 粒子数
- halo residency ref count
- neighbor slot の再計算

### GPU が持つもの

- 粒子物理量
- chunk meta mirror
- 格子実体
- mover 抽出
- active tile mark / clear
- P2G / Grid Update / G2P

## 13. 将来拡張

- mover 率が低いことを前提にした v1 のあと、必要なら粒子 reorder を追加する
- chunk 内 tile list を compaction して、grid update を tile range 単位に indirect dispatch してもよい
- region / cluster は実行スケジューリング上の概念として後から追加できるが、v1 では first-class object にしない
- slot meta の世代カウンタや debug cookie を追加して、古い `home_chunk_slot_id` の検出を強化してよい
