# Chunk Residency / Active Tile Design

## 1. 本ドキュメントの役割

- `docs/chunks.md` は、MLS-MPM 計算領域の **chunk residency**, **slot 管理**, **active tile**, **CPU/GPU 同期** の詳細仕様を定義する。
- `docs/design.md` は概要と責務分離のみを持ち、本ドキュメントを正本とする。
- 対象は GPU 常駐・単一解像度グリッド経路であり、空間 LoD や block 境界フラックスは扱わない。

## 2. 設計方針

- world 全域を dense な grid バッファとして保持しない。
- 物理メモリ上では **resident chunk slot** のみを確保する。
- 粒子バッファは毎フレーム全量ソートしない。
- active tile は毎ステップ GPU 上で再構築する。
- CPU/GPU 同期は粒子ごとの mover 全件ではなく、**occupancy / residency change event** を主経路にする。
- 粒子ごとの mover 同期は **exceptional mover fallback** 専用に残す。
- slot table / neighbor 差分の反映は非同期キューで扱い、通常ステップを同期停止させない。

## 3. 用語と定数

### 3.1 幾何

- `CELL_SIZE_M = 0.25`
- `NODE_SPACING_M = 0.125`
- `CHUNK_CELL_SIZE = 16`
- `CHUNK_NODE_SIZE = 32`
- `TILE_NODE_SIZE = 8`

1 chunk は
- 16x16 cells
- 32x32 owned nodes
- 8m x 8m
を表す。

### 3.2 座標と所有

- `chunk_coord = (cx, cy)` は world 上で一意な `i32, i32`。
- `global_node = (nx, ny)` から
  - `chunk_coord = floor_div(global_node, 32)`
  - `local_node = mod(global_node, 32)`
- ノード所有は half-open とし、chunk `(cx, cy)` は
  - `x in [32*cx, 32*cx+32)`
  - `y in [32*cy, 32*cy+32)`
  を所有する。

### 3.3 用語

- **occupied chunk**: `home` 粒子が1つ以上存在する chunk
- **resident chunk**: `occupied chunk` とその Moore 近傍 8 chunk（3x3）
- **slot id**: resident chunk に対応する GPU 上の物理連番
- **home chunk slot id**: 粒子が現在所属している occupied chunk の slot id
- **frontier chunk**: occupied かつ `neighbor_slot_id` に `INVALID` を含む chunk
- **exceptional mover**: `delta_chunk` が Moore 近傍外（8近傍外）に飛ぶ粒子移動

## 4. resident 定義と free 条件

- resident は `occupied` のみでは決まらない。
- resident 判定は常に「occupied + halo」で行う。
- slot の free 条件は以下の両方を満たすこと。
  - `occupied_particle_count == 0`
  - `resident_ref_count == 0`（または同値の halo 判定が false）

## 5. バッファ構成

## 5.1 CPU 側メタデータ

### 5.1.1 Chunk Slot Ledger

index は `slot id`。

各 slot は少なくとも以下を持つ。
- `chunk_coord_x: i32`
- `chunk_coord_y: i32`
- `occupied_particle_count: u32`
- `resident_ref_count: u32`
- `is_allocated: bool`

### 5.1.2 Chunk Table

- key: `chunk_coord`
- value: `slot id`

resident chunk の正本辞書として使う。

### 5.1.3 Free List

- 空き `slot id` の再利用キュー
- 必要なら末尾拡張

## 5.2 GPU 側メタデータ

### 5.2.1 Chunk Meta Buffer

index は `slot id`。

各 slot は少なくとも以下を持つ。
- `chunk_coord_x`
- `chunk_coord_y`
- `neighbor_slot_id[8]`
- `active_tile_mask: u32`
- `particle_count_curr: u32`
- `particle_count_next: u32`
- `occupied_bit_curr: u32`
- `occupied_bit_next: u32`

`particle_count_*` と `occupied_bit_*` は occupancy 変化抽出に使う。

### 5.2.2 Event Buffer（GPU -> CPU）

粒子 mover 全件ではなく、以下イベントを CPU へ返す。
- `newly_occupied_chunk`
- `newly_empty_chunk`
- `frontier_expansion_request`
- `exceptional_mover`

### 5.2.3 Exceptional Mover Buffer（GPU -> CPU -> GPU）

通常経路では使わない。

各 record:
- `particle_id`
- `old_home_slot_id`
- `new_chunk_coord`

CPU が slot を解決し、必要時のみ GPU へ `particle_id -> new_home_slot_id` を返す。

## 5.3 GPU 格子バッファ

- 格子実体は `slot id * 1024 + local_index` で管理する。
- 材料別質量・運動量・補助配列は slot-major の SoA を維持する。

## 5.4 GPU 粒子バッファ

- 粒子は位置・速度・応力関連量に加え `home_chunk_slot_id` を持つ。
- 通常の隣接 chunk 移動は GPU 内で `home_chunk_slot_id` を更新する。

## 6. 不変条件

- `home_chunk_slot_id` は occupied chunk を指す。
- occupied chunk は必ず resident。
- occupied chunk の 8 近傍は常に resident。
- `neighbor_slot_id[8]` は CPU の `chunk_table` と整合する。
- `not occupied && not halo` のときのみ free。

## 7. ステップアルゴリズム

### 7.1 Step 0: 前提

- 前ステップの G2P 後、粒子位置 `x_p` は更新済み。
- 次ステップ冒頭で `new home chunk` を再評価する。

### 7.2 Step 1: GPU で new home chunk 判定

粒子並列で以下を実行。
- `old_slot = home_chunk_slot_id[p]`
- `old_coord = chunk_meta[old_slot].chunk_coord`
- `new_coord = world_pos -> chunk_coord`
- `delta = new_coord - old_coord`

### 7.3 Step 2: 隣接移動は GPU 内で完結

- `delta` が Moore 近傍内なら `neighbor_slot_id` で新 slot を解決し、`home_chunk_slot_id` を GPU 内で更新する。
- これが通常経路。

### 7.4 Step 3: occupancy 集計

- 粒子並列で `particle_count_next[slot]` を atomic 加算。
- chunk 並列で `curr/next` を比較し、以下を抽出。
  - `0 -> >0`: newly occupied
  - `>0 -> 0`: newly empty
  - occupied かつ invalid neighbor: frontier expansion request
- `delta` が 8近傍外の粒子は exceptional mover として別バッファに積む。

### 7.5 Step 4: CPU へイベント同期

CPU へ送るのは以下のみ。
- newly occupied chunk
- newly empty chunk
- frontier expansion request
- exceptional mover（必要時のみ）

### 7.6 Step 5: CPU で residency / slot table 更新

- newly occupied / newly empty を使って `occupied_particle_count` と `resident_ref_count` を更新。
- resident 集合を再判定し、slot allocate/free を行う。
- `neighbor_slot_id` 差分を再計算する。
- exceptional mover がある場合のみ `chunk_coord -> slot_id` を解決し、GPU へ反映リストを作る。
- この更新は非同期でよく、GPU 側ステップは直近の安定スナップショットを使って継続してよい。

### 7.7 Step 6: GPU へ差分反映

- `chunk meta diff` を upload。
- exceptional mover 解決結果がある場合のみ `particle_id -> new_home_slot_id` を upload。
- 通常フレームは粒子 mover 全件 upload を行わない。

### 7.8 Step 7-11: 物理本体

- Active Tile Build
- Active Tile Clear
- P2G
- Grid Update
- G2P

## 8. 通常経路と例外経路

### 8.1 通常経路

- 隣接移動は GPU 内で完結。
- CPU は occupancy/residency change だけ処理。
- 通信量は frontier 変化量にほぼ比例する。

### 8.2 例外経路

- 8近傍外移動だけ exceptional mover として処理。
- CPU で slot 解決して GPU に返す。
- mover 全件 readback は行わない。

## 9. full rebuild fallback

以下では full rebuild を許容。
- occupant/resident 整合が崩れた
- slot 容量上限を超えた
- 大量 spawn/despawn
- セーブロード直後

full rebuild では
- occupied 集合再構築
- resident(occupied+halo) 再構築
- chunk table / neighbor / counters 再初期化
を実行する。

## 10. 計測指標

- occupied chunk 数
- resident chunk 数
- newly occupied / newly empty 数
- frontier expansion request 数
- exceptional mover 数
- slot allocate/free 数
- fallback 回数
- invalid neighbor 参照数

## 11. CPU / GPU 責務分離

### CPU

- resident 辞書管理
- slot allocate/free
- `occupied_particle_count`, `resident_ref_count` 管理
- neighbor 差分更新
- exceptional mover のみ slot 解決

### GPU

- 粒子物理量
- chunk meta mirror
- occupancy 集計
- occupancy/residency change event 抽出
- 通常隣接移動の home slot 更新
- active tile / P2G / Grid / G2P

## 12. 将来拡張

- exceptional mover 率が高いシーン向けに multi-hop neighbor 解決を追加可能。
- chunk meta に世代カウンタや debug cookie を持たせ、古い slot 参照検出を強化可能。
- event compaction を scan ベースに置換し、高密度ケースの競合を低減可能。
