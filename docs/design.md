# 物理シミュレーション×工業 2Dサンドボックス Design Doc

## 1. 本ドキュメントの役割

- `docs/design.md` は現在の目標仕様のみを記載する。
- 実装順序、背景、移行文脈は `docs/tasks.md` で管理する。
- chunk residency / slot 管理 / active tile の詳細アルゴリズムは `docs/chunks.md` を正本とする。

## 2. プロダクト目標

- 2D横視点のドット絵サンドボックスで、流体・地形・工業設備の相互作用を扱う。
- 連続体の中核ソルバは MLS-MPM（明示更新）とする。
- 連続体計算は **単一解像度グリッド + GPU常駐データ** を前提に構築する。
- 近傍のみ計算する最適化は、空間LoDではなく **sparse chunk residency + active tile 制御** で実現する。
- 水、粉体、剛体連成を段階的に追加できる共通インターフェースを維持する。

## 3. 単位系と空間定義

### 3.1 単位系

- SIベースを使用し、Bevy座標は `1.0 = 1m` とする。
- 地形セルサイズは `CELL_SIZE_M = 0.25`。
- 描画基準は `PIXELS_PER_METER = 64`、1セルは16px。

### 3.2 セル・ノード・チャンク

- 地形の最小編集単位は **16x16 cells** とする。
- 物理グリッド間隔は `NODE_SPACING_M = CELL_SIZE_M / 2 = 0.125m` とする。
- 1 chunk は **16x16 cells = 32x32 owned nodes** とする。
- tile は chunk 内の計算最適化単位で、v1 では **8x8 nodes** を既定とする。
  - したがって 1 chunk は `4x4 = 16 tiles` を持つ。
- 幾何学的な cell / chunk 境界は node line 上に置く。
- ノード所有は half-open とし、chunk `(cx, cy)` は node index の
  - `x in [32*cx, 32*cx + 32)`
  - `y in [32*cy, 32*cy + 32)`
  を所有する。
- 右端・上端の境界 node は隣接 chunk が所有する。

### 3.3 座標系

- 座標系は `global_cell` / `chunk_coord` / `local_cell` / `global_node` / `local_node` / `world_pos_m` を明示的に分離する。
- `chunk_coord` は地形・物理で共通の **世界チャンク座標**（`i32, i32`）とする。
- world は地球オーダーまで拡張可能なことを目標とし、chunk key は packed 32bit ではなく **`i32 x 2` の固定幅座標**を正本とする。

## 4. 世界表現

### 4.1 TerrainWorld

- `TerrainWorld` は chunk 疎管理の地形台帳とする。
- セルは少なくとも以下を保持する。
  - `Empty`
  - `Solid { material, hp }`
- 地形は通常 `Frozen` の静的境界として扱い、MLS-MPM 側には SDF 境界として供給する。

### 4.2 ContinuumParticleWorld

- 連続体粒子は SoA で保持する。
- 水・粉体粒子の最低限属性。
  - `x_p`（位置）
  - `v_p`（速度）
  - `m_p`（粒子質量）
  - `V0_p`（初期体積）
  - `F_p`（変形勾配）
  - `C_p`（APIC affine 行列）
  - `material_id`
  - `phase_id`（`Water` / `GranularSoil` / `GranularSand`）
  - `v_vol_p`（粉体の体積補正スカラー、粉体のみ）
  - `home_chunk_slot_id`（現在所属する resident chunk slot）
- 粒子状態の計算正本は GPU バッファに置く。
- CPU 側は UI/シナリオ入力と最小限メタデータのみを保持する。
- v1 の steady-state では粒子バッファの毎フレーム並べ替えを前提にしない。

### 4.3 SparseChunkGridField

- MLS-MPM 計算格子は **単一解像度の一様グリッド**を論理空間として持つ。
- ただし物理メモリ上では、world 全域を dense に持たず **resident chunk slot** のみを確保する。
- chunk slot は GPU 上の格子データ実体を指す連番 ID とする。
- chunk slot ごとに以下の GPU メタデータを保持する。
  - chunk 座標（`chunk_coord_x`, `chunk_coord_y`）
  - `neighbor_slot_id[8]`（Moore近傍）
  - `active_tile_mask`（`u32`）
- 地形境界参照のため、slot ごとに `ChunkSdfBuffer`（`phi`, `normal`）を保持する。
- v1 では以下を明確に分ける。
  - **occupied chunk**: `home` 粒子が1つ以上存在する chunk
  - **resident chunk**: occupied chunk 本体と、その Moore 近傍 8 chunk（3x3）を含む、GPU 上に確保された chunk
- resident chunk の確保・解放は CPU が保持する `chunk_coord -> slot_id` 台帳で管理する。
- slot table / `neighbor_slot_id` 差分の GPU 反映は非同期で扱い、通常ステップを同期停止させない。
- 導入初段（`MPM-CHUNK-01`）では resident chunk を起動時に one-shot 構築し、実行中は固定する。
- 次段（`MPM-CHUNK-02` 以降）で occupancy/residency change event を使った incremental 更新へ移行する。
- 詳細アルゴリズムとバッファ構成は `docs/chunks.md` を参照する。

### 4.4 Active Tile Metadata

- 計算削減は active tile 方式で行う。
- tile は固定サイズの node 集合（v1: `8x8` nodes）とする。
- 各ステップで粒子分布から active tile を GPU 上で再構築し、非 active tile の clear/update を省略する。
- v1 では 1 chunk あたり 16 tiles を `u32` bitmask で表現する。

### 4.5 CouplingWorld

- 連成層は「境界条件入力」と「反力出力」の交換を責務に持つ。
- v1 で有効化する連成。
  - 地形 SDF 境界（必須）
  - 水-粉体（土/砂）のノード運動量交換（非貫通 + クーロン摩擦 + 対称ドラッグ）
- 後段で追加する連成。
  - 剛体境界速度場と反力受け渡し

## 5. マテリアル

- 粒子は `material_id` を持つ。
- 物性は `material_id -> category -> parameter set` の参照で与える。
- 水で必須な物性。
  - `rho0`（基準密度）
  - `bulk_modulus`
  - `alpha_apic`（APIC↔PIC ブレンド係数）
  - `boundary_friction`
  - `negative_pressure_fill_min`, `negative_pressure_fill_max`
- 粉体（土/砂）で必須な物性（Drucker-Prager）。
  - `E`（ヤング率）
  - `nu`（ポアソン比）
  - `friction_angle_deg`（内部摩擦角）
  - `dilation_angle_deg`（ダイレイタンシー角）
  - `boundary_friction`
  - `alpha_apic`
- 水-粉体連成で共有する物性。
  - `drag_rate`
  - `mu_wg`（相間摩擦係数）

## 6. 物理シミュレーション（MLS-MPM）

### 6.1 基本フロー

- `FixedUpdate` で基準刻み `dt_frame` を進める。
- 1 step の実行順は以下を既定とする。
  1. `Home Chunk Update`: 粒子ごとに world chunk 座標を再評価し、隣接移動は GPU 内で `home_chunk_slot_id` を更新する
  2. `Occupancy Event Extract`: GPU で `newly occupied` / `newly empty` / `frontier expansion request` / `exceptional mover` を抽出する
  3. `Chunk Residency Update`: CPU 側で `chunk_coord -> slot_id` 台帳を更新し、resident 追加/解放と近傍 slot 情報差分を反映する（exceptional mover の slot 解決は必要時のみ）
  4. `Active Tile Build`: GPU 上で active tile mask を再構築する
  5. `Active Tile Clear`: active tile に対応する grid ノードだけを clear する
  6. `P2G`: 粒子質量・運動量・応力寄与を格子へ転送する
  7. `Grid Update`: 外力・内部力・境界条件・材料間連成を適用して node 速度を更新する
  8. `G2P`: 格子速度から粒子速度・位置・`C_p`・`F_p` を更新する
  9. `Post Process`: 保存量と監視量を集計する
- 段階導入ポリシー:
  - **Stage A（chunk導入初段）**: `Home Chunk Update` と `Chunk Residency Update` をスキップし、起動時に作った固定resident集合を使う。
  - **Stage B（incremental化）**: occupancy/residency event readback + CPU台帳更新 + chunk meta差分反映を行う。
- v1 の steady-state は incremental chunk 管理を前提にし、毎フレーム full sort / unique を要求しない。
- frontier 変化や exceptional mover が極端に増えたフレームでは full rebuild を fallback として許容するが、通常経路の正しさ条件ではない。

### 6.2 水の構成則

- 水は弱圧縮性モデルを採用する。
- 圧力は `J_p = det(F_p)` 由来の弱圧縮 EOS で評価する。
- 負圧（張力）側は、ノード/粒子の充填率に応じて滑らかに抑制する。
- 水の応力は等方圧を既定とし、せん断蓄積は G2P 後の等方化で捨てる。

### 6.3 粉体の構成則（Drucker-Prager）

- 土/砂の粉体は frictional elastoplastic MPM（Drucker-Prager）を採用する。
- 応力更新は GPU 向けの **SVD + log-strain 射影**を既定とする。
- 粉体粒子は `v_vol_p` を保持し、体積補正スカラーとして更新する。
- 粉体の引張強度は持たない。

### 6.4 水-粉体連成（非液状化）

- 水と粉体は同一グリッド上の別 material phase として更新する。
- 各 node は材料別の質量・運動量・速度場を持つ。
- v1 は「非液状化」モデルとし、以下を対象外とする。
  - 間隙水圧の陽的追跡
  - 有効応力モデル
  - 飽和度・透水の進化
- 交換量は node 上で定義し、運動量の対称更新を行う。
  - 法線方向の非貫通補正
  - 接線方向のクーロン摩擦
  - 追加安定化としての対称ドラッグ

### 6.5 境界条件（地形）

- 地形境界は SDF で与える。
- SDF は `slot_id * 1024 + local_node_index` で引ける chunk-local バッファを正本とする。
- node が `phi(x) < threshold` の場合に補正する。
- 補正仕様。
  - 法線方向の非貫通を強制する
  - 接線は **クーロン摩擦（stick/slip の近似）** で処理する
  - 深部侵入時は押し戻し速度を加算する
- 導入初段では「初期地形のみ」を chunk SDF に one-shot 反映し、実行中の差分更新は行わない。
- 次段で地形改変/ロード差分に応じた chunk SDF の dirty 更新を導入する。

### 6.6 時間刻みと安定条件

- グローバル刻み `dt` は CFL 条件を満たすよう設定する。
- 必要な場合のみ、同一解像度グリッド上で一様 substep を導入する。
- block ごとに異なる `dt_b` を持つ時間 LoD は現行仕様に含めない。

### 6.7 計算領域最適化（resident chunk + active tile）

- 物理計算対象は resident chunk 内の **active tile とその必要近傍**に限定する。
- chunk residency は incremental に維持し、active tile 判定は GPU 上で毎ステップ更新する。
- 非 active tile は clear/update をスキップする。
- region / cluster は v1 では first-class な実行オブジェクトにしない。

### 6.8 保存量ポリシー

- 質量保存は P2G/G2P の離散化で満たす。
- 運動量交換（境界・連成）は必ず対称更新する。
- エネルギーは厳密保存を要求しないが、散逸量メトリクスを保持する。

### 6.9 GPU実行モデル

- 粒子/格子バッファは GPU 常駐とし、毎ステップ全量 readback しない。
- CPU へ戻すのは以下に限定する。
  - occupancy/residency change の低カードinalityイベント
  - exceptional mover（8近傍外移動）情報
  - 最小限の統計量とデバッグ用途データ
- 1 step は GPU compute pass を中心に連続実行する。
- v1 の常用経路では、粒子を chunk 順に全量ソートすることを前提にしない。
- `MPM-CHUNK-01` の段階では residency 更新 readback を無効化し、GPU内完結の静的residencyで検証する。

### 6.10 旧CPU経路の整理

- 旧 CPU 版の連続体ソルバは本番経路として維持しない。
- 期待される挙動の確認に一時参照してよいが、不要になった CPU コードは速やかに削除する。

## 7. 連成ポリシー

### 7.1 地形連成（有効）

- 地形は静的境界として MLS-MPM に接続する。
- 地形改変は `TerrainWorld` の責務とし、物理フレーム境界で SDF を更新する。
- 地形 chunk 座標系と物理 chunk 座標系は一致させる。

### 7.2 粉体連成（有効化対象）

- 粉体は MLS-MPM の material phase として扱う（別ソルバ併存は採用しない）。
- v1 は Drucker-Prager + 非液状化水連成を採用する。
- 土/砂はパラメータセット差で表現し、構成則実装は共有する。

### 7.3 剛体連成（段階導入）

- 剛体との連成 I/F は GPU 主導で設計する。
- 剛体の連成データ経路も最終的に GPU へ統合する。

## 8. レンダリング

本章は設計の要点のみを示す。詳細仕様は `docs/render.md` を参照する。

### 8.1 地形（GPUキャッシュ + 2層）

- 地形描画は **GPU常駐キャッシュ**を中心に構築し、CPU→GPU の画像転送を常態化させない。
- 表示は `Near` / `Far` の **2層キャッシュ**で構成する。
  - `Near`: 近距離・高解像・「ガチタイル」表現（材料IDベース、8x8px タイルの最近傍拡大）。
  - `Far`: 遠距離・低解像の背景（材料の集約/密度表現）。`Near` の未更新領域の穴埋めと、極端なズームアウトの可視化を担う。
- 並進（パン）に対しては、画面 + マージン領域を保持した **リングバッファ更新**を行い、移動で新規に可視化される行/列（またはタイル矩形）のみを再計算する。
- ズームに対しては、内部LODを 2^k の離散スケールで管理し、ヒステリシス付きで切り替える。
  - `Near` は拡大モード（セルが >= 1px）では材料IDの直接描画を維持する。
  - サブピクセル領域では `Far` を優先して見せ、`Near` は解像度上限により早期に縮小モードへ遷移して負荷を抑える。
- `Near` / `Far` のキャッシュ解像度はユーザー設定（例: 1/1, 1/2, 1/4）で上限を設け、最悪ケース（セル≈1px）で全面更新にならないようにする。

### 8.2 水・粒子

- まずデバッグ用オーバーレイで粒子を円表示し、MPM 結果の妥当性を検証する。
- 検証完了後、ドット絵テクスチャ描画を GPU で実装する。
- ドット絵描画はフラグメントシェーダーを第一候補とし、必要に応じて compute 前処理を併用する。

### 8.3 Tileベース描画 / 更新スケジューリング

- `Chunk` は保存/生成単位、`Tile` は描画/更新最適化単位として分離する。
- required tile set はカメラ位置・ズームで決定し、`Near` / `Far` それぞれに dirty タイルキューを持つ。
- 1フレームあたりの更新タイル数・更新矩形面積に上限（budget）を設け、パン/ズームの急変時でもフレーム時間を平準化する。

## 9. ECSとモジュール方針

- 主要責務は plugin で分離する。
  - `physics_mpm_gpu`
  - `terrain`
  - `render_tile`
  - `interface`
  - `overlay`
- 連続体の実行経路は GPU に一本化する。

### 9.1 パラメータ資産管理

- 実行時調整対象のパラメータは Bevy Asset として `assets/params/` 配下で管理する。
- ファイル構成は責務単位で固定し、以下の6ファイルを正本とする。
  - `assets/params/camera.ron`
  - `assets/params/interface.ron`
  - `assets/params/overlay.ron`
  - `assets/params/palette.ron`
  - `assets/params/physics.ron`
  - `assets/params/render.ron`
- Rust/WGSL で使うパラメータは「Asset -> 検証/クランプ済み Resource -> GPU uniform/storage」の順で反映する。
- WGSL は asset を直接参照しない。shader から参照する値は必ず Rust 側で解決した uniform/storage 経由で渡す。
- hot reload は Bevy の標準 asset 更新通知を使用し、`Modified` 発生時に再検証して反映する。
- 不正値を検出した場合は新値を採用せず、直前の有効値を維持してログを出す。
- 実行中に変更しない定数（バッファサイズ、レイアウト、セルサイズ、workgroupサイズなど最適化/整合性に直結する値）は asset 管理対象外とする。
- 各 `.ron` ファイルでは、全パラメータに用途・単位・許容範囲をコメントで明記する。コメントのない項目は追加しない。

## 10. セーブ/ロードと世界生成

### 10.1 セーブ/ロード

- 保存対象。
  - 地形差分
  - 連続体粒子状態
  - 互換判定メタ情報（`save_version`, `generator_version`）
- active tile などのランタイム最適化情報は保存しない。
- chunk residency / active tile / mover buffer などの一時最適化状態は保存しない。

### 10.2 決定論的地形生成

- ワールドは seed ベースの決定論生成を前提とする。
- 未変更チャンクは再生成し、変更チャンクのみ差分保存する。

## 11. テスト方針

### 11.1 Unit test

- 転送カーネル、構成則、CFL判定、境界補正、chunk 座標変換を単体テストで検証する。

### 11.2 Physics integration test

- headless シナリオテストで固定入力・固定 step の再現性を検証する。
- GPU 経路の最小受け入れとして `water_drop` を必須シナリオにする。
- 描画検証では、デバッグ円オーバーレイと GPU ドット絵描画の両方で破綻がないことを確認する。
- chunk residency の回帰検証として、chunk 境界を跨ぐ粒子移流シナリオを必須にする。

### 11.3 計測

- フレームごとに以下を収集する。
  - 総質量誤差
  - 総運動量誤差
  - 最大 CFL 比
  - resident chunk 数 / active tile 数
  - newly occupied / newly empty / frontier expansion request / exceptional mover 数
  - step wall/cpu
  - GPU pass 別時間（home-chunk-update / occupancy-event-extract / active-build / p2g / grid / g2p / render）
