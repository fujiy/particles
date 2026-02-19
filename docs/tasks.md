# Tasks

## Todo
- (No open tasks)

## Done
- (Keep completed tasks here; do not delete history)
- [x] 水粒子v1アルゴリズム仕様を `design.md` に確定（PBF/XPBD + Frozen地形干渉 + Space/R操作）
- [x] 地形v0実装仕様の確定（`cell_size=0.25m`, `chunk=32x32`, `TerrainWorld/ParticleWorld` 分離、Plugin/描画フロー）
- [x] Bevy開発環境の構築（Rust + Bevy 0.18系）
- [x] 開発高速化設定の導入（`bevy/dynamic_linking`, `bevy/file_watcher`）
- [x] ホットリロード方針の確定と検証（アセット再読込 + コードホットパッチ可否）
- [x] ファイル監視ベースの自動再コンパイル・自動再ロード手順の整備
- [x] 初期運用方針を「安定運用のみ」に固定（`file_watcher` + `dynamic_linking`、`hotpatching` は採用しない）
- [x] 地形データ管理の初期実装（1ブロック=1セル、全地形 `Frozen` 固定）
- [x] 地形の初期生成・保持・参照APIを実装（チャンク/グリッド単位の読み書き）
- [x] シミュレータ基本構造を実装（`ParticleWorld`/`TerrainWorld` Resource と更新ループの骨組み）
- [x] Bevy Plugin化（`SimulationPlugin`）して `App` から有効化できるようにする
- [x] `FixedUpdate` にシミュレータ更新システムを接続（現段階では地形は非動的更新）
- [x] 地形描画の初期実装（グリッド->タイル/スプライト描画）
- [x] 地形更新と描画更新の接続（変更セルのみ dirty 更新）
- [x] 最小動作確認シーンを作成（地形の生成・表示・カメラ追従）
- [x] カメラ操作を実装（スクロール拡大縮小、`WASD`/矢印パン、中ボタンまたは`Option(Alt)+ドラッグ`パン）
- [x] カメラ処理をモジュール分離し、調整定数を先頭へ集約（`camera_controller.rs`）
- [x] 拡大縮小をカーソル位置基準に変更し、スクロール感度を調整
- [x] 地形描画を単色から「materialごと4色パレット + 決定論的ランダム埋め」に変更
- [x] 世界生成を固定`4x4`チャンクに変更し、カメラ移動時の追加チャンク生成を停止
- [x] 世界の左右端に上端までの壁を追加（上境界はオープン）
- [x] 水粒子シミュレーションの定数セットを実装（重力、substep、反復回数、半径、粘性）
- [x] `ParticleWorld` を水粒子SoA構造に拡張（`pos/prev_pos/vel/mass/material` + ワークバッファ）
- [x] 動的粒子用の近傍探索グリッドを実装（substepごと再構築）
- [x] 水の密度拘束ソルバ（PBF/XPBD系）を実装（density/lambda/delta_pos）
- [x] 水粒子と `Frozen` 地形セルの衝突投影を実装（円 vs セルAABB）
- [x] `FixedUpdate` に水粒子stepを接続（`SUBSTEPS` ループ）
- [x] 水粒子の可視化を実装（まずはデバッグ描画で可）
- [x] `Space` キーでシミュレーション再生/停止トグルを実装
- [x] `R` キーで初期状態リセットを実装（地形再生成 + 水粒子再生成 + 停止状態）
- [x] 受け入れ条件テストを実施（落下、非貫通、停止中不変、リセット復元）

## Design Feedback (from Impl sessions)
- (If implementation reveals required design changes, record requests here)
