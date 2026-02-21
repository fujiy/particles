# particles

Bevy 0.18 ベースの 2D 物理シミュレーション実験プロジェクトです。

## 実行方法

### 通常実行

```bash
cargo run
```

## ファイル監視で自動コンパイル実行

`cargo-watch` を使って、ファイル変更時に自動で再コンパイルして再実行します。

```bash
cargo install cargo-watch
cargo watch -x run
```

- Rust コード変更: 自動で再コンパイルして再起動
- アセット変更: `bevy/file_watcher` により自動再読込

## Tracy でプロファイリング

Bevy の `trace` と `LogPlugin` の `custom_layer` で Tracy を有効化できます。
物理ステップ内は `physics::*` スパンで分割して計測されます（`predict_positions` / `solve_density_constraints` / `shape_matching` など）。
このプロジェクトの Tracy は `ondemand` 構成なので、Profiler 未接続時の負荷を抑えます。

### Tracy プロファイル（通常実行）

```bash
cargo run --release --features tracy
```

### Tracy プロファイル（ファイル監視）

```bash
cargo watch -x "run --release --features tracy"
```

- もし表示が不安定な場合は `cargo run --no-default-features --features tracy` で `dynamic_linking` を外して試してください
