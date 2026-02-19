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
