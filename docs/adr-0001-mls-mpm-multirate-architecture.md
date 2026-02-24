# ADR-0001: 広域連続シミュレーション向け MLS-MPM 採用とマルチレート設計

- Status: Accepted
- Date: 2026-02-24
- Deciders: Physics/Simulation
- Scope: 連続体（流体・粉体系）の基盤ソルバ選定、広域ワールドでの計算資源配分方針

## Context

現行実装は粒子ベース（主に PBD/XPBD）で、Sub-block ごとの更新間引き（マルチレート）を導入している。  
この方式は近傍での安定性は高いが、広域ワールドで以下の課題が顕在化した。

1. 低頻度 block で有効 $\Delta t$ が大きくなると、外力（特に重力）の寄与が再更新時に集中し、接触拘束での戻し量が増えて跳ねやすい。  
2. 「外力を先に積分してから拘束で戻す」分離は、静止近傍や接触保持で不利。  
3. 計算対象/非対象を二値で切る sleep 方式は、境界整合（影響伝播、出入り、反作用）を保ちにくい。  
4. 広域ワールドでは「局所だけ激しく動き、遠方は低活動」が常態で、時間 LoD だけでなく空間 LoD も必要。  
5. 将来要件として、流体・粉体系・地形/剛体連成・資源保存・運動量整合を同時に満たしたい。

## Decision Drivers

1. 広域で計算量を制御できること（空間 LoD + 時間 LoD）。
2. 接触/境界を含むシーンでの安定性。
3. 複数連続体の統一的な取り扱い。
4. 質量保存と運動量整合を設計しやすいこと。
5. 実装を段階導入できること（既存剛体/地形系との共存）。

## Considered Options

### A. 現行 PBD/XPBD マルチレートの延長

- 長所:
  - 既存資産を活用しやすい。
  - 実時間で安定を得やすい。
- 短所:
  - 大きい有効 $\Delta t$ で外力と拘束の分離誤差が残りやすい。
  - 広域での滑らかな LoD と境界整合に追加ヒューリスティックが必要。

### B. 接触同時解法系（PD/ADMM, NSN, IPC など）への全面移行

- 長所:
  - 外力・拘束の同時扱いにより物理整合を上げやすい。
  - 接触品質を高く取りやすい。
- 短所:
  - 実装難度と計算コストが高い。
  - 多相・多物質・ゲーム向け広域 LoD を一気に満たすには重い。

### C. MLS-MPM を中核にしたハイブリッド

- 長所:
  - 連続体（流体/粉体系）を統一フレームで扱いやすい。
  - 相互作用をグリッド媒介でき、空間 LoD との整合が取りやすい。
  - 質量保存・運動量整合を設計しやすい。
- 短所:
  - 明示更新では CFL 制約があり、音速/剛性で $\Delta t$ が縛られる。
  - 接触・剛体連成・LoD 境界設計に実装負荷がある。

## Decision

**Option C（MLS-MPM 中核のハイブリッド）を採用する。**

理由:

1. 本件の本質は「広域連続性を保ったまま資源配分すること」であり、粒子直接相互作用よりグリッド媒介の方が空間 LoD と相性が良い。  
2. 低活動領域のコスト削減は、時間 LoD 単独ではなく空間 LoD 併用が必要。MLS-MPM はこの方針に適合する。  
3. 流体と粉体系を同一基盤で扱い、剛体/地形とは連成層で接続する構成が段階導入しやすい。  
4. 重力問題は「大きい有効 $\Delta t$ を単に許す」のではなく、CFL と LoD を管理して回避する方針が妥当。

## Final Architecture Policy

### 1) 基本時間積分

- 連続体（流体/粉体系）は MLS-MPM の明示系を基本とする。
- block/level ごとに subcycling を行い、時間刻みは
  $$
  \Delta t_b = \min\left(C_u \frac{h_b}{u_{\max,b}},\; C_c \frac{h_b}{c_b + u_{\max,b}},\; C_a \sqrt{\frac{h_b}{a_{\max,b}}}\right)
  $$
  に基づき決める。
- 許容範囲として、空間刻み $h_b$ に応じて時間刻みを連動させる（空間と時間の完全独立は初期要件にしない）。

### 2) LoD 方針

- 空間 LoD:
  - 遠方/大スケール施設周辺は粗いグリッドで計算。
  - 近傍/高活動領域は細かいグリッドで計算。
- 時間 LoD:
  - 低活動 block は更新頻度を下げる。
  - 高活動 block は細かいサイクルで更新。
- LoD 境界では、質量・運動量フラックス交換を明示して連続性を保つ。

### 3) 物質モデル

- 粒子は `material_id` を持つ。
- 物性は `material_id -> category -> parameter set` の参照で与える。
- 数十〜数百種のゲーム素材はカテゴリ共有で処理し、局所混相を許容する。

### 4) 資源保存性

- 連続体としての質量保存は MPM 側で担保。
- レア資源/アイテムは離散エンティティとして別管理し、必要時のみ MPM と双方向連成する。

### 5) 地形・剛体・弾性体連成

- 地形・剛体・弾性体は別ソルバを許容する。
- MPM 側とは接触/反力交換インターフェースで接続し、作用反作用を同時に記録する。
- 破壊・分裂は各ソルバ責務で行い、連成層で運動量整合を取る。

### 6) 保存量ポリシー

- 運動量保存を優先し、界面交換は必ず対称更新する。
- エネルギーは数値散逸を許容するが、散逸量を観測可能にし、必要に応じ温度等へ変換可能な設計にする。

## Consequences

### Positive

1. 広域ワールドでの計算資源配分（近傍高精度/遠方低コスト）がしやすくなる。
2. 流体/粉体系の統一が進み、モデル追加時の分岐負債が減る。
3. 将来の宇宙環境要件（運動量重視）に向けた基盤を作れる。

### Negative / Risks

1. LoD 境界・連成境界の実装難度が高い。
2. 明示 MLS-MPM では CFL 制約が支配的で、音速設定次第で時間 LoD 効果が薄れる。
3. デバッグ対象が「ソルバ本体 + LoD + 連成」に増える。

## Rejected Ideas (for now)

1. 現行 XPBD のみで大きい有効 $\Delta t$ を直接許容する方式  
   - 接触・外力分離誤差の管理コストが高く、広域連続性要求に対して拡張性が低い。
2. いきなり完全同時解法（全面陰解法）へ移行  
   - 初期導入コストが高く、段階実装と相反する。

## Migration Notes (high-level)

1. 第1段階: 流体/粉体系の最小 MLS-MPM 経路を導入し、既存剛体は別ソルバ連成。  
2. 第2段階: 空間 LoD（粗密グリッド）と時間 LoD（subcycling）を導入。  
3. 第3段階: 物質カテゴリ拡張、離散資源連成、保存量モニタリングを強化。  
4. 第4段階: 必要領域のみ部分陰解法/投影（圧力・接触）を追加。

## References

- XPBD: https://matthias-research.github.io/pages/publications/XPBD.pdf
- Small Steps in Physics Simulation: https://doi.org/10.1145/3309486.3340247
- Position Based Dynamics Survey: https://doi.org/10.2312/egt.20171034
- Unified Particle Physics / FleX: https://blog.mmacklin.com/project/flex/
- Projective Dynamics: https://www.projectivedynamics.org/Projective_Dynamics/index.html
- ADMM >= Projective Dynamics: https://doi.org/10.2312/sca.20161219
- Non-Smooth Newton methods: https://arxiv.org/abs/1907.04587
- IPC: https://ipc-sim.github.io/
- MPM Snow: https://alexey.stomakhin.com/research/snow.html
- MLS-MPM + rigid coupling: https://yuanming.taichi.graphics/publication/2018-mlsmpm/
