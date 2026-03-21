# num_boost_round 感度分析 設計書

Gemini CLI への実装指示書。
設計方針・パイプライン全体は `designs/current_model.md` を先に読むこと。

---

## 目的

現在 `num_boost_round=300` を固定値として使っている。
これが適切かどうかを確認するため、ラウンド数を変化させたときの OOS IC の推移を測定する。

結果に応じて以下を判断する：
- **300でプラトー済み** → 現行設定を維持して特徴量実験へ進む
- **500〜700でまだ改善** → underfitting の可能性 → rounds を増やして固定する
- **200でピーク後に悪化** → overfitting の可能性 → 正則化を強化する

---

## 実装手順

### Step 1：ノートブック作成

`notebooks/rounds_sensitivity.ipynb` を新規作成する（`.ipynb` は Python `json` ライブラリ経由で生成）。

### Step 2：標準パイプラインの実行（セル1）

```python
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.processing import load_and_clean_data
from src.features import generate_features
from src.pooling import pool_boj_data
from src.modeling import walk_forward_validation

EXCEL_PATH   = '../data/BOJ_data.xlsx'
MEETING_PATH = '../data/BOJ_meeting_history.csv'
START_DATE   = '2024-01-01'

df_raw    = load_and_clean_data(EXCEL_PATH, MEETING_PATH)
df_feat   = generate_features(df_raw)
df_pooled = pool_boj_data(df_feat)

print(f'データ準備完了: {len(df_pooled):,} 行')
```

### Step 3：感度分析の実行（セル2）

```python
ROUNDS_LIST = [100, 200, 300, 500, 700, 1000]

records = []

for rounds in ROUNDS_LIST:
    print(f'rounds={rounds} を実行中...')

    res_3d = walk_forward_validation(
        df_pooled, 'Target_3d_norm', START_DATE, num_boost_round=rounds
    )
    res_5d = walk_forward_validation(
        df_pooled, 'Target_5d_norm', START_DATE, num_boost_round=rounds
    )

    # OOS IC（全体）
    ic_3d, _ = spearmanr(res_3d['Actual'], res_3d['Pred'])
    ic_5d, _ = spearmanr(res_5d['Actual'], res_5d['Pred'])

    # Train IC（過学習の確認用）
    train_ic_3d = res_3d['Train_IC'].mean()
    train_ic_5d = res_5d['Train_IC'].mean()

    records.append({
        'rounds':       rounds,
        'OOS_IC_3d':    round(ic_3d, 4),
        'OOS_IC_5d':    round(ic_5d, 4),
        'Train_IC_3d':  round(train_ic_3d, 4),
        'Train_IC_5d':  round(train_ic_5d, 4),
        'Gap_3d':       round(train_ic_3d - ic_3d, 4),  # 過学習ギャップ
        'Gap_5d':       round(train_ic_5d - ic_5d, 4),
    })

df_results = pd.DataFrame(records)
print(df_results.to_string(index=False))
```

### Step 4：可視化（セル3）

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, oos_col, train_col, gap_col, label in [
    (axes[0], 'OOS_IC_3d', 'Train_IC_3d', 'Gap_3d', '3d モデル'),
    (axes[1], 'OOS_IC_5d', 'Train_IC_5d', 'Gap_5d', '5d モデル'),
]:
    ax.plot(df_results['rounds'], df_results[oos_col],  'o-', label='OOS IC', color='steelblue', linewidth=2)
    ax.plot(df_results['rounds'], df_results[train_col], 's--', label='Train IC', color='tomato', linewidth=1.5, alpha=0.7)
    ax.axvline(300, color='gray', linestyle=':', linewidth=1, label='現行 (300)')
    ax.set_title(f'{label}: rounds vs IC')
    ax.set_xlabel('num_boost_round')
    ax.set_ylabel('IC (Spearman)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../designs/pending/rounds_sensitivity_result.png', dpi=120, bbox_inches='tight')
plt.show()
```

### Step 5：判定と結論（セル4）

```python
# 現行(300) と各 rounds の OOS IC 差分を表示
baseline_3d = df_results.loc[df_results['rounds'] == 300, 'OOS_IC_3d'].values[0]
baseline_5d = df_results.loc[df_results['rounds'] == 300, 'OOS_IC_5d'].values[0]

df_results['Delta_IC_3d'] = (df_results['OOS_IC_3d'] - baseline_3d).round(4)
df_results['Delta_IC_5d'] = (df_results['OOS_IC_5d'] - baseline_5d).round(4)

print('=== 現行(300) との OOS IC 差分 ===')
print(df_results[['rounds', 'OOS_IC_3d', 'Delta_IC_3d', 'OOS_IC_5d', 'Delta_IC_5d', 'Gap_3d', 'Gap_5d']].to_string(index=False))

# 推奨 rounds を自動判定（OOS IC が最大の rounds）
best_rounds_3d = df_results.loc[df_results['OOS_IC_3d'].idxmax(), 'rounds']
best_rounds_5d = df_results.loc[df_results['OOS_IC_5d'].idxmax(), 'rounds']
print(f'\n推奨 rounds: 3d={best_rounds_3d}, 5d={best_rounds_5d}')
print('（3d と 5d で異なる場合は大きい方、または両方のICが高い中間値を検討）')
```

---

## 報告してほしい内容

実験完了後に以下を報告すること：

1. `df_results` の全行を表形式で（rounds, OOS_IC_3d, OOS_IC_5d, Train_IC_3d, Train_IC_5d, Gap_3d, Gap_5d）
2. OOS IC が最大になる rounds（3d・5d それぞれ）
3. Train IC と OOS IC の Gap（過学習の度合い）の傾向コメント
4. グラフ画像（`rounds_sensitivity_result.png` として保存済みのパス）

---

## 注意事項

- このノートブックは感度分析専用。`main_model.ipynb` は変更しないこと
- `num_boost_round` 以外のパラメータはメインモデル設定のまま変えないこと
- 計算時間目安：rounds × フォールド数 × 2モデル（3d/5d）= 約6×4×2=48 回の LightGBM 訓練
- 実行時間が長い場合は ROUNDS_LIST を `[100, 300, 500, 1000]` の4点に絞ってよい
