import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 実験: Exp-BFly-A（全バタフライ水準追加）\n",
    "\n",
    "**目的**: B2〜B7 全ての水準を全行に特徴量として追加した場合に CS IC が改善するか確認する。\n",
    "\n",
    "## ベースライン IC（目標）\n",
    "| Horizon | Global IC | CS IC | Train IC | Gap |\n",
    "|---------|-----------|-------|----------|-----|\n",
    "| 3d | 0.373 | **0.304** | 0.521 | 0.148 |\n",
    "| 5d | 0.418 | **0.346** | 0.573 | 0.154 |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '..')\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.stats import spearmanr\n",
    "\n",
    "from src.processing        import load_and_clean_data\n",
    "from src.features_rv       import generate_rv_features\n",
    "from src.pooling_butterfly import pool_butterfly_data\n",
    "from src.modeling          import walk_forward_with_model, summarize_ic\n",
    "\n",
    "START_DATE = '2024-01-01'\n",
    "INSTRUMENT_INDICES = set(range(2, 8))\n",
    "\n",
    "print('Loading data...')\n",
    "df_raw = load_and_clean_data('../data/BOJ_data.xlsx', '../data/BOJ_meeting_history.csv')\n",
    "df_rv  = generate_rv_features(df_raw)\n",
    "\n",
    "# 1. Baseline Data\n",
    "df_fly_baseline = pool_butterfly_data(df_rv)\n",
    "\n",
    "print(f'Data: {df_raw[\"Date\"].min().date()} → {df_raw[\"Date\"].max().date()}')\n",
    "print(f'Baseline shape: {df_fly_baseline.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. ベースライン IC の再現"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Running Baseline Walk-forward...')\n",
    "res_3d_b, _, _, _, _ = walk_forward_with_model(df_fly_baseline, 'Target_3d_norm', START_DATE)\n",
    "res_5d_b, _, _, _, _ = walk_forward_with_model(df_fly_baseline, 'Target_5d_norm', START_DATE)\n",
    "\n",
    "ic3_b = summarize_ic(res_3d_b, instrument_indices=INSTRUMENT_INDICES)\n",
    "ic5_b = summarize_ic(res_5d_b, instrument_indices=INSTRUMENT_INDICES)\n",
    "\n",
    "print('\\n=== Baseline IC (RV Butterfly) ===')\n",
    "print(f'               3d        5d')\n",
    "print(f'Global IC  : {ic3_b[\"ic_all\"]:>8.4f}  {ic5_b[\"ic_all\"]:>8.4f}')\n",
    "print(f'CS IC      : {ic3_b[\"cs_ic\"]:>8.4f}  {ic5_b[\"cs_ic\"]:>8.4f}')\n",
    "print(f'Train IC   : {ic3_b[\"train_ic\"]:>8.4f}  {ic5_b[\"train_ic\"]:>8.4f}')\n",
    "print(f'Gap        : {ic3_b[\"gap\"]:>8.4f}  {ic5_b[\"gap\"]:>8.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Exp-BFly-A (全B水準追加) の実装\n",
    "\n",
    "ノートブック内でインラインプーリングを実装し、B2〜B7 level を全行に持たせる。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pool_butterfly_data_exp_a(feat_df: pd.DataFrame) -> pd.DataFrame:\n",
    "    df = feat_df.copy()\n",
    "    \n",
    "    # B2_level〜B7_level を全行に引き継ぐため、melt 前に列として追加\n",
    "    b_level_cols = []\n",
    "    for n in range(2, 8):\n",
    "        col_name = f'B{n}_level'\n",
    "        df[col_name] = df[f'B{n}']\n",
    "        b_level_cols.append(col_name)\n",
    "        \n",
    "    fly_raw_cols = [f'B{n}' for n in range(2, 8)]\n",
    "    id_cols = [c for c in df.columns if c not in fly_raw_cols]\n",
    "    \n",
    "    pooled = df.melt(\n",
    "        id_vars=id_cols,\n",
    "        value_vars=fly_raw_cols,\n",
    "        var_name='Rate_Label',\n",
    "        value_name='Rate_Value',\n",
    "    )\n",
    "    \n",
    "    pooled['Meeting_Index'] = pooled['Rate_Label'].str.extract('(\\d+)').astype(int)\n",
    "    pooled = pooled.sort_values(['Rate_Label', 'Date']).reset_index(drop=True)\n",
    "\n",
    "    # 目的変数の生成\n",
    "    for h in [1, 3, 5]:\n",
    "        pooled[f'Target_{h}d'] = (\n",
    "            pooled.groupby('Rate_Label')['Rate_Value'].shift(-h) - pooled['Rate_Value']\n",
    "        )\n",
    "    for h in [3, 5]:\n",
    "        instr_std = pooled.groupby('Rate_Label')[f'Target_{h}d'].transform('std')\n",
    "        pooled[f'Target_{h}d_std']  = instr_std\n",
    "        pooled[f'Target_{h}d_norm'] = pooled[f'Target_{h}d'] / instr_std\n",
    "    \n",
    "    # Feature selection (matching src/pooling_butterfly.py names)\n",
    "    imputed_cols_fly = []\n",
    "    for n in range(2, 8):\n",
    "        for col in [f'M{n-1}_is_imputed', f'M{n}_is_imputed', f'M{n+1}_is_imputed']:\n",
    "            if col in pooled.columns and col not in imputed_cols_fly:\n",
    "                imputed_cols_fly.append(col)\n",
    "\n",
    "    basic_cols   = ['Meeting_Index', 'is_post_mpm', 'Days_to_MPM', 'Actual_Policy_Rate']\n",
    "    anchor_cols  = ['M1_spread', 'M1_frac_diff', 'Slope_M1M8', 'Slope_M1M8_frac_diff']\n",
    "    fly_fd_cols  = [f'B{n}_frac_diff' for n in range(2, 8)]\n",
    "    ext_fd_cols  = ['USDJPY_frac_diff', 'JGB_Future_frac_diff', 'Nikkei225_frac_diff', 'DXY_frac_diff']\n",
    "    \n",
    "    target_cols = ['Date', 'Target_1d_norm', 'Target_3d_norm', 'Target_5d_norm', \n",
    "                   'Target_1d_std', 'Target_3d_std', 'Target_5d_std']\n",
    "    \n",
    "    final_cols = (\n",
    "        basic_cols\n",
    "        + b_level_cols\n",
    "        + anchor_cols\n",
    "        + fly_fd_cols\n",
    "        + imputed_cols_fly\n",
    "        + ext_fd_cols\n",
    "        + target_cols\n",
    "    )\n",
    "    \n",
    "    available = [c for c in final_cols if c in pooled.columns]\n",
    "    return pooled[available]\n",
    "\n",
    "df_fly_exp_a = pool_butterfly_data_exp_a(df_rv)\n",
    "print(f'Exp-A shape: {df_fly_exp_a.shape}')\n",
    "print(f'Features: {len([c for c in df_fly_exp_a.columns if c not in [\"Date\", \"Target_1d_norm\", \"Target_3d_norm\", \"Target_5d_norm\", \"Target_1d_std\", \"Target_3d_std\", \"Target_5d_std\", \"is_post_mpm\"]])}')\n",
    "print(f'Feature names: {[c for c in df_fly_exp_a.columns if c not in [\"Date\", \"Target_1d_norm\", \"Target_3d_norm\", \"Target_5d_norm\", \"Target_1d_std\", \"Target_3d_std\", \"Target_5d_std\", \"is_post_mpm\"]]}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Exp-BFly-A の実行"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Running Exp-BFly-A Walk-forward...')\n",
    "res_3d_a, mdl_3d_a, _, _, _ = walk_forward_with_model(df_fly_exp_a, 'Target_3d_norm', START_DATE)\n",
    "res_5d_a, mdl_5d_a, _, _, _ = walk_forward_with_model(df_fly_exp_a, 'Target_5d_norm', START_DATE)\n",
    "\n",
    "ic3_a = summarize_ic(res_3d_a, instrument_indices=INSTRUMENT_INDICES)\n",
    "ic5_a = summarize_ic(res_5d_a, instrument_indices=INSTRUMENT_INDICES)\n",
    "\n",
    "print('\\n=== Exp-BFly-A IC (All B Levels) ===')\n",
    "print(f'               3d        5d')\n",
    "print(f'Global IC  : {ic3_a[\"ic_all\"]:>8.4f}  {ic5_a[\"ic_all\"]:>8.4f}')\n",
    "print(f'CS IC      : {ic3_a[\"cs_ic\"]:>8.4f}  {ic5_a[\"cs_ic\"]:>8.4f}')\n",
    "print(f'Train IC   : {ic3_a[\"train_ic\"]:>8.4f}  {ic5_a[\"train_ic\"]:>8.4f}')\n",
    "print(f'Gap        : {ic3_a[\"gap\"]:>8.4f}  {ic5_a[\"gap\"]:>8.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. 比較と可視化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df = pd.DataFrame({\n",
    "    'Metric': ['3d Global IC', '3d CS IC', '3d Train IC', '3d Gap', \n",
    "               '5d Global IC', '5d CS IC', '5d Train IC', '5d Gap'],\n",
    "    'Baseline': [ic3_b['ic_all'], ic3_b['cs_ic'], ic3_b['train_ic'], ic3_b['gap'],\n",
    "                 ic5_b['ic_all'], ic5_b['cs_ic'], ic5_b['train_ic'], ic5_b['gap']],\n",
    "    'Exp-BFly-A': [ic3_a['ic_all'], ic3_a['cs_ic'], ic3_a['train_ic'], ic3_a['gap'],\n",
    "                   ic5_a['ic_all'], ic5_a['cs_ic'], ic5_a['train_ic'], ic5_a['gap']]\n",
    "})\n",
    "results_df['Delta'] = results_df['Exp-BFly-A'] - results_df['Baseline']\n",
    "print(results_df.to_string(index=False))\n",
    "\n",
    "# 可視化\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "fig.suptitle('Butterfly Global IC by Fold: Baseline vs Exp-BFly-A', fontsize=12, fontweight='bold')\n",
    "\n",
    "for i, (ic_b, ic_a, title) in enumerate([(ic3_b, ic3_a, '3d'), (ic5_b, ic5_a, '5d')]):\n",
    "    ax = axes[i]\n",
    "    folds = sorted(ic_b['ic_by_fold'].keys())\n",
    "    b_vals = [ic_b['ic_by_fold'][f] for f in folds]\n",
    "    a_vals = [ic_a['ic_by_fold'][f] for f in folds]\n",
    "    \n",
    "    x = np.arange(len(folds))\n",
    "    width = 0.35\n",
    "    \n",
    "    ax.bar(x - width/2, b_vals, width, label='Baseline', color='#cccccc')\n",
    "    ax.bar(x + width/2, a_vals, width, label='Exp-BFly-A', color='#ff7f0e' if i==0 else '#d62728')\n",
    "    \n",
    "    ax.set_title(f'{title} Horizon')\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels([str(f) for f in folds])\n",
    "    ax.legend()\n",
    "    ax.grid(True, axis='y', alpha=0.3)\n",
    "    ax.axhline(0, color='black', lw=0.8)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('exp_bfly_a_results.png')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. 特徴量重要度の確認"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
    "fig.suptitle('Feature Importance (gain, last fold, top 20) - Exp-BFly-A', fontsize=12, fontweight='bold')\n",
    "\n",
    "for ax, model, title, color in [\n",
    "        (axes[0], mdl_3d_a, 'Butterfly 3d', '#ff7f0e'),\n",
    "        (axes[1], mdl_5d_a, 'Butterfly 5d', '#d62728')]:\n",
    "    imp = pd.DataFrame({'feature': model.feature_name(),\n",
    "                        'gain':    model.feature_importance(importance_type='gain')})\n",
    "    imp = imp.sort_values('gain', ascending=False).head(20)\n",
    "    ax.barh(imp['feature'][::-1], imp['gain'][::-1], color=color, edgecolor='white')\n",
    "    ax.set_title(title, fontsize=11, fontweight='bold')\n",
    "    ax.set_xlabel('Gain'); ax.tick_params(labelsize=8); ax.grid(True, axis='x', alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('notebooks/exp_bfly_a_all_b_levels.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
