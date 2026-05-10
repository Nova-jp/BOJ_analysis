import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 実験: Exp-BFly-A-v2（Fly_Level + 全バタフライ水準追加）\n",
    "\n",
    "**目的**: B2〜B7 全ての水準を全行に追加し、かつ Fly_Level（自分自身の水準）も残した場合の効果を確認する。"
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
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
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
    "df_fly_baseline = pool_butterfly_data(df_rv)\n",
    "\n",
    "def pool_butterfly_data_exp_a_v2(feat_df: pd.DataFrame) -> pd.DataFrame:\n",
    "    df = feat_df.copy()\n",
    "    b_level_cols = []\n",
    "    for n in range(2, 8):\n",
    "        col_name = f'B{n}_level'\n",
    "        df[col_name] = df[f'B{n}']\n",
    "        b_level_cols.append(col_name)\n",
    "    fly_raw_cols = [f'B{n}' for n in range(2, 8)]\n",
    "    id_cols = [c for c in df.columns if c not in fly_raw_cols]\n",
    "    pooled = df.melt(id_vars=id_cols, value_vars=fly_raw_cols, var_name='Rate_Label', value_name='Rate_Value')\n",
    "    pooled['Meeting_Index'] = pooled['Rate_Label'].str.extract('(\\d+)').astype(int)\n",
    "    pooled = pooled.sort_values(['Rate_Label', 'Date']).reset_index(drop=True)\n",
    "    for h in [1, 3, 5]:\n",
    "        pooled[f'Target_{h}d'] = pooled.groupby('Rate_Label')['Rate_Value'].shift(-h) - pooled['Rate_Value']\n",
    "    for h in [3, 5]:\n",
    "        instr_std = pooled.groupby('Rate_Label')[f'Target_{h}d'].transform('std')\n",
    "        pooled[f'Target_{h}d_std']  = instr_std\n",
    "        pooled[f'Target_{h}d_norm'] = pooled[f'Target_{h}d'] / instr_std\n",
    "    \n",
    "    pooled['Fly_Level'] = pooled['Rate_Value']\n",
    "    \n",
    "    imputed_cols_fly = []\n",
    "    for n in range(2, 8):\n",
    "        for col in [f'M{n-1}_is_imputed', f'M{n}_is_imputed', f'M{n+1}_is_imputed']:\n",
    "            if col in pooled.columns and col not in imputed_cols_fly: imputed_cols_fly.append(col)\n",
    "\n",
    "    basic_cols   = ['Meeting_Index', 'is_post_mpm', 'Days_to_MPM', 'Actual_Policy_Rate']\n",
    "    anchor_cols  = ['M1_spread', 'M1_frac_diff', 'Slope_M1M8', 'Slope_M1M8_frac_diff']\n",
    "    fly_fd_cols  = [f'B{n}_frac_diff' for n in range(2, 8)]\n",
    "    ext_fd_cols  = ['USDJPY_frac_diff', 'JGB_Future_frac_diff', 'Nikkei225_frac_diff', 'DXY_frac_diff']\n",
    "    target_cols = ['Date', 'Target_1d_norm', 'Target_3d_norm', 'Target_5d_norm', 'Target_1d_std', 'Target_3d_std', 'Target_5d_std']\n",
    "    \n",
    "    final_cols = basic_cols + ['Fly_Level'] + b_level_cols + anchor_cols + fly_fd_cols + imputed_cols_fly + ext_fd_cols + target_cols\n",
    "    available = [c for c in final_cols if c in pooled.columns]\n",
    "    return pooled[available]\n",
    "\n",
    "df_fly_exp_v2 = pool_butterfly_data_exp_a_v2(df_rv)\n",
    "\n",
    "print('Running walk-forward...')\n",
    "res_3d_b, _, _, _, _ = walk_forward_with_model(df_fly_baseline, 'Target_3d_norm', START_DATE)\n",
    "ic3_b = summarize_ic(res_3d_b, instrument_indices=INSTRUMENT_INDICES)\n",
    "res_3d_v2, mdl_3d_v2, _, _, _ = walk_forward_with_model(df_fly_exp_v2, 'Target_3d_norm', START_DATE)\n",
    "ic3_v2 = summarize_ic(res_3d_v2, instrument_indices=INSTRUMENT_INDICES)\n",
    "\n",
    "print(f'\\nBaseline 3d Global IC: {ic3_b[\"ic_all\"]:.4f}')\n",
    "print(f'Exp-A-v2 3d Global IC: {ic3_v2[\"ic_all\"]:.4f}')\n",
    "\n",
    "imp = pd.DataFrame({'feature': mdl_3d_v2.feature_name(),\n",
    "                    'gain':    mdl_3d_v2.feature_importance(importance_type='gain')})\n",
    "print(\"\\nTop 20 Features (Exp-A-v2):\")\n",
    "print(imp.sort_values('gain', ascending=False).head(20).to_string(index=False))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": { "display_name": "Python 3", "language": "python", "name": "python3" },
  "language_info": { "name": "python", "version": "3.12.0" }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('notebooks/exp_bfly_a_v2.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
