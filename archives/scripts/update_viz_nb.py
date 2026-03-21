import json
import pandas as pd

with open('notebooks/06_prediction_visualization.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 5: Figure 1 (All period spread) -> Last 1 Year
cell5 = nb['cells'][5]
source5 = cell5['source']
# Modify Figure 1 to last 1 year
new_source5 = []
for line in source5:
    if "ax.set_title" in line:
        new_source5.append("    # 直近1年に制限\n")
        new_source5.append("    end_date = df_feat['Date'].max()\n")
        new_source5.append("    one_year_ago = end_date - pd.DateOffset(years=1)\n")
        new_source5.append("    ax.set_xlim(one_year_ago, end_date)\n")
        new_source5.append("    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))\n")
        new_source5.append("    ax.set_title('BOJ OIS Spread M1-M8 (Last 1 Year)')\n")
    else:
        new_source5.append(line)
cell5['source'] = new_source5

# Cell 7: plot_level_grid function and its calls
cell7 = nb['cells'][7]
source7 = cell7['source']
new_source7 = []
skip = False
for line in source7:
    if "for idx, mi in enumerate(mi_list):" in line:
        new_source7.append(line)
        new_source7.append("        ax = axes[idx]\n")
        new_source7.append("        spread_col = f'M{mi}_spread'\n\n")
        new_source7.append("        # 直近1年に制限\n")
        new_source7.append("        end_date = actual_spread['Date'].max()\n")
        new_source7.append("        one_year_ago = end_date - pd.DateOffset(years=1)\n")
        new_source7.append("        \n")
        new_source7.append("        actual_sub = actual_spread[actual_spread['Date'] >= one_year_ago]\n")
        new_source7.append("        ax.plot(actual_sub['Date'], actual_sub[spread_col],\n")
        new_source7.append("                color='black', linewidth=1.0, alpha=0.85, label='Actual Level', zorder=3)\n\n")
        new_source7.append("        pred_df = build_predicted_level(results[h], future_preds[h], mi, h)\n")
        new_source7.append("        pred_df = pred_df[pred_df['Plot_Date'] >= one_year_ago]\n")
        new_source7.append("        \n")
        new_source7.append("        hist_pred = pred_df[~pred_df['Is_Future']].dropna(subset=['Pred_Level'])\n")
        new_source7.append("        fut_pred  = pred_df[pred_df['Is_Future']].dropna(subset=['Pred_Level'])\n\n")
        new_source7.append("        ax.plot(hist_pred['Plot_Date'], hist_pred['Pred_Level'],\n")
        new_source7.append("                color='darkorange', linewidth=1.1, alpha=0.85, label='Predicted Level', zorder=4)\n\n")
        new_source7.append("        if len(fut_pred) > 0:\n")
        new_source7.append("            if len(hist_pred) > 0:\n")
        new_source7.append("                connect = hist_pred.iloc[[-1]].copy()\n")
        new_source7.append("                fut_pred = pd.concat([connect, fut_pred], ignore_index=True)\n")
        new_source7.append("            ax.plot(fut_pred['Plot_Date'], fut_pred['Pred_Level'],\n")
        new_source7.append("                    color='red', linewidth=1.3, linestyle='--', alpha=0.9,\n")
        new_source7.append("                    label='Future Forecast', zorder=5)\n\n")
        new_source7.append("        # x軸の範囲を固定\n")
        new_source7.append("        xmin = one_year_ago\n")
        new_source7.append("        xmax = end_date + pd.Timedelta(days=max(h, 10))\n")
        new_source7.append("        ax.set_xlim(xmin, xmax)\n")
        new_source7.append("        \n")
        new_source7.append("        # MPM縦線を追加\n")
        new_source7.append("        add_vlines(ax, meeting_dates, xmin, xmax)\n")
        skip = True
    elif skip and "sub_res = results[h]" in line:
        new_source7.append(line)
        skip = False
    elif not skip:
        if "fig.suptitle" in line:
            new_source7.append("    fig.suptitle(f'Predicted vs Actual Spread Level (Last 1 Year) [{h}d]{title_suffix}', fontsize=11)\n")
        elif "mdates.MonthLocator(interval=2)" in line:
            new_source7.append("        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))\n")
        else:
            new_source7.append(line)
cell7['source'] = new_source7

# Cell 9: Figure 4 (Prediction Error) -> Last 1 Year
cell9 = nb['cells'][9]
source9 = cell9['source']
new_source9 = []
skip_err = False
for line in source9:
    if "merged = hist_pred.merge(" in line:
        new_source9.append(line)
        new_source9.append("            actual_spread[['Date', spread_col]].rename(columns={'Date': 'Plot_Date', spread_col: 'Actual_Level'}),\n")
        new_source9.append("            on='Plot_Date', how='inner'\n")
        new_source9.append("        )\n")
        new_source9.append("        # 直近1年に制限\n")
        new_source9.append("        end_date = actual_spread['Date'].max()\n")
        new_source9.append("        one_year_ago = end_date - pd.DateOffset(years=1)\n")
        new_source9.append("        merged = merged[merged['Plot_Date'] >= one_year_ago]\n")
        skip_err = True
    elif skip_err and "merged['Error'] =" in line:
        new_source9.append(line)
        skip_err = False
    elif not skip_err:
        if "ax.bar(" in line:
            new_source9.append("        ax.bar(merged['Plot_Date'], merged['Error'], color=colors_err, width=1.0, alpha=0.7)\n")
        elif "add_vlines(ax, meeting_dates" in line:
            new_source9.append("        ax.set_xlim(one_year_ago, end_date)\n")
            new_source9.append("        add_vlines(ax, meeting_dates, one_year_ago, end_date)\n")
        elif "mdates.MonthLocator(interval=3)" in line:
            new_source9.append("        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))\n")
        elif "fig.suptitle" in line:
            new_source9.append("    fig.suptitle(f'Prediction Error (Pred − Actual Level) - Last 1 Year [{h}d]', fontsize=11)\n")
        else:
            new_source9.append(line)
cell9['source'] = new_source9

with open('notebooks/06_prediction_visualization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
