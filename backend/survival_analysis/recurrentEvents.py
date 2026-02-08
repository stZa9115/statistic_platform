import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, NelsonAalenFitter
import io

# 假設這些是您的基礎類別與註冊裝飾器
from .sv_base import svTest
from . import sv_register

@sv_register
class RecurrentSurvivalAnalysis(svTest):
    name = "recurrent_survival"
    display_name = "重復事件分析 (Andersen-Gill)"
    result_prefix = "recurrent"

    def run(self, df: pd.DataFrame, group_col: str = 'TREAT'):
        # 1. 執行 Cox Andersen-Gill 模型
        covariates = ['TREAT', 'AGE'] 
        id_col = 'ID'
        start_col = 'TIME0'
        stop_col = 'TIME1'
        event_col = 'CENSOR'

        cph = CoxPHFitter()
        formula_str = " + ".join(covariates)
        
        cph.fit(df, 
                duration_col=stop_col, 
                entry_col=start_col, 
                event_col=event_col, 
                cluster_col=id_col, 
                formula=formula_str)

        # 2. 準備 Excel 用的 DataFrame 字典
        # Sheet 1: Coefficients
        coef_df = cph.summary.reset_index()
        # 強制將第一個欄位（原本的 Index）改名為 'variable'，避免名稱不符的問題
        coef_df = coef_df.rename(columns={coef_df.columns[0]: 'variable'})
        
        # 針對 p-value 格式化 (Excel 用，顯示部分會在前端處理)
        for index, row in coef_df.iterrows():
            if coef_df.at[index, 'p'] < 0.001:
                coef_df.at[index, 'p'] = 0.000  # 數值型態較好處理，或保持原樣

        # --- 新增：將係數表拆分為兩部分 (Part 1 & Part 2) 以便並排顯示 ---
        all_cols = list(coef_df.columns)
        # 移除 'variable' 以便重新分配，確保它出現在兩個表中
        metric_cols = [c for c in all_cols if c != 'variable']
        
        # 切分欄位
        mid_idx = (len(metric_cols) + 1) // 2
        cols1 = ['variable'] + metric_cols[:mid_idx+1]  # 前半部 (Coef, exp, se...)
        cols2 = ['variable'] + metric_cols[mid_idx+1:]  # 後半部 (z, p, CI...)
        
        coef_df1 = coef_df[cols1]
        coef_df2 = coef_df[cols2]
        
        # Sheet 2: Model Stats
        stats_data = {
            'Item': ['AIC (partial)', 'Log-likelihood', 'Concordance Index'],
            'Value': [round(cph.AIC_partial_, 3), round(cph.log_likelihood_, 3), round(cph.concordance_index_, 3)]
        }
        stats_df = pd.DataFrame(stats_data)
        
        # Sheet 3: Study Info
        info_data = {
            'Item': ['Observations', 'Events Observed', 'Unique Subjects'],
            'Value': [cph._n_examples, cph.event_observed.sum(), df[id_col].nunique()]
        }
        info_df = pd.DataFrame(info_data)

        # 3. 繪圖 (Nelson-Aalen)
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(10, 6))
        naf = NelsonAalenFitter()
        
        if group_col and group_col in df.columns:
            print(f"Plotting by groups in column: {group_col}")
            groups = sorted(df[group_col].unique())
            for g in groups:
                mask = (df[group_col] == g)
                naf.fit(df.loc[mask, stop_col], 
                        event_observed=df.loc[mask, event_col], 
                        entry=df.loc[mask, start_col], 
                        label=f"{group_col}={g}")
                naf.plot_cumulative_hazard(ax=ax, ci_show=True)
        else:
            naf.fit(df[stop_col], df[event_col], entry=df[start_col], label='Overall')
            naf.plot_cumulative_hazard(ax=ax, ci_show=True)

        ax.set_title(f"Cumulative Hazard of Recurrent Events")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Hazard (Mean Events per Person)")
        ax.grid(True, linestyle='--', alpha=0.5)

        print("Saving figure to test.png")
        plt.savefig('test.png')

        # 4. 封裝回傳格式
        return {
            "fig": fig,
            "cox_df": coef_df,       # 完整版供下載
            "stats_df": stats_df,
            "info_df": info_df,
            "sections": [
                # 第一排：基本資訊與模型統計 (維持左右並排)
                {
                    "title": "Study Population Information",
                    "columns": ["Item", "Value"],
                    "data": info_df.to_dict(orient='records'),
                    "layout": "half"
                },
                {
                    "title": "Model Statistics",
                    "columns": ["Item", "Value"],
                    "data": stats_df.to_dict(orient='records'),
                    "layout": "half"
                },
                # 第二部分：係數表 Part 1 (改為 Full，強制佔滿整行)
                {
                    "title": "Model Coefficients (Part 1)",
                    "columns": cols1,
                    "data": coef_df1.fillna("").to_dict(orient='records'),
                    "layout": "full"  # <--- 修改這裡：由 'half' 改為 'full'
                },
                # 第三部分：係數表 Part 2 (改為 Full，強制佔滿整行)
                {
                    "title": "Model Coefficients (Part 2)",
                    "columns": cols2,
                    "data": coef_df2.fillna("").to_dict(orient='records'),
                    "layout": "full"  # <--- 修改這裡：由 'half' 改為 'full'
                }
            ]
        }