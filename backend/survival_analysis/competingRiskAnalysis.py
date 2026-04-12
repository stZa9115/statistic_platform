import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, AalenJohansenFitter
from .sv_base import svTest
from . import sv_register

@sv_register
class CompetingRisksAnalysis(svTest):
    name = "competing_risks"
    display_name = "競爭風險分析 (CIF & Cause-Specific Cox)"
    result_prefix = "competing"

    def run(self, df: pd.DataFrame, group_col: str = 'TREAT'):
        # --- 0. 參數設定 ---
        time_col = 'ptime'       
        event_col = 'pstat'     
        event_of_interest = 1   

        # --- 1. 統計分析 (Cause-Specific Cox Model) ---
        cph = CoxPHFitter()
        
        df_cox = df[[time_col, event_col]].copy()
        if group_col and group_col in df.columns:
            df_cox[group_col] = df[group_col]
            formula_str = group_col
        else:
            df_cox['dummy'] = 1 
            formula_str = 'dummy'

        df_cox['event_binary'] = (df_cox[event_col] == event_of_interest).astype(int)
        
        cph.fit(df_cox, 
                duration_col=time_col, 
                event_col='event_binary', 
                formula=formula_str)

        # --- 2. 準備數據表 ---
        
        # (A) Coefficients 表格處理 (拆分邏輯) --- 修正處 ---
        coef_df = cph.summary.reset_index()
        # 強制將第一欄 (原本的 Index) 改名為 'variable'
        coef_df = coef_df.rename(columns={coef_df.columns[0]: 'variable'})
        
        # P-value 格式化
        if 'p' in coef_df.columns:
            for index, row in coef_df.iterrows():
                if coef_df.at[index, 'p'] < 0.001:
                    coef_df.at[index, 'p'] = 0.000

        # 拆分欄位 (Part 1 & Part 2)
        all_cols = list(coef_df.columns)
        # 排除 'variable' 欄位後計算切分點
        metric_cols = [c for c in all_cols if c != 'variable']
        mid_idx = (len(metric_cols) + 1) // 2
        
        cols1 = ['variable'] + metric_cols[:mid_idx+1]
        cols2 = ['variable'] + metric_cols[mid_idx+1:]
        
        # 這裡可能會遇到變數少導致 cols2 只有 variable 的情況，但通常不影響顯示
        coef_df1 = coef_df[cols1]
        coef_df2 = coef_df[cols2]

        # (B) Model Stats
        stats_data = {
            'Statistic': ['AIC (partial)', 'Log-likelihood', 'Concordance Index', 'Event of Interest'],
            'Value': [
                round(cph.AIC_partial_, 3), 
                round(cph.log_likelihood_, 3), 
                round(cph.concordance_index_, 3),
                str(event_of_interest)
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        
        # (C) Study Info
        info_data = {
            'Item': ['Observations', 'Events of Interest', 'Unique Subjects'],
            'Value': [cph._n_examples, cph.event_observed.sum(), df.shape[0]]
        }
        info_df = pd.DataFrame(info_data)

        # --- 3. 繪圖 (Aalen-Johansen CIF) ---
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(10, 6))
        ajf = AalenJohansenFitter(calculate_variance=True)
        
        if group_col and group_col in df.columns:
            groups = sorted(df[group_col].unique())
            for g in groups:
                mask = (df[group_col] == g)
                ajf.fit(df.loc[mask, time_col], 
                        df.loc[mask, event_col], 
                        event_of_interest=event_of_interest)
                ajf.plot(ax=ax, label=f"{group_col}={g}")
            
            if 'p' in coef_df.columns and not coef_df.empty:
                p_val = coef_df['p'].iloc[0]
                ax.text(0.05, 0.95, f"CS-Cox P: {p_val:.4f}", 
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ajf.fit(df[time_col], df[event_col], event_of_interest=event_of_interest)
            ajf.plot(ax=ax, label='Overall Population')

        ax.set_title(f"Cumulative Incidence Function (Event {event_of_interest})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability of Event")
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- 4. 封裝回傳格式 ---
        return {
            "fig": fig,
            "cox_df": coef_df,
            "stats_df": stats_df,
            "info_df": info_df,
            "report_text": f"Competing Risks Analysis (Event {event_of_interest}) complete.",
            "sections": [
                {
                    "title": "Study Population Information",
                    "columns": list(info_df.columns),
                    "data": info_df.to_dict(orient='records'),
                    "layout": "half"
                },
                {
                    "title": "Model Statistics",
                    "columns": list(stats_df.columns),
                    "data": stats_df.to_dict(orient='records'),
                    "layout": "half"
                },
                {
                    "title": "Cause-Specific Cox Coefficients",
                    "layout": "full",
                    "tables": [
                        {
                            "sub_title": "Part 1: Coefficients & SE",
                            "columns": cols1,
                            "data": coef_df1.fillna("").to_dict(orient='records')
                        },
                        {
                            "sub_title": "Part 2: Significance & CI",
                            "columns": cols2,
                            "data": coef_df2.fillna("").to_dict(orient='records')
                        }
                    ]
                }
            ]
        }