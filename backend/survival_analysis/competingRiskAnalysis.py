import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, AalenJohansenFitter

# 假設這些是您的基礎類別與註冊裝飾器 (保持與 recurrentEvents.py 一致)
from .sv_base import svTest
from . import sv_register

@sv_register
class CompetingRisksAnalysis(svTest):
    name = "competing_risks"
    display_name = "競爭風險分析 (CIF & Cause-Specific Cox)"
    result_prefix = "competing"

    def run(self, df: pd.DataFrame, group_col: str = 'TREAT'):
        """
        執行競爭風險分析：
        1. 統計模型：使用 Cause-Specific Cox Model 計算 HR 與 P-value。
        2. 繪圖：使用 Aalen-Johansen Estimator 繪製 Cumulative Incidence Function (CIF)。
        """
        
        # --- 0. 參數設定 ---
        # 根據您的資料結構，這裡需要指定欄位名稱
        # 若需要動態傳入 event_of_interest，可考慮透過 UI 參數或固定邏輯
        time_col = 'TIME'       # 存活時間
        event_col = 'EVENT'     # 事件狀態 (0=Censored, 1=Event of Interest, 2=Competing Risk)
        event_of_interest = 1   # 主要關注的事件代碼

        time_col = 'ptime'       # 存活時間
        event_col = 'pstat'     # 事件狀態 (0=Censored, 1=Event of Interest, 2=Competing Risk)

        
        # --- 1. 統計分析 (Cause-Specific Cox Model) ---
        # 原理：將競爭事件 (Competing Risk) 視為 Censored (0)，主要事件視為 1
        
        cph = CoxPHFitter()
        
        # 準備 Cox 專用資料
        df_cox = df[[time_col, event_col]].copy()
        if group_col and group_col in df.columns:
            df_cox[group_col] = df[group_col]
            formula_str = group_col  # 簡單單變量分析，也可改為 "group + age + ..."
        else:
            # 若無分組，僅做整體分析 (雖然 Cox 主要用於比較，但為避免報錯仍需執行)
            df_cox['dummy'] = 1 
            formula_str = 'dummy'

        # 轉換事件狀態：非主要事件全部視為 Censored (0)
        df_cox['event_binary'] = (df_cox[event_col] == event_of_interest).astype(int)
        
        # 擬合模型
        cph.fit(df_cox, 
                duration_col=time_col, 
                event_col='event_binary', 
                formula=formula_str)

        # --- 2. 準備 Excel/前端 顯示用的 DataFrame ---
        
        # Sheet 1: Coefficients (主要結果)
        coef_df = cph.summary.reset_index().rename(columns={'index': 'variable'})
        
        # Sheet 2: Model Stats (模型適配度)
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
        
        # Sheet 3: Study Info (基本資訊)
        info_data = {
            'Item': ['Observations', 'Events of Interest Observed', 'Unique Subjects'],
            'Value': [cph._n_examples, cph.event_observed.sum(), df.shape[0]] # 假設每列為一受試者
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
                # AJF 自動處理競爭風險，不需要手動轉 binary
                ajf.fit(df.loc[mask, time_col], 
                        df.loc[mask, event_col], 
                        event_of_interest=event_of_interest)
                ajf.plot(ax=ax, label=f"{group_col}={g}")
            
            # 在圖上標註 P-value (來自 Cox 模型)
            if 'p' in coef_df.columns and not coef_df.empty:
                p_val = coef_df['p'].iloc[0] # 取第一個變數的 P 值
                ax.text(0.05, 0.95, f"Cause-Specific P: {p_val:.4f}", 
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
            "cox_df": coef_df,        # 對應 router 儲存邏輯
            "stats_df": stats_df,
            "info_df": info_df,
            "report_text": f"Competing Risks Analysis (Event {event_of_interest}) complete.",
            "sections": [
                {
                    "title": "Cause-Specific Cox Coefficients",
                    "columns": list(coef_df.columns),
                    "data": coef_df.fillna("").to_dict(orient='records')
                },
                {
                    "title": "Model Statistics",
                    "columns": list(stats_df.columns),
                    "data": stats_df.to_dict(orient='records')
                },
                {
                    "title": "Study Population Information",
                    "columns": list(info_df.columns),
                    "data": info_df.to_dict(orient='records')
                }
            ]
        }