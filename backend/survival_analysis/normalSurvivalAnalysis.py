import pandas as pd
import numpy as np
import io
import re
import contextlib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test, multivariate_logrank_test

from .sv_base import svTest
from . import sv_register

def _clean_report_text(full_text):
    """
    清洗 check_assumptions 的輸出 (保持不變)
    """
    table_start_marker = "<lifelines.StatisticalResult"
    advice_pattern = re.compile(r"(\n\d+\. Variable|\nNo variables)")
    
    start_idx = full_text.find(table_start_marker)
    match = advice_pattern.search(full_text)
    
    if start_idx == -1 or not match:
        return full_text
    
    end_idx = match.start()
    
    intro_part = full_text[:start_idx].strip()
    advice_part = full_text[end_idx:].strip()
    
    cleaned_text = (
        f"{intro_part}\n\n"
        f"--- [Statistical Table removed. Please refer to the CSV/DataFrame] ---\n\n"
        f"{advice_part}"
    )
    return cleaned_text

def analyze_survival_logic(df: pd.DataFrame, group_col: str = None):
    """
    執行生存分析邏輯 (支援分組比較)
    """
    if "time" not in df.columns or "status" not in df.columns:
        raise ValueError("DataFrame 必須包含 'time' (時間) 與 'status' (事件狀態 0/1) 欄位")

    # 檢查分組欄位
    if group_col and group_col not in df.columns:
        raise ValueError(f"指定的分類欄位 '{group_col}' 不存在於資料中")

    # 1. 資料編碼 (Cox 模型需要數值)
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 2. 產生圖表 (Image)
    current_backend = plt.get_backend()
    plt.switch_backend('Agg') 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    logrank_df = pd.DataFrame() # 預設空的 Log-Rank 結果

    # === 分組處理邏輯 ===
    if group_col:
        # (A) 有分組：分別畫線
        groups = df[group_col].unique()
        for i, group in enumerate(groups):
            mask = df[group_col] == group
            # label 設定為組名，顯示在圖例中
            kmf.fit(df.loc[mask, "time"], event_observed=df.loc[mask, "status"], label=str(group))
            kmf.plot(ax=ax, ci_show=False) 
        
        ax.set_title(f"Kaplan–Meier Curve by {group_col}")
        
        # (B) 執行 Log-Rank Test
        try:
            results = multivariate_logrank_test(
                df['time'], 
                df[group_col], 
                df['status']
            )
            
            logrank_data = {
                "Test": ["Multivariate Log-Rank"],
                "Group Column": [group_col],
                "test_statistic": [results.test_statistic],
                "p-value": [results.p_value],
                "Significant (p<0.05)": ["Yes" if results.p_value < 0.05 else "No"]
            }
            logrank_df = pd.DataFrame(logrank_data)
        except Exception as e:
            print(f"Log-Rank failed: {e}")

    else:
        # (A) 無分組：畫單一條線
        kmf.fit(df["time"], event_observed=df["status"])
        kmf.plot(ax=ax)
        ax.set_title("Kaplan–Meier Survival Curve (Overall)")

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    
    # 3. 訓練 Cox 模型
    cph = CoxPHFitter()
    report_text = ""
    cox_df = pd.DataFrame()

    try:
        cph.fit(df_encoded, duration_col="time", event_col="status")
        
        # 4. 獨立取出 Cox 統計表格
        stats_result = proportional_hazard_test(cph, df_encoded, time_transform='rank')
        cox_df = stats_result.summary.reset_index()
        cox_df.rename(columns={'index': 'variable'}, inplace=True)
        
        # 5. 提取文字報告
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            cph.check_assumptions(df_encoded, p_value_threshold=0.05, show_plots=False)
        report_text = _clean_report_text(f.getvalue())

    except Exception as e:
        report_text = f"Cox PH Model Fitting Failed: {str(e)}"
        
    plt.switch_backend(current_backend)

    return {
        "fig": fig,             
        "cox_df": cox_df,         # Cox 檢定表
        "logrank_df": logrank_df, # Log-Rank 檢定表
        "report_text": report_text
    }

@sv_register
class SurvivalAnalysis(svTest):
    name = "survival"
    display_name = "存活分析(KM+Cox+LogRank)"
    result_prefix = "survival"

    def run(self, df: pd.DataFrame, group_col: str = None):
        return analyze_survival_logic(df, group_col)