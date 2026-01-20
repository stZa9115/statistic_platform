import pandas as pd
import numpy as np
import io
import re
import contextlib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# 假設您的架構中有這些 import，依據 anova.py 的結構
from .sv_base import svTest
from . import sv_register

def _clean_report_text(full_text):
    """
    清洗 check_assumptions 的輸出，移除中間的 ASCII 表格。
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

def analyze_survival_logic(df: pd.DataFrame):
    """
    執行生存分析邏輯
    """
    # 欄位檢查 (這裡假設 Excel 必須包含 time 和 status)
    # 如果您的資料欄位名稱不同，請在此修改或在前端統一
    if "time" not in df.columns or "status" not in df.columns:
        raise ValueError("DataFrame 必須包含 'time' (時間) 與 'status' (事件狀態 0/1) 欄位")

    # 1. 資料編碼 (Cox 模型需要數值)
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 2. 產生圖表 (Image)
    # 使用 Agg backend 避免在伺服器端跳出視窗
    current_backend = plt.get_backend()
    plt.switch_backend('Agg') 
    
    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], event_observed=df["status"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf.plot(ax=ax)
    ax.set_title("Kaplan–Meier Survival Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    
    # 3. 訓練 Cox 模型
    cph = CoxPHFitter()
    cph.fit(df_encoded, duration_col="time", event_col="status")
    
    # 4. 【Table】獨立取出統計表格 DataFrame
    stats_result = proportional_hazard_test(cph, df_encoded, time_transform='rank')
    result_df = stats_result.summary.reset_index() # reset index 讓變數名稱變成一個欄位
    result_df.rename(columns={'index': 'variable'}, inplace=True)
    
    # 5. 【Text】提取並清洗文字報告
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        cph.check_assumptions(df_encoded, p_value_threshold=0.05, show_plots=False)
    
    full_raw_log = f.getvalue()
    report_text = _clean_report_text(full_raw_log)
    
    # 恢復原本的 backend (雖然在 API context 通常不需要，但為了安全)
    plt.switch_backend(current_backend)

    return {
        "fig": fig,             # Matplotlib Figure 物件
        "stats_df": result_df,  # 統計 DataFrame
        "report_text": report_text # 清洗後的文字報告
    }

@sv_register
class SurvivalAnalysis(svTest):
    name = "survival"
    display_name = "存活分析(KM+Cox)"
    result_prefix = "survival"

    def run(self, df: pd.DataFrame):
        # 這裡呼叫上面的邏輯函數
        return analyze_survival_logic(df)