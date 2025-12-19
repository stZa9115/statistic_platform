import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import stats
from .base import statTest
from . import register

def paired_t_test(data: pd.DataFrame):
    col = data.columns
    if len(col) < 2:
        raise ValueError("資料需要至少兩個欄位才能進行 t-test")
    
    group1 = data[col[0]].dropna()
    group2 = data[col[1]].dropna()

    # --- 檢定 ---
    p1 = st.shapiro(group1).pvalue
    p2 = st.shapiro(group2).pvalue
    levene_p = st.levene(group1, group2, center='mean').pvalue

    # --- 選擇方法 ---
    if p1 < 0.05 or p2 < 0.05 or levene_p < 0.05:
        method = "Wilcoxon Signed-Rank"
        # p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided').pvalue
        p_value = stats.wilcoxon(group1, group2).pvalue
    else:
        equal_var = levene_p >= 0.05
        method = f"Paired t-test"
        p_value = stats.ttest_rel(group1, group2).pvalue

    # --- 結果組成 ---
    results = [
        {
            "Group": col[0],
            "Mean": np.mean(group1),
            "Median": np.median(group1),
            "Normality p": p1,
            "Levene p": levene_p,
            "Method": method,
            "p-value": p_value,
            "is_diff": p_value < 0.05,
        },
        {
            "Group": col[1],
            "Mean": np.mean(group2),
            "Median": np.median(group2),
            "Normality p": p2,
            # 只在第一列顯示 Levene / method / p-value
            "Levene p": None,
            "Method": None,
            "p-value": None,
            "is_diff": None,
        }
    ]

    results = {
        "Group": [col[0], col[1]],
        "Sample Size": [len(group1), len(group2)],
        "Mean": [np.round(np.mean(group1),2), np.round(np.mean(group2),2)],
        # "Median": [np.round(np.median(group1),2), np.round(np.median(group2),2)],
        
        "Normality p": [np.round(p1,2), np.round(p2,2)],
        "Levene p": [np.round(levene_p,2), None],
        "Method": [method, None],
        "p-value": [np.round(p_value,2), None],
        "is_diff": [p_value < 0.05, None],
    }

    return results

@register
class pairedTTest(statTest):
    name = "pairedTtest"
    display_name = "成對樣本 t 檢定"
    result_prefix = "paired_t_test"

    def run(self, df: pd.DataFrame):
        return paired_t_test(df)
    
if __name__ == "__main__":
    # 測試用
    data = pd.DataFrame({
        "A": [1,2,3,4,5,6,7,8,9,10],
        "B": [2,3,4,5,6,7,8,9,10,11],
    })
    result = paired_t_test(data)
    result = pd.DataFrame(result)
    result.to_csv("./check/test_independent_t_test_result.csv", index=False)