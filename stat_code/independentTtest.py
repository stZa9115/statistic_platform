import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import stats

def t_test(data: pd.DataFrame):
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
        method = "Mann_Whitney U"
        p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided').pvalue
    else:
        equal_var = levene_p >= 0.05
        method = f"t-test ({'Equal Var' if equal_var else 'Unequal Var'})"
        p_value = stats.ttest_ind(group1, group2, equal_var=equal_var).pvalue

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
