# 球形檢定
import numpy as np
import pandas as pd
import scipy.stats as st
import pingouin as pg
from .base import statTest
from . import register

class Pretest():
    def __init__(self):
        self.groupList = []
        self.sampleSizeList = []
        self.meanList = []
        self.p_value = []
        self.is_diff = []
        self.Method = []
class Posttest():
    def __init__(self):
        self.groupHighList = []
        self.groupLowList = []
        self.mean_diffList = []
        self.p_valueList = []
        self.is_diffList = []
        self.postHocList = []

def repeated_anova(data: pd.DataFrame):
    # --- 資料前處理 ---
    # 判斷是長格式 (Long) 還是寬格式 (Wide)
    
    # 情況 A: 長格式 (Long Format)
    if "group" in data.columns and "score" in data.columns:
        subject_col = None
        for col in data.columns:
            if col.lower() in ['subject', 'id', 'participant', 'user']:
                subject_col = col
                break
        
        if subject_col:
             long_data = data.rename(columns={"group": "Time", "score": "Score", subject_col: "Subject"})
        else:
             raise ValueError("執行重複測量 ANOVA 的長格式資料必須包含 'Subject' (受試者) 欄位")
    
    # 情況 B: 寬格式 (Wide Format)
    else:
        long_data = data.copy()
        long_data['Subject'] = long_data.index
        long_data = long_data.melt(id_vars='Subject', var_name='Time', value_name='Score')
    
    long_data['Score'] = pd.to_numeric(long_data['Score'], errors='coerce')
    long_data = long_data.dropna(subset=['Score'])

    pre = Pretest()
    
    # --- 計算 Pretest 描述統計 ---
    groups = long_data['Time'].unique()
    
    for g in groups:
        group_data = long_data.loc[long_data['Time'] == g, "Score"]
        pre.groupList.append(g)
        pre.sampleSizeList.append(len(group_data))
        pre.meanList.append(np.round(np.mean(group_data), 2))
        pre.p_value.append(None)
        pre.is_diff.append(None)
        pre.Method.append(None)
    
    # --- 執行重複測量變異數分析 (RM ANOVA) ---
    try:
        aov = pg.rm_anova(data=long_data, dv='Score', within='Time', subject='Subject', detailed=True, correction='auto')
        
        # 判斷球形假設並選取 p-value
        if 'p-GG-corr' in aov.columns and 'p-spher' in aov.columns:
            p_spher = aov['p-spher'][0]
            if p_spher < 0.05:
                p_val = aov['p-GG-corr'][0]
                method_name = 'RM ANOVA (Greenhouse-Geisser)'
            else:
                p_val = aov['p-unc'][0]
                method_name = 'Repeated Measures ANOVA'
        else:
            p_val = aov['p-unc'][0]
            method_name = 'Repeated Measures ANOVA'

        pre.p_value[0] = round(p_val, 2) if p_val >= 0.05 else "< 0.05"
        pre.is_diff[0] = p_val < 0.05
        pre.Method[0] = method_name
        
        is_significant = p_val < 0.05

    except Exception as e:
        pre.p_value[0] = str(e)
        pre.is_diff[0] = False
        pre.Method[0] = "Error"
        is_significant = False
    
    # --- 執行事後檢定 (Post-hoc) ---
    post = None
    if is_significant:
        post = Posttest()
        # Pairwise tests (Bonferroni correction)
        pwt = pg.pairwise_tests(data=long_data, dv='Score', within='Time', subject='Subject', padjust='bonf')
        
        for _, row in pwt.iterrows():
            # 修正處：手動計算平均值，不依賴 row['mean(A)']
            group_a_name = row['A']
            group_b_name = row['B']
            
            mean_A = long_data.loc[long_data['Time'] == group_a_name, 'Score'].mean()
            mean_B = long_data.loc[long_data['Time'] == group_b_name, 'Score'].mean()
            
            # 取得 p-value (優先取用校正後的值)
            if 'p-corr' in row:
                p_corr = row['p-corr']
            elif 'p-bonf' in row:
                 p_corr = row['p-bonf']
            else:
                 p_corr = row['p-unc']

            # 記錄比較結果
            if mean_A >= mean_B:
                post.groupHighList.append(group_a_name)
                post.groupLowList.append(group_b_name)
                post.mean_diffList.append(round(mean_A - mean_B, 2))
            else:
                post.groupHighList.append(group_b_name)
                post.groupLowList.append(group_a_name)
                post.mean_diffList.append(round(mean_B - mean_A, 2))
            
            post.p_valueList.append(round(p_corr, 2) if p_corr >= 0.05 else "< 0.05")
            post.is_diffList.append(p_corr < 0.05)
            post.postHocList.append('Pairwise t-test (Bonferroni)')
            
    # --- 建構回傳結果 ---
    if not is_significant:
        result = {
            "post": False,
            "pretest": {
                'Group': pre.groupList,
                'Sample Size': pre.sampleSizeList,
                'Mean': pre.meanList,
                'p-value': pre.p_value,
                'is_diff': pre.is_diff,
                'Method': pre.Method
            }
        }
    else:
        result = {
            "post": True,
            "pretest": {
                'Group': pre.groupList,
                'Sample Size': pre.sampleSizeList,
                'Mean': pre.meanList,
                'p-value': pre.p_value,
                'is_diff': pre.is_diff,
                'Method': pre.Method
            },
            "posttest": {
                'Group_High': post.groupHighList,
                'Group_Low': post.groupLowList,
                'Mean_Difference': post.mean_diffList,
                'p-value': post.p_valueList,
                'is_diff': post.is_diffList,
                'Method': post.postHocList
            }
        }
    return result

# 測試資料生成函式
def generate_anova_test_data(seed=42, is_significant=True):
    np.random.seed(seed)
    n = 30
    subject_means = np.random.normal(50, 10, n)
    
    if is_significant:
        effects = [0, 5, 10]
    else:
        effects = [0, 0, 0]
        
    data = {}
    time_points = ['Time1', 'Time2', 'Time3']
    for i, effect in enumerate(effects):
        scores = subject_means + effect + np.random.normal(0, 3, n)
        data[time_points[i]] = scores
        
    return pd.DataFrame(data)

@register
class RepeatedANOVA(statTest):
    name = "repeatedAnova"
    display_name = "重複測量變異數分析"
    result_prefix = "repeated_anova"

    def run(self, df: pd.DataFrame):
        return repeated_anova(df)

if __name__ == "__main__":
    df = generate_anova_test_data(is_significant=True)
    print("Generated Data Head:")
    print(df.head())
    
    result = repeated_anova(df)

    print("\nResult Dictionary:")
    print(result)