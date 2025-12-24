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

def anova(data: pd.DataFrame):
    if "group" not in data.columns or "score" not in data.columns:
        raise ValueError("DataFrame 必須包含 'group' 與 'score' 欄位")
    pre = Pretest()
    
    name_list, Group1_list, Group2_list = [], [], []
    Group1_num, Group2_num, M1_list, M2_list = [], [], [], []
    pval_list, is_diff_list, method_list = [], [], []

    groups = data["group"].unique()
    for g in groups:
        group_data = data.loc[data["group"] == g, "score"].dropna()
        pre.groupList.append(g)
        pre.sampleSizeList.append(len(group_data))
        pre.meanList.append(np.round(np.mean(group_data), 2))
        pre.p_value.append(None)
        pre.is_diff.append(None)
        pre.Method.append(None)
    
    dataList = [data.loc[data["group"] == g, "score"].dropna().values for g in groups]
    use_welch = any(st.shapiro(c).pvalue < 0.05 for c in dataList)
    

    if use_welch:
        welch_res = pg.welch_anova(data=data, dv='score', between='group')
        welch_p = welch_res['p-unc'][0]

        pre.p_value[0] =round(welch_p, 2) if welch_p >=0.05 else "< 0.05"
        pre.is_diff[0] = welch_p < 0.05
        pre.Method[0] = 'Welch ANOVA'

        if welch_p < 0.05:
            post = Posttest()

            gh = pg.pairwise_gameshowell(data=data, dv='score', between='group')
            for _, row in gh.iterrows():
                name_list.append(f"{row['A']}_vs_{row['B']}")
                if row['mean(A)'] >= row['mean(B)']:
                    post.groupHighList.append(row['A'])
                    post.groupLowList.append(row['B'])
                    post.mean_diffList.append(round(row['mean(A)'] - row['mean(B)'], 2))
                else:
                    post.groupHighList.append(row['B'])
                    post.groupLowList.append(row['A'])
                    post.mean_diffList.append(round(row['mean(B)'] - row['mean(A)'], 2))
                post.p_valueList.append(round(row['pval'], 2) if row['pval'] >=0.05 else "< 0.05")
                post.is_diffList.append(row['pval'] < 0.05)
                post.postHocList.append('Welch_GamesHowell')
                

    else:
        anova_res = pg.anova(data=data, dv='score', between='group')
        anova_p = anova_res['p-unc'][0]

        pre.p_value[0] = round(anova_p, 2) if anova_p >=0.05 else "< 0.05"
        pre.is_diff[0] = anova_p < 0.05
        pre.Method[0] = 'One-way ANOVA'
        if anova_p < 0.05:

            post = Posttest()

            hsd = pg.pairwise_tukey(data=data, dv='score', between='group')
            for _, row in hsd.iterrows():
                name_list.append(f"{row['A']}_vs_{row['B']}")
                if row['mean(A)'] >= row['mean(B)']:
                    post.groupHighList.append(row['A'])
                    post.groupLowList.append(row['B'])
                    post.mean_diffList.append(round(row['mean(A)'] - row['mean(B)'], 2))
                else:
                    post.groupHighList.append(row['B'])
                    post.groupLowList.append(row['A'])
                    post.mean_diffList.append(round(row['mean(B)'] - row['mean(A)'], 2))
                post.p_valueList.append(round(row['p-tukey'], 2) if row['p-tukey'] >=0.05 else "< 0.05")
                post.is_diffList.append(row['p-tukey'] < 0.05)
                post.postHocList.append('Tukey_HSD')

    if pre.is_diff[0] == False:
        result = {
            "post":False,
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
            "post":True,
            "pretest": {
                'Group': pre.groupList,
                'Sample Size': pre.sampleSizeList,
                'Mean': pre.meanList,
                'p-value': pre.p_value,
                'is_diff': pre.is_diff,
                'Method': pre.Method
            },
            "posttest": {
                # 'Comparison': name_list,
                'Group_High': post.groupHighList,
                'Group_Low': post.groupLowList,
                'Mean_Difference': post.mean_diffList,
                'p-value': post.p_valueList,
                'is_diff': post.is_diffList,
                'Method': post.postHocList
            }
        }
    return result


# 測試用資料
def generate_anova_test_data(seed=42, use_welch=False, is_significant=True):
    np.random.seed(seed)
    scale = 12 if use_welch else 5
    if is_significant:
        means = [50, 52, 65]
    else:
        means = [50, 51, 50]
    group_A = np.random.normal(loc=means[0], scale=scale, size=30)
    group_B = np.random.normal(loc=means[1], scale=scale, size=30)
    group_C = np.random.normal(loc=means[2], scale=scale, size=30)
    df = pd.DataFrame({
        "group": (["A"] * len(group_A)) + (["B"] * len(group_B)) + (["C"] * len(group_C)),
        "score": np.concatenate([group_A, group_B, group_C])
    })
    return df

@register
class ANOVA(statTest):
    name = "anova"
    display_name = "單因子變異數分析"
    result_prefix = "anova"

    def run(self, df: pd.DataFrame):
        return anova(df)
    

if __name__ == "__main__":
    df = generate_anova_test_data(is_significant=True)
    result = anova(df)

    print(result)
    print("-----")
    print(pd.DataFrame(result['pretest']))
    if result['post']:
        print(pd.DataFrame(result['posttest']))
