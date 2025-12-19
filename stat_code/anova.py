import numpy as np
import pandas as pd
import scipy.stats as st
import pingouin as pg
from .base import statTest
from . import register

def anova(data: pd.DataFrame):
    if "group" not in data.columns or "score" not in data.columns:
        raise ValueError("DataFrame 必須包含 'group' 與 'score' 欄位")

    name_list, Group1_list, Group2_list = [], [], []
    Group1_num, Group2_num, M1_list, M2_list = [], [], [], []
    pval_list, is_diff_list, method_list = [], [], []

    groups = data["group"].unique()
    dataList = [data.loc[data["group"] == g, "score"].dropna().values for g in groups]

    use_welch = any(st.shapiro(c).pvalue < 0.05 for c in dataList)

    if use_welch:
        welch_res = pg.welch_anova(data=data, dv='score', between='group')
        welch_p = welch_res['p-unc'][0]
        if welch_p < 0.05:
            gh = pg.pairwise_gameshowell(data=data, dv='score', between='group')
            for _, row in gh.iterrows():
                name_list.append(f"{row['A']}_vs_{row['B']}")
                Group1_list.append(row['A'])
                Group2_list.append(row['B'])
                Group1_num.append(len(data[data["group"] == row['A']]))
                Group2_num.append(len(data[data["group"] == row['B']]))
                M1_list.append(round(row['mean(A)'], 2))
                M2_list.append(round(row['mean(B)'], 2))
                # pval_list.append(round(row['pval'], 4))
                is_diff_list.append(row['pval'] < 0.05)
                method_list.append('Welch_GamesHowell')
                if row['pval'] >= 0.05:
                    pval_list.append(round(row['pval'], 2))
                else:
                    pval_list.append("< 0.05")
    else:
        anova_res = pg.anova(data=data, dv='score', between='group')
        anova_p = anova_res['p-unc'][0]
        if anova_p < 0.05:
            hsd = pg.pairwise_tukey(data=data, dv='score', between='group')
            for _, row in hsd.iterrows():
                name_list.append(f"{row['A']}_vs_{row['B']}")
                Group1_list.append(row['A'])
                Group2_list.append(row['B'])
                Group1_num.append(len(data[data["group"] == row['A']]))
                Group2_num.append(len(data[data["group"] == row['B']]))
                M1_list.append(round(row['mean(A)'], 2))
                M2_list.append(round(row['mean(B)'], 2))
                # pval_list.append(round(row['p-tukey'], 4))
                is_diff_list.append(row['p-tukey'] < 0.05)
                method_list.append('Tukey HSD')

                if row['p-tukey'] >= 0.05:
                    pval_list.append(round(row['p-tukey'], 2))
                else:
                    pval_list.append("< 0.05")

    result = {
        # 'Comparison': name_list,
        'Group1': Group1_list,
        'Sample Size 1': Group1_num,
        'Group2': Group2_list,
        'Sample Size 2': Group2_num,
        'Mean1': M1_list,
        'Mean2': M2_list,
        'p-value': pval_list,
        'is_diff': is_diff_list,
        'Method': method_list
    }

    return result

# 測試用資料
def generate_anova_test_data(seed=42):
    np.random.seed(seed)
    group_A = np.random.normal(loc=50, scale=5, size=30)
    group_B = np.random.normal(loc=52, scale=5, size=30)
    group_C = np.random.normal(loc=65, scale=5, size=30)
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
    df = generate_anova_test_data()
    print(anova(df))
