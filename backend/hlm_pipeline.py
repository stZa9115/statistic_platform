import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ==========================================
# 輔助函數區 (共用邏輯與格式化)
# ==========================================
def fmt(val):
    """將數值強制格式化為小數點後三位字串"""
    if isinstance(val, str):
        return val
    return f"{val:.3f}"

def format_p(p):
    """格式化 p 值，小於 .001 時顯示 <.001，否則顯示三位小數"""
    if isinstance(p, str):
        return p
    return "<.001" if p < 0.001 else f"{p:.3f}"

def safe_sqrt(val):
    """安全計算標準差，避免極小負數導致錯誤"""
    return np.sqrt(max(0.0, val))

def get_model_fit(result_full, data_len):
    """計算並回傳模型適配度 (Deviance, AIC, BIC) 表格"""
    k = result_full.k_fe + result_full.k_re + result_full.k_vc + 1
    dev = -2 * result_full.llf
    aic = dev + 2 * k
    bic = dev + k * np.log(data_len)
    return pd.DataFrame({
        "指標": ["Deviance (-2LL)", "AIC", "BIC"], 
        "數值": [fmt(dev), fmt(aic), fmt(bic)]
    })

def get_lrt(llf_full, formula, data, groups, re_formula, vc_formula=None):
    """自動執行縮減模型並計算 LRT 卡方值與 p 值 (回傳浮點數)"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if vc_formula is not None:
                model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula, vc_formula=vc_formula)
            else:
                model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
            res = model.fit(method='bfgs')
            lrt_chi2 = max(0.0, 2 * (llf_full - res.llf))
            p_val = chi2.sf(lrt_chi2, 1) if lrt_chi2 > 0 else 1.0
            return lrt_chi2, p_val
    except:
        return 0.0, 1.0

# ==========================================
# 步驟 1 ~ 6 模型定義區
# ==========================================

def run_step1(data, target_var, group_l2, group_l3):
    formula = f"{target_var} ~ 1"
    vcf = {group_l2: f"0 + C({group_l2})"}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = smf.mixedlm(formula, data, groups=data[group_l3], re_formula="1", vc_formula=vcf).fit()

    llf_full = res.llf
    lrt_l3_chi2, lrt_l3_p = get_lrt(llf_full, formula, data, data[group_l2], "1")
    lrt_l2_chi2, lrt_l2_p = get_lrt(llf_full, formula, data, data[group_l3], "1")

    df_fixed = pd.DataFrame({
        "固定效果": ["γ000 (截距)"], 
        "係數": [fmt(res.params.iloc[0])],
        "估計標準誤": [fmt(res.bse.iloc[0])], 
        "p 值": [format_p(res.pvalues.iloc[0])]
    })

    var_l3 = res.cov_re.iloc[0, 0] if hasattr(res, 'cov_re') and not res.cov_re.empty else 0.0
    var_l2 = res.vcomp[0] if hasattr(res, 'vcomp') and len(res.vcomp) > 0 else 0.0
    var_resid = res.scale

    total_var = var_l3 + var_l2 + var_resid
    icc_l3 = var_l3 / total_var if total_var > 0 else 0.0
    icc_l2 = var_l2 / total_var if total_var > 0 else 0.0
    icc_resid = var_resid / total_var if total_var > 0 else 0.0

    # 傳統 HLM 自由度算法
    df_l3 = data[group_l3].nunique() - 1
    df_l2 = data[group_l2].nunique() - data[group_l3].nunique()

    df_random = pd.DataFrame({
        "隨機效果": ["μ00k (階層三)", "r0jk (階層二)", "eijk (階層一)"],
        "標準差": [fmt(safe_sqrt(var_l3)), fmt(safe_sqrt(var_l2)), fmt(safe_sqrt(var_resid))],
        "變異數": [fmt(var_l3), fmt(var_l2), fmt(var_resid)],
        "自由度": [int(df_l3), int(df_l2), "-"],
        "Chi-square": [fmt(lrt_l3_chi2), fmt(lrt_l2_chi2), "-"], 
        "p 值": [format_p(lrt_l3_p), format_p(lrt_l2_p), "-"],
        "解釋變異比例(ICC)": [fmt(icc_l3), fmt(icc_l2), fmt(icc_resid)]
    })
    return df_fixed, df_random, get_model_fit(res, len(data))

def run_step2(data, target_var, l1_vars, group_l2, group_l3):
    formula = f"{target_var} ~ {' + '.join(l1_vars)}"
    vcf = {group_l2: f"0 + C({group_l2})"}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = smf.mixedlm(formula, data, groups=data[group_l3], re_formula="1", vc_formula=vcf).fit(method='bfgs')
    
    llf_full = res.llf
    lrt_l3_chi2, lrt_l3_p = get_lrt(llf_full, formula, data, data[group_l2], "1")
    lrt_l2_chi2, lrt_l2_p = get_lrt(llf_full, formula, data, data[group_l3], "1")
        
    fixed_data = [{"固定效果": "γ000 (截距)", "係數": fmt(res.params["Intercept"]), 
                   "估計標準誤": fmt(res.bse["Intercept"]), "p 值": format_p(res.pvalues["Intercept"])}]
    for i, var in enumerate(l1_vars):
        fixed_data.append({"固定效果": f"γ{i+1}00 ({var})", "係數": fmt(res.params[var]),
                           "估計標準誤": fmt(res.bse[var]), "p 值": format_p(res.pvalues[var])})
        
    var_l3 = res.cov_re.iloc[0, 0] if not res.cov_re.empty else 0.0
    var_l2 = res.vcomp[0] if len(res.vcomp) > 0 else 0.0
    var_resid = res.scale

    total_var = var_l3 + var_l2 + var_resid
    icc_l3 = var_l3 / total_var if total_var > 0 else 0.0
    icc_l2 = var_l2 / total_var if total_var > 0 else 0.0
    icc_resid = var_resid / total_var if total_var > 0 else 0.0
    
    df_l3 = data[group_l3].nunique() - 1
    df_l2 = data[group_l2].nunique() - data[group_l3].nunique()

    df_random = pd.DataFrame({
        "隨機效果": ["μ00k (階層三)", "r0jk (階層二)", "eijk (殘差)"],
        "標準差": [fmt(safe_sqrt(var_l3)), fmt(safe_sqrt(var_l2)), fmt(safe_sqrt(var_resid))],
        "變異數": [fmt(var_l3), fmt(var_l2), fmt(var_resid)],
        "自由度": [df_l3, df_l2, "-"],
        "Chi-square": [fmt(lrt_l3_chi2), fmt(lrt_l2_chi2), "-"],
        "p 值": [format_p(lrt_l3_p), format_p(lrt_l2_p), "-"],
        "解釋變異比例(ICC)": [fmt(icc_l3), fmt(icc_l2), fmt(icc_resid)]
    })
    return pd.DataFrame(fixed_data), df_random, get_model_fit(res, len(data))

def run_step3(data, target_var, l1_vars, group_l2, group_l3):
    formula = f"{target_var} ~ {' + '.join(l1_vars)}"
    vcf = {"Level2_Intercept": f"0 + C({group_l2})"}
    for v in l1_vars: vcf[f"Level2_Slope_{v}"] = f"0 + C({group_l2}):{v}"
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = smf.mixedlm(formula, data, groups=data[group_l3], re_formula="1", vc_formula=vcf)
        res = model.fit(method='bfgs')
    
    llf_full = res.llf
    vcomp_dict = dict(zip(model.exog_vc.names, res.vcomp)) if hasattr(model, 'exog_vc') else {}
    
    fixed_data = [{"固定效果": "γ000 (截距)", "係數": fmt(res.params["Intercept"]), 
                   "估計標準誤": fmt(res.bse["Intercept"]), "p 值": format_p(res.pvalues["Intercept"])}]
    for i, var in enumerate(l1_vars):
        fixed_data.append({"固定效果": f"γ{i+1}00 ({var})", "係數": fmt(res.params[var]),
                           "估計標準誤": fmt(res.bse[var]), "p 值": format_p(res.pvalues[var])})

    vcf_no_l3 = {f"Slope_{v}": f"0 + {v}" for v in l1_vars}
    lrt_l3_chi2, lrt_l3_p = get_lrt(llf_full, formula, data, data[group_l2], "1", vc_formula=vcf_no_l3)
    
    vcf_no_l2_int = {k: v for k, v in vcf.items() if k != "Level2_Intercept"}
    lrt_l2_chi2, lrt_l2_p = get_lrt(llf_full, formula, data, data[group_l3], "1", vc_formula=vcf_no_l2_int)

    var_l3 = res.cov_re.iloc[0, 0] if not res.cov_re.empty else 0.0
    var_l2_int = vcomp_dict.get("Level2_Intercept", 0.0)

    df_l3 = data[group_l3].nunique() - 1
    df_l2 = data[group_l2].nunique() - data[group_l3].nunique()

    rand_data = [
        {"隨機效果": "μ00k (階層三)", "標準差": fmt(safe_sqrt(var_l3)), "變異數": fmt(var_l3), "自由度": df_l3, "Chi-square": fmt(lrt_l3_chi2), "p 值": format_p(lrt_l3_p)},
        {"隨機效果": "r0jk (階層二截距)", "標準差": fmt(safe_sqrt(var_l2_int)), "變異數": fmt(var_l2_int), "自由度": df_l2, "Chi-square": fmt(lrt_l2_chi2), "p 值": format_p(lrt_l2_p)}
    ]
    for i, var in enumerate(l1_vars):
        var_slope = vcomp_dict.get(f"Level2_Slope_{var}", 0.0)
        vcf_no_s = {k: v for k, v in vcf.items() if k != f"Level2_Slope_{var}"}
        s_chi2, s_p = get_lrt(llf_full, formula, data, data[group_l3], "1", vc_formula=vcf_no_s)
        rand_data.append({"隨機效果": f"r{i+1}jk (斜率-{var})", "標準差": fmt(safe_sqrt(var_slope)), "變異數": fmt(var_slope), "自由度": df_l2, "Chi-square": fmt(s_chi2), "p 值": format_p(s_p)})
        
    rand_data.append({"隨機效果": "eijk (殘差)", "標準差": fmt(safe_sqrt(res.scale)), "變異數": fmt(res.scale), "自由度": "-", "Chi-square": "-", "p 值": "-"})
    
    return pd.DataFrame(fixed_data), pd.DataFrame(rand_data), get_model_fit(res, len(data))

def run_step4(data, target_var, l2_vars, group_l2, group_l3):
    formula = f"{target_var} ~ {' + '.join(l2_vars)}"
    vcf = {group_l2: f"0 + C({group_l2})"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = smf.mixedlm(formula, data, groups=data[group_l3], re_formula="1", vc_formula=vcf).fit(method='bfgs')
        
    llf_full = res.llf
    lrt_l3_chi2, lrt_l3_p = get_lrt(llf_full, formula, data, data[group_l2], "1")
    lrt_l2_chi2, lrt_l2_p = get_lrt(llf_full, formula, data, data[group_l3], "1")
        
    fixed_data = [{"固定效果": "γ000 (截距)", "係數": fmt(res.params["Intercept"]), 
                   "估計標準誤": fmt(res.bse["Intercept"]), "p 值": format_p(res.pvalues["Intercept"])}]
    for i, var in enumerate(l2_vars):
        fixed_data.append({"固定效果": f"γ0{i+1}0 ({var})", "係數": fmt(res.params[var]),
                           "估計標準誤": fmt(res.bse[var]), "p 值": format_p(res.pvalues[var])})
        
    var_l3 = res.cov_re.iloc[0, 0] if not res.cov_re.empty else 0.0
    var_l2 = res.vcomp[0] if len(res.vcomp) > 0 else 0.0
    
    df_l3 = data[group_l3].nunique() - 1
    df_l2 = data[group_l2].nunique() - data[group_l3].nunique()

    df_random = pd.DataFrame({
        "隨機效果": ["μ00k (階層三)", "r0jk (階層二)", "eijk (殘差)"],
        "標準差": [fmt(safe_sqrt(var_l3)), fmt(safe_sqrt(var_l2)), fmt(safe_sqrt(res.scale))],
        "變異數": [fmt(var_l3), fmt(var_l2), fmt(res.scale)],
        "自由度": [df_l3, df_l2, "-"],
        "Chi-square": [fmt(lrt_l3_chi2), fmt(lrt_l2_chi2), "-"],
        "p 值": [format_p(lrt_l3_p), format_p(lrt_l2_p), "-"]
    })
    return pd.DataFrame(fixed_data), df_random, get_model_fit(res, len(data))

def run_step5(data, target_var, l1_var, l2_var, l3_var, group_l2, group_l3):
    formula = f"{target_var} ~ {l1_var} * {l2_var} * {l3_var}"
    vcf = {"Level2_Intercept": f"0 + C({group_l2})", f"Level2_Slope_{l1_var}": f"0 + C({group_l2}):{l1_var}"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = smf.mixedlm(formula, data, groups=data[group_l3], re_formula="1", vc_formula=vcf).fit(method='bfgs')
    
    llf_full = res.llf
    param_names_map = {
        "Intercept": ("γ000", "截距"),
        l3_var: ("γ001", l3_var),
        l2_var: ("γ010", l2_var),
        f"{l2_var}:{l3_var}": ("γ011", f"{l2_var}*{l3_var}"),
        l1_var: ("γ100", l1_var),
        f"{l1_var}:{l3_var}": ("γ101", f"{l1_var}*{l3_var}"),
        f"{l1_var}:{l2_var}": ("γ110", f"{l1_var}*{l2_var}"),
        f"{l1_var}:{l2_var}:{l3_var}": ("γ111", f"{l1_var}*{l2_var}*{l3_var}")
    }
    
    fixed_data = []
    for raw_name, val in res.params.items():
        if raw_name in param_names_map:
            gamma, label = param_names_map[raw_name]
            fixed_data.append({
                "固定效果": f"{gamma} ({label})",
                "係數": fmt(val),
                "估計標準誤": fmt(res.bse[raw_name]),
                "p 值": format_p(res.pvalues[raw_name])
            })
    fixed_data = sorted(fixed_data, key=lambda x: x["固定效果"])
        
    vcf_no_l3 = {f"Slope_{l1_var}": f"0 + {l1_var}"}
    lrt_l3_chi2, lrt_l3_p = get_lrt(llf_full, formula, data, data[group_l2], "1", vc_formula=vcf_no_l3)
    
    vcf_no_l2_int = {k: v for k, v in vcf.items() if k != "Level2_Intercept"}
    lrt_l2_int_chi2, lrt_l2_int_p = get_lrt(llf_full, formula, data, data[group_l3], "1", vc_formula=vcf_no_l2_int)
    
    vcf_no_l2_slope = {k: v for k, v in vcf.items() if k != f"Level2_Slope_{l1_var}"}
    lrt_l2_s_chi2, lrt_l2_s_p = get_lrt(llf_full, formula, data, data[group_l3], "1", vc_formula=vcf_no_l2_slope)

    var_l3 = res.cov_re.iloc[0, 0] if not res.cov_re.empty else 0.0
    vcomp_dict = dict(zip(res.model.exog_vc.names, res.vcomp)) if hasattr(res.model, 'exog_vc') else {}
    var_l2_int = vcomp_dict.get("Level2_Intercept", 0.0)
    var_l2_slope = vcomp_dict.get(f"Level2_Slope_{l1_var}", 0.0)

    df_l3 = data[group_l3].nunique() - 1
    df_l2 = data[group_l2].nunique() - data[group_l3].nunique()

    df_random = pd.DataFrame({
        "隨機效果": ["μ00k (階層三)", "r0jk (階層二截距)", f"r1jk (階層二斜率-{l1_var})", "eijk (殘差)"],
        "標準差": [fmt(safe_sqrt(var_l3)), fmt(safe_sqrt(var_l2_int)), fmt(safe_sqrt(var_l2_slope)), fmt(safe_sqrt(res.scale))],
        "變異數": [fmt(var_l3), fmt(var_l2_int), fmt(var_l2_slope), fmt(res.scale)],
        "自由度": [df_l3, df_l2, df_l2, "-"],
        "Chi-square": [fmt(lrt_l3_chi2), fmt(lrt_l2_int_chi2), fmt(lrt_l2_s_chi2), "-"],
        "p 值": [format_p(lrt_l3_p), format_p(lrt_l2_int_p), format_p(lrt_l2_s_p), "-"]
    })
    return pd.DataFrame(fixed_data), df_random, get_model_fit(res, len(data))

def run_step6(data, target_var, l1_vars, l2_vars, l3_vars, group_l2, group_l3):
    l1_str, l2_str, l3_str = " + ".join(l1_vars), " + ".join(l2_vars), " + ".join(l3_vars)
    formula = f"{target_var} ~ ({l1_str}) * ({l2_str}) * ({l3_str})"
    vcf = {"Level2_Intercept": f"0 + C({group_l2})"}
    for v in l1_vars: vcf[f"Level2_Slope_{v}"] = f"0 + C({group_l2}):{v}"
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = smf.mixedlm(formula, data, groups=data[group_l3], re_formula="1", vc_formula=vcf).fit(method='bfgs')
    
    llf_full = res.llf
    param_names_map = {"Intercept": ("γ000", "截距")}
    for l3_idx, l3_var in enumerate(l3_vars):
        param_names_map[l3_var] = (f"γ00{l3_idx+1}", l3_var)
    for l2_idx, l2_var in enumerate(l2_vars):
        param_names_map[l2_var] = (f"γ0{l2_idx+1}0", l2_var)
        for l3_idx, l3_var in enumerate(l3_vars):
            param_names_map[f"{l2_var}:{l3_var}"] = (f"γ0{l2_idx+1}{l3_idx+1}", f"{l2_var}*{l3_var}")
    for l1_idx, l1_var in enumerate(l1_vars):
        param_names_map[l1_var] = (f"γ{l1_idx+1}00", l1_var)
        for l3_idx, l3_var in enumerate(l3_vars):
            param_names_map[f"{l1_var}:{l3_var}"] = (f"γ{l1_idx+1}0{l3_idx+1}", f"{l1_var}*{l3_var}")
        for l2_idx, l2_var in enumerate(l2_vars):
            param_names_map[f"{l1_var}:{l2_var}"] = (f"γ{l1_idx+1}{l2_idx+1}0", f"{l1_var}*{l2_var}")
            for l3_idx, l3_var in enumerate(l3_vars):
                param_names_map[f"{l1_var}:{l2_var}:{l3_var}"] = (f"γ{l1_idx+1}{l2_idx+1}{l3_idx+1}", f"{l1_var}*{l2_var}*{l3_var}")
    
    fixed_data = []
    for raw_name, val in res.params.items():
        if raw_name in param_names_map:
            gamma, label = param_names_map[raw_name]
            fixed_data.append({
                "固定效果": f"{gamma} ({label})",
                "係數": fmt(val),
                "估計標準誤": fmt(res.bse[raw_name]),
                "p 值": format_p(res.pvalues[raw_name])
            })
    fixed_data = sorted(fixed_data, key=lambda x: x["固定效果"])

    vcf_no_l3 = {f"Slope_{v}": f"0 + {v}" for v in l1_vars}
    lrt_l3_chi2, lrt_l3_p = get_lrt(llf_full, formula, data, data[group_l2], "1", vc_formula=vcf_no_l3)
    
    vcf_no_l2_int = {k: v for k, v in vcf.items() if k != "Level2_Intercept"}
    lrt_l2_int_chi2, lrt_l2_int_p = get_lrt(llf_full, formula, data, data[group_l3], "1", vc_formula=vcf_no_l2_int)

    var_l3 = res.cov_re.iloc[0, 0] if not res.cov_re.empty else 0.0
    vcomp_dict = dict(zip(res.model.exog_vc.names, res.vcomp)) if hasattr(res.model, 'exog_vc') else {}
    var_l2_int = vcomp_dict.get("Level2_Intercept", 0.0)

    df_l3 = data[group_l3].nunique() - 1
    df_l2 = data[group_l2].nunique() - data[group_l3].nunique()

    rand_data = [
        {"隨機效果": "μ00k (階層三)", "標準差": fmt(safe_sqrt(var_l3)), "變異數": fmt(var_l3), "自由度": df_l3, "Chi-square": fmt(lrt_l3_chi2), "p 值": format_p(lrt_l3_p)},
        {"隨機效果": "r0jk (階層二截距)", "標準差": fmt(safe_sqrt(var_l2_int)), "變異數": fmt(var_l2_int), "自由度": df_l2, "Chi-square": fmt(lrt_l2_int_chi2), "p 值": format_p(lrt_l2_int_p)}
    ]
    
    for i, v in enumerate(l1_vars):
        var_slope = vcomp_dict.get(f"Level2_Slope_{v}", 0.0)
        vcf_no_s = {k: val for k, val in vcf.items() if k != f"Level2_Slope_{v}"}
        s_chi2, s_p = get_lrt(llf_full, formula, data, data[group_l3], "1", vc_formula=vcf_no_s)
        rand_data.append({"隨機效果": f"r_jk (斜率-{v})", "標準差": fmt(safe_sqrt(var_slope)), "變異數": fmt(var_slope), "自由度": df_l2, "Chi-square": fmt(s_chi2), "p 值": format_p(s_p)})
        
    rand_data.append({"隨機效果": "eijk (殘差)", "標準差": fmt(safe_sqrt(res.scale)), "變異數": fmt(res.scale), "自由度": "-", "Chi-square": "-", "p 值": "-"})

    return pd.DataFrame(fixed_data), pd.DataFrame(rand_data), get_model_fit(res, len(data))


# ==========================================
# 主執行區塊與 Excel 匯出 / CMD 展示
# ==========================================
if __name__ == "__main__":
    print("載入資料中...")
    df = pd.read_csv("hlm_simulated_data_new.csv")
    
    target = '認知表現'
    l1_vars = ['AI使用程度code', '英語能力code']
    l2_vars = ['數位政策推行code']
    l3_vars = ['武力預算展現']
    g2 = '學校'
    g3 = '國家'
    # df = pd.read_csv("student_subset_2018.csv")
    
    # target = 'math'
    # l1_vars = ['escs']
    # l2_vars = ['']
    # l3_vars = ['']
    # g2 = 'school_id'
    # g3 = 'country'
    print(len(set(df['學校'])))
    
    
    # df = df.dropna(subset=[target, g2, g3] + l1_vars + l2_vars + l3_vars)
    df = df.dropna(subset=[target, g2, g3] + l1_vars)
    
    excel_file = "HLM_Results_R1.xlsx"
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    
    steps = [
        ("Step1_具有隨機效果單因子變異數分析", lambda: run_step1(df, target, g2, g3)),
        ("Step2_隨機效果單因子共變數分析", lambda: run_step2(df, target, l1_vars, g2, g3)),
        ("Step3_隨機係數迴歸模式", lambda: run_step3(df, target, l1_vars, g2, g3)),
        ("Step4_各組平均數做為結果變項迴歸", lambda: run_step4(df, target, l2_vars, g2, g3)),
        ("Step5_帶有非隨機變化之斜率的模式", lambda: run_step5(df, target, l1_vars[0], l2_vars[0], l3_vars[0], g2, g3)),
        ("Step6_完整模式", lambda: run_step6(df, target, l1_vars, l2_vars, l3_vars, g2, g3))
    ]
    
    print("開始執行 HLM 分析流程 (共 6 步)...\n")
    
    results_store = []
    
    for step_name, step_func in steps:
        print(f"正在運算 {step_name} ...")
        df_fixed, df_random, df_fit = step_func()
        
        results_store.append((step_name, df_fixed, df_random, df_fit))
        
        df_fixed.to_excel(writer, sheet_name=step_name[:31], startrow=0, index=False)
        df_random.to_excel(writer, sheet_name=step_name[:31], startrow=len(df_fixed)+2, index=False)
        df_fit.to_excel(writer, sheet_name=step_name[:31], startrow=len(df_fixed)+len(df_random)+4, index=False)
        
    writer.close()
    print(f"\n分析完成，報表已匯出至：{excel_file}")
    
    # --- 在程式最後，依序輸出所有表格到 CMD ---
    print("\n" + "=" * 80)
    print(f"{'HLM 六大步驟表格結果總覽':^70}")
    print("=" * 80 + "\n")
    
    for step_name, df_fixed, df_random, df_fit in results_store:
        print(f"【 {step_name} 】")
        print("-" * 30 + " 固定效果 " + "-" * 30)
        print(df_fixed.to_string(index=False))
        
        print("\n" + "-" * 30 + " 隨機效果 " + "-" * 30)
        print(df_random.to_string(index=False))
        
        print("\n" + "-" * 30 + " 模型適配度 " + "-" * 28)
        print(df_fit.to_string(index=False))
        
        print("\n" + "=" * 80 + "\n")
