# hlm_server.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import io
import os
import traceback
# 導入你原本的邏輯 (假設存放在同目錄的 hlm_logic.py 或直接寫入此檔)
from hlm_pipeline import run_step1, run_step2, run_step3, run_step4, run_step5, run_step6
import numpy as np
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False  # 這是給新版 Flask (>=2.2) 使用的

# 用於暫存最後一次分析結果的字典 (簡單實作)
last_analysis_results = {}

def safe_to_dict(df):
    if df is None or df.empty:
        return []
    # 將 NaN, Inf 等替換為 None (JSON 會轉成 null) 或空字串
    df_clean = df.replace([np.inf, -np.inf, np.nan], None)
    return df_clean.to_dict(orient='records')

@app.route('/analyze', methods=['POST'])
def analyze():
    global last_analysis_results
    try:
        file = request.files['file']
        target = request.form.get('target', '').strip()
        g2 = request.form.get('g2', '').strip()
        g3 = request.form.get('g3', '').strip()
        
        # 確保過濾掉空字串
        l1_vars = [v for v in request.form.getlist('l1_vars[]') if v.strip()]
        l2_vars = [v for v in request.form.getlist('l2_vars[]') if v.strip()]
        l3_vars = [v for v in request.form.getlist('l3_vars[]') if v.strip()]

        # 雙重編碼防護讀取 CSV
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0) # 游標歸零重讀
            df = pd.read_csv(file, encoding='big5')

        # 檢查必填變項是否存在
        all_selected_vars = [target, g2, g3] + l1_vars + l2_vars + l3_vars
        all_selected_vars = [v for v in all_selected_vars if v] # 過濾空值
        
        # 檢查欄位是否真的在 DataFrame 裡面，避免 KeyError
        missing_cols = [col for col in all_selected_vars if col not in df.columns]
        if missing_cols:
            return jsonify({"status": "error", "message": f"CSV 找不到以下欄位: {', '.join(missing_cols)}"}), 400

        # 清理缺失值
        df = df.dropna(subset=all_selected_vars)

        results = {}
        
        # --- 執行步驟 (這裡包裝了錯誤攔截) ---
        print("Step 1...")
        f1, r1, fit1 = run_step1(df, target, g2, g3)
        results["step1"] = {"name": "Step1_具有隨機效果的單因子變異數分析模式", "fixed": f1, "random": r1, "fit": fit1}
        print(results["step1"])
        if l1_vars:
            print("Step 2...")
            f2, r2, fit2 = run_step2(df, target, l1_vars, g2, g3)
            results["step2"] = {"name": "Step2_隨機效果單因子共變數分析", "fixed": f2, "random": r2, "fit": fit2}
            
            print("Step 3...")
            f3, r3, fit3 = run_step3(df, target, l1_vars, g2, g3)
            results["step3"] = {"name": "Step3_隨機係數迴歸模式", "fixed": f3, "random": r3, "fit": fit3}

        if l2_vars:
            print("Step 4...")
            f4, r4, fit4 = run_step4(df, target, l2_vars, g2, g3)
            results["step4"] = {"name": "Step4_各組平均數做為結果變項迴歸", "fixed": f4, "random": r4, "fit": fit4}
        # ... (如果你有實作 Step 2 ~ 6，請依照同樣邏輯放進來) ...

        if l1_vars and l2_vars and l3_vars:
            print("Step 5...")
            f5, r5, fit5 = run_step5(df, target, l1_vars[0], l2_vars[0], l3_vars[0], g2, g3)
            results["step5"] = {"name": "Step5_帶有非隨機變化之斜率的模式", "fixed": f5, "random": r5, "fit": fit5}
            
            print("Step 6...")
            f6, r6, fit6 = run_step6(df, target, l1_vars, l2_vars, l3_vars, g2, g3)
            results["step6"] = {"name": "Step6_完整模式", "fixed": f6, "random": r6, "fit": fit6}

        last_analysis_results = results

        # 回傳給前端 JSON (使用我們寫好的 safe_to_dict 過濾 NaN)
        json_response = {}
        for k, v in results.items():
            if k == "step1":
                json_response[k] = {
                    "name": v["name"],
                    "fixed": safe_to_dict(v["fixed"]),
                    "random": safe_to_dict(v["random"]),
                    "fit": safe_to_dict(v["fit"])
                }
            else:
                json_response[k] = {
                    "name": v["name"],
                    "fixed": safe_to_dict(v["fixed"]),
                    "random": safe_to_dict(v["random"]),
                    "fit": safe_to_dict(v["fit"])
                }

        return jsonify({"status": "success", "data": json_response})

    except Exception as e:
        error_trace = traceback.format_exc()
        print("【伺服器錯誤】\n", error_trace) # 在後端終端機印出完整紅色錯誤
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/download', methods=['GET'])
def download():
    global last_analysis_results
    if not last_analysis_results:
        return "No results available", 404

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for step_key, step_data in last_analysis_results.items():
            sheet_name = step_data["name"][:31]
            step_data["fixed"].to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
            step_data["random"].to_excel(writer, sheet_name=sheet_name, startrow=len(step_data["fixed"])+2, index=False)
            step_data["fit"].to_excel(writer, sheet_name=sheet_name, startrow=len(step_data["fixed"])+len(step_data["random"])+4, index=False)
    
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="HLM_Analysis_Results.xlsx")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)