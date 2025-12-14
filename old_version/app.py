from flask import Flask, request, jsonify, send_file
import pandas as pd
from io import BytesIO
from datetime import datetime
import secrets
from independentTtest import t_test
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允許前端跨來源存取（很重要）

# 用記憶體儲存所有分析結果
TASK_RESULTS = {}   # {task_id: {"name":..., "buffer":..., "data":...}}


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "T-test API is running",
        "endpoints": {
            "/upload": "POST - upload CSV/XLSX and run t-test",
            "/download/<task_id>": "GET - download result csv"
        }
    })


@app.route("/independentTtest/upload", methods=["POST"])
def independentTtest_upload():
    if "file" not in request.files:
        return jsonify({"error": "缺少檔案欄位 file"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "檔名為空"}), 400

    # 讀取檔案
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "檔案格式不支援（僅限 CSV/XLSX）"}), 400
    except Exception as e:
        return jsonify({"error": f"讀檔失敗: {e}"}), 400

    # 執行 t-test
    try:
        result_dict = t_test(df)
    except Exception as e:
        return jsonify({"error": f"分析錯誤: {e}"}), 500

    result_df = pd.DataFrame(result_dict).fillna("")

    # 存成 CSV buffer
    buffer = BytesIO()
    result_df.to_csv(buffer, index=False)
    buffer.seek(0)

    # 生成唯一 task_id
    task_id = secrets.token_hex(8)
    download_name = f"t_test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # 儲存任務結果
    TASK_RESULTS[task_id] = {
        "name": download_name,
        "buffer": buffer,
        "data": result_df,
    }

    # 回傳 task_id 給前端
    return jsonify({
        "task_id": task_id,
        "columns": list(result_df.columns),
        "data": result_df.to_dict(orient="records"),
        "download_name": download_name
    })


@app.route("/independentTtest/download/<task_id>", methods=["GET"])
def independentTtest_download(task_id):
    result = TASK_RESULTS.get(task_id)

    if not result:
        return jsonify({"error": "找不到此 task_id 的結果"}), 404

    buffer = result["buffer"]
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=result["name"],
        mimetype="text/csv"
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)
