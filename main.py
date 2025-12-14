# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import time
import threading
import pandas as pd
from stat_code.independentTtest import t_test
from tempfile import NamedTemporaryFile
import shutil

RESULT_DIR = "results"
# EXPIRE_SECONDS = 60 * 60  # 1 小時
EXPIRE_SECONDS = 60  # 5 分鐘

os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def cleanup_worker():
    while True:
        now = time.time()
        for fname in os.listdir(RESULT_DIR):
            path = os.path.join(RESULT_DIR, fname)
            if not os.path.isfile(path):
                continue

            if now - os.path.getmtime(path) > EXPIRE_SECONDS:
                try:
                    os.remove(path)
                    print(f"[CLEANUP] removed {fname}")
                except Exception as e:
                    print("[CLEANUP ERROR]", e)
        time.sleep(10)
        # time.sleep(300)  # 每 5 分鐘掃一次

@app.on_event("startup")
def start_cleanup():
    t = threading.Thread(target=cleanup_worker, daemon=True)
    t.start()


@app.post("/independentTtest/upload")
async def upload(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    result_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")

    # 🔹 1. 先把上傳檔案存成真正的檔案
    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 🔹 2. 再用 pandas 讀「真正的檔案」
        df = pd.read_excel(tmp_path)

        # 🔹 3. 跑你的 t-test
        result_dict = t_test(df)
        result_df = pd.DataFrame(result_dict).fillna("")
        # 🔹 4. 存結果
        result_df.to_excel(result_path, index=True)

    finally:
        # 🔹 5. 清掉暫存上傳檔
        os.remove(tmp_path)

    # with open('./check/return_info', "w") as f:
    #     f.write(str({
    #         "task_id": task_id,
    #         "columns": list(result_df.columns),
    #         "data": result_df.reset_index().to_dict(orient="records")
    #     }))

    return {
        "task_id": task_id,
        "columns": list(result_df.columns),
        "data": result_df.reset_index().to_dict(orient="records")
    }

@app.get("/independentTtest/download/{task_id}")
def download(task_id: str):
    path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="檔案不存在或已過期")

    return FileResponse(
        path,
        filename="t_test_result.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
