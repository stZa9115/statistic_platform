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
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    original_name = os.path.splitext(file.filename)[0]

    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        df = pd.read_excel(tmp_path)
        result_dict = t_test(df)
        result_df = pd.DataFrame(result_dict).fillna("")
        result_df.to_excel(result_path, index=True)

        # 🔹 存原始檔名
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(original_name)

    finally:
        os.remove(tmp_path)

    return {
        "task_id": task_id,
        "columns": list(result_df.columns),
        "data": result_df.reset_index().to_dict(orient="records")
    }

@app.get("/independentTtest/download/{task_id}")
def download(task_id: str):
    path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="檔案不存在或已過期")

    original_name = "uploaded_file"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            original_name = f.read().strip()

    download_name = f"t_test_result_{original_name}.xlsx"

    return FileResponse(
        path,
        filename=download_name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )