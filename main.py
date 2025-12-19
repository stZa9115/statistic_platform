# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body

import json
import uuid
import os
import time
import threading
import pandas as pd
import shutil
from zipfile import ZipFile
from tempfile import NamedTemporaryFile

from stat_code import REGISTRY
from costomTools.santize import sanitize_filename

from stat_code.independentTtest import t_test
RESULT_DIR = "results"
# EXPIRE_SECONDS = 60 * 60  
EXPIRE_SECONDS = 600

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
        # time.sleep(300)  

@app.on_event("startup")
def start_cleanup():
    t = threading.Thread(target=cleanup_worker, daemon=True)
    t.start()
    print('REGISTRY:', REGISTRY.keys())


@app.post("/stat/{test_name}/upload")
async def upload_new(test_name: str, file:UploadFile = File(...)):
    if test_name not in REGISTRY:
        raise HTTPException(status_code=404, detail="未知的統計檢定")
    test = REGISTRY[test_name]

    task_id = str(uuid.uuid4())
    result_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    original_name = sanitize_filename(
        os.path.splitext(file.filename)[0], max_length=30
    )

    with NamedTemporaryFile(delete=False, suffix='xlsx') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        df = pd.read_excel(tmp_path)

        result = test.run(df)

        result_df = pd.DataFrame(result).fillna("")
        result_df.to_excel(result_path, index=False)

        with open(meta_path, "w", encoding='utf-8') as f:
            json.dump(
                {
                    "original_name":original_name,
                    "test_name":test.display_name
                },
                f,
                ensure_ascii=False
            )

    finally:
        os.remove(tmp_path)

    return {
        "task_id":task_id,
        "test": test.display_name,
        "columns": list(result_df.columns),
        "data": result_df.to_dict(orient="records")
    }


@app.get("/stat/download/{task_id}")
def download_stat(task_id: str):
    path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="檔案不存在或已過期")
    
    original_name = "uploaded_file"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            original_name = meta['original_name']
            test_name = meta['test_name']

    download_name = f"{original_name}_{test_name}.xlsx"

    return FileResponse(
        path,
        filename=download_name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.post("/stat/download_zip")
def download_zip_stat(task_ids: list[str] = Body(...)):
    if not task_ids:
        raise HTTPException(status_code=400, detail="沒有選擇任何結果")

    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name

    try:
        with ZipFile(zip_path, "w") as zipf:
            for task_id in task_ids:
                result_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
                meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

                if not os.path.exists(result_path):
                    continue  # 已過期或不存在

                # 讀原始檔名
                original_name = "uploaded_file"
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                        original_name = meta['original_name']
                        test_name = meta['test_name']

                filename_in_zip = f"{original_name}_{test_name}.xlsx"
                zipf.write(result_path, arcname=filename_in_zip)

        return FileResponse(
            zip_path,
            filename=f"a.zip",
            media_type="application/zip"
        )

    finally:
        # ⚠️ FileResponse 傳完後再刪（保險起見可延後）
        pass
