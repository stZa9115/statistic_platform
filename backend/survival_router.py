from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
import json
import shutil
from zipfile import ZipFile
from tempfile import NamedTemporaryFile

# 引入您獨立出來的存活分析模組
from survival_analysis import SV_REGISTRY
from costomTools import sanitize_filename

# 建立 Router
router = APIRouter(
    prefix="/survival",
    tags=["Survival Analysis"]
)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 設定 Matplotlib 不使用 GUI
plt.switch_backend('Agg')

@router.post("/{test_name}/upload")
async def upload_survival_analysis(test_name: str, file: UploadFile = File(...)):
    if test_name not in SV_REGISTRY:
        raise HTTPException(status_code=404, detail="未知的存活分析檢定方法")
    
    test = SV_REGISTRY[test_name]
    task_id = str(uuid.uuid4())
    
    # 定義檔案路徑
    xlsx_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    txt_path = os.path.join(RESULT_DIR, f"{task_id}.txt")
    img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    original_name = sanitize_filename(os.path.splitext(file.filename)[0], max_length=30)

    with NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        df = pd.read_excel(tmp_path)
        result = test.run(df)
        
        # 1. 儲存 Excel
        stats_df = result['stats_df'].fillna("")
        stats_df.to_excel(xlsx_path, index=False)
        
        # 2. 儲存圖片
        result['fig'].savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(result['fig']) 
        
        # 3. 儲存文字報告
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write(result['report_text'])

        # 4. 寫入 Metadata
        with open(meta_path, "w", encoding='utf-8') as f:
            json.dump({ 
                "original_name": original_name, 
                "test_name": test.display_name,
                "type": "survival"
            }, f, ensure_ascii=False)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"分析錯誤: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 回傳 JSON 給前端
    return {
        "task_id": task_id,
        "test": test.display_name,
        "report_text": result['report_text'], # 讓前端直接顯示
        "sections": [{
            "title": "Cox Proportional Hazards Test Result",
            "columns": list(stats_df.columns),
            "data": list(stats_df.to_dict(orient='records'))
        }]
    }

# 用於前端 <img> 標籤顯示圖片
@router.get("/image/{task_id}")
def get_survival_image(task_id: str):
    img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="圖片不存在")
    return FileResponse(img_path, media_type="image/png")

# [New] 單一任務下載 (回傳 ZIP)
@router.get("/download/{task_id}")
def download_single_task_zip(task_id: str):
    xlsx_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    if not os.path.exists(xlsx_path):
        raise HTTPException(status_code=404, detail="結果不存在或已過期")

    # 讀取 Metadata 以取得原始檔名
    original_name = "survival"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            original_name = meta.get('original_name', 'survival')

    # 建立暫存 ZIP
    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name

    try:
        with ZipFile(zip_path, "w") as zipf:
            # 加入 Excel
            zipf.write(xlsx_path, arcname=f"{original_name}_stats.xlsx")
            
            # 加入圖片
            img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
            if os.path.exists(img_path):
                zipf.write(img_path, arcname=f"{original_name}_plot.png")
            
            # 加入報告
            txt_path = os.path.join(RESULT_DIR, f"{task_id}.txt")
            if os.path.exists(txt_path):
                zipf.write(txt_path, arcname=f"{original_name}_report.txt")

        return FileResponse(
            zip_path,
            filename=f"{original_name}_result.zip",
            media_type="application/zip"
        )
    finally:
        pass # Temp file cleanup strategy needs to be handled carefully in prod

# [New] 批次下載 (回傳 ZIP，內部依資料夾分類)
@router.post("/download_zip")
def download_bulk_zip(task_ids: list[str] = Body(...)):
    if not task_ids:
        raise HTTPException(status_code=400, detail="未選擇任何項目")

    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name

    try:
        with ZipFile(zip_path, "w") as zipf:
            for task_id in task_ids:
                xlsx_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
                if not os.path.exists(xlsx_path):
                    continue

                meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")
                original_name = "survival"
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                        original_name = meta.get('original_name', 'survival')

                # 建立資料夾結構: original_name/filename
                folder_name = original_name
                
                zipf.write(xlsx_path, arcname=f"{folder_name}/{original_name}_stats.xlsx")
                
                img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
                if os.path.exists(img_path):
                    zipf.write(img_path, arcname=f"{folder_name}/{original_name}_plot.png")
                
                txt_path = os.path.join(RESULT_DIR, f"{task_id}.txt")
                if os.path.exists(txt_path):
                    zipf.write(txt_path, arcname=f"{folder_name}/{original_name}_report.txt")

        return FileResponse(
            zip_path,
            filename="All_Survival_Results.zip",
            media_type="application/zip"
        )
    except Exception as e:
        print(f"Zip Error: {e}")
        raise HTTPException(status_code=500, detail="建立壓縮檔失敗")