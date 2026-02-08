from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
import json
import shutil
from zipfile import ZipFile
from tempfile import NamedTemporaryFile
from typing import Optional

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

plt.switch_backend('Agg')

@router.post("/survival/upload")
async def upload_survival_analysis(
    # test_name: str, 
    file: UploadFile = File(...),
    group_col: Optional[str] = Form(None)
):
    test_name = 'survival'  # 固定使用 'survival' 測試
    if test_name not in SV_REGISTRY:
        raise HTTPException(status_code=404, detail="未知的存活分析檢定方法")
    
    test = SV_REGISTRY[test_name]
    task_id = str(uuid.uuid4())
    
    xlsx_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    txt_path = os.path.join(RESULT_DIR, f"{task_id}.txt")
    img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    original_name = sanitize_filename(os.path.splitext(file.filename)[0], max_length=30)

    with NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        df = pd.read_excel(tmp_path, engine='openpyxl')
        
        if group_col and group_col.strip() == "":
            group_col = None
            
        # 執行分析
        result = test.run(df, group_col=group_col)
        
        # === 儲存 Excel (包含多個 Sheet) ===
        # 使用 ExcelWriter 來同時寫入 Log-Rank 與 Cox 結果
        with pd.ExcelWriter(xlsx_path) as writer:
            # 1. Log-Rank 結果 (若有)
            if 'logrank_df' in result and not result['logrank_df'].empty:
                result['logrank_df'].fillna("").to_excel(writer, sheet_name="Log-Rank Test", index=False)
            
            # 2. Cox 結果 (若有)
            if 'cox_df' in result and not result['cox_df'].empty:
                result['cox_df'].fillna("").to_excel(writer, sheet_name="Cox Statistics", index=False)
            elif 'cox_df' in result:
                 # 若 Cox 跑失敗但也許有空表，寫入空表以防檔案為空
                 result['cox_df'].to_excel(writer, sheet_name="Cox Statistics", index=False)

        # 儲存圖片
        result['fig'].savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(result['fig']) 
        
        # 儲存報告
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write(result['report_text'])

        # 儲存 Metadata
        with open(meta_path, "w", encoding='utf-8') as f:
            json.dump({ 
                "original_name": original_name, 
                "test_name": test.display_name,
                "type": "survival"
            }, f, ensure_ascii=False)

        # 準備回傳給前端的 JSON 預覽
        sections = []
        if 'logrank_df' in result and not result['logrank_df'].empty:
            lr_df = result['logrank_df'].fillna("")
            sections.append({
                "title": f"Log-Rank Test Result (Group by: {group_col})",
                "columns": list(lr_df.columns),
                "data": list(lr_df.to_dict(orient='records'))
            })
        
        if 'cox_df' in result and not result['cox_df'].empty:
            stats_df = result['cox_df'].fillna("")
            sections.append({
                "title": "Cox Proportional Hazards Test Result",
                "columns": list(stats_df.columns),
                "data": list(stats_df.to_dict(orient='records'))
            })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"分析錯誤: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "task_id": task_id,
        "test": test.display_name,
        # "report_text": result['report_text'],
        "sections": sections 
    }


@router.post("/recurrent_survival/upload")
async def upload_recurrent_events_analysis(
    file: UploadFile = File(...),
    group_col: Optional[str] = Form(None)
):
    print("Recurrent Events Survival Analysis Upload Triggered")
    test_name = 'recurrent_survival'
    if test_name not in SV_REGISTRY:
        raise HTTPException(status_code=404, detail="未知的存活分析檢定方法")
    
    test = SV_REGISTRY[test_name]
    task_id = str(uuid.uuid4())
    
    # 建立臨時檔案路徑
    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    xlsx_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    txt_path = os.path.join(RESULT_DIR, f"{task_id}.txt")
    img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    try:
        # --- 修正處：顯式指定 engine 並確保路徑正確 ---
        # 如果您的檔案是 CSV，這裡需要做判斷，或者強制要求上傳 Excel
        if file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path, engine='openpyxl')
        
        # 執行分析邏輯
        result = test.run(df, group_col=group_col)
        
        # --- 儲存結果到本地 (Excel/PNG/TXT) ---
        # 這部分維持您原本的邏輯，將 result 中的 DataFrame 寫入 xlsx_path
        with pd.ExcelWriter(xlsx_path) as writer:
            # 1. Log-Rank 結果 (若有)
            if 'stats_df' in result and not result['stats_df'].empty:
                result['stats_df'].fillna("").to_excel(writer, sheet_name="stats_df", index=False)
            if 'info_df' in result and not result['info_df'].empty:
                result['info_df'].fillna("").to_excel(writer, sheet_name="info_df", index=False)
            
            # 2. Cox 結果 (若有)
            if 'cox_df' in result and not result['cox_df'].empty:
                result['cox_df'].fillna("").to_excel(writer, sheet_name="Cox Statistics", index=False)
            elif 'cox_df' in result:
                 # 若 Cox 跑失敗但也許有空表，寫入空表以防檔案為空
                 result['cox_df'].to_excel(writer, sheet_name="Cox Statistics", index=False)

        # 儲存圖片
        result['fig'].savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(result['fig']) 

        # 準備回傳給前端的 JSON
        # 確保優先使用 result['sections'] 以便呈現 recurrentEvents.py 的三張表
        sections = result.get('sections', [])
        if not sections:
            # 原本的備用邏輯 (處理 Log-rank 或 Cox)
            if 'cox_df' in result and not result['cox_df'].empty:
                sections.append({
                    "title": "Analysis Result",
                    "columns": list(result['cox_df'].columns),
                    "data": result['cox_df'].fillna("").to_dict(orient='records')
                })

        return {
            "task_id": task_id,
            "test": test.display_name,
            "sections": sections,
            # "report_text": result.get("report_text", "")
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # 在終端印出詳細錯誤
        raise HTTPException(status_code=500, detail=f"分析錯誤: {str(e)}")
    finally:
        # 刪除臨時檔案
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/competing_risk/upload")
async def upload_competing_risks_analysis(
    file: UploadFile = File(...),
    group_col: Optional[str] = Form(None)
):
    # 修正 1: Log 訊息改為正確的名稱
    print("Competing Risks Survival Analysis Upload Triggered")
    
    # 修正 2: 名稱需與 Class 定義的 name 一致 (上一輪程式碼定義為 "competing_risks")
    test_name = 'competing_risks' 
    
    if test_name not in SV_REGISTRY:
        # 如果您的 registry key 是用單數 'competing_risk'，請改回單數，但需確保與 sv_register 的 key 一致
        raise HTTPException(status_code=404, detail=f"未知的存活分析檢定方法: {test_name}")
    
    test = SV_REGISTRY[test_name]
    task_id = str(uuid.uuid4())
    
    # 建立臨時檔案
    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    xlsx_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    txt_path = os.path.join(RESULT_DIR, f"{task_id}.txt")
    img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    try:
        # 讀取檔案
        if file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path, engine='openpyxl')
        
        # 執行分析
        result = test.run(df, group_col=group_col)
        
        # 修正 3: 儲存 Excel 邏輯 (對應 CompetingRisksAnalysis 回傳的三個 DataFrame)
        with pd.ExcelWriter(xlsx_path) as writer:
            # Sheet 1: Coefficients (cox_df)
            if 'cox_df' in result and not result['cox_df'].empty:
                result['cox_df'].fillna("").to_excel(writer, sheet_name="Cause_Specific_Cox", index=False)
            
            # Sheet 2: Model Stats (stats_df)
            if 'stats_df' in result and not result['stats_df'].empty:
                result['stats_df'].fillna("").to_excel(writer, sheet_name="Model_Stats", index=False)
                
            # Sheet 3: Study Info (info_df)
            if 'info_df' in result and not result['info_df'].empty:
                result['info_df'].fillna("").to_excel(writer, sheet_name="Study_Info", index=False)

        # 儲存圖片
        if 'fig' in result:
            result['fig'].savefig(img_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(result['fig']) 
        
        # 儲存報告文字
        if 'report_text' in result:
             with open(txt_path, "w", encoding='utf-8') as f:
                f.write(result['report_text'])

        # 儲存 Meta
        with open(meta_path, "w", encoding='utf-8') as f:
            json.dump({ 
                "original_name": sanitize_filename(os.path.splitext(file.filename)[0], max_length=30), 
                "test_name": test.display_name,
                "type": "competing_risk"
            }, f, ensure_ascii=False)

        # 準備回傳給前端的 JSON
        # CompetingRisksAnalysis 已回傳完整的 'sections'，直接使用即可
        sections = result.get('sections', [])
        
        # 如果 sections 為空 (防呆)，嘗試手動構建
        if not sections and 'cox_df' in result:
             sections.append({
                "title": "Cause-Specific Cox Results",
                "columns": list(result['cox_df'].columns),
                "data": result['cox_df'].fillna("").to_dict(orient='records')
            })

        return {
            "task_id": task_id,
            "test": test.display_name,
            "sections": sections,
            "report_text": result.get("report_text", "")
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"分析錯誤: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# Image, Download, Zip endpoints 保持不變 (因為它們是直接讀取我們剛剛產生的 xlsx 檔)
@router.get("/image/{task_id}")
def get_survival_image(task_id: str):
    img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="圖片不存在")
    return FileResponse(img_path, media_type="image/png")

@router.get("/download/{task_id}")
def download_single_task_zip(task_id: str):
    xlsx_path = os.path.join(RESULT_DIR, f"{task_id}.xlsx")
    img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
    txt_path = os.path.join(RESULT_DIR, f"{task_id}.txt")
    meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")

    if not os.path.exists(xlsx_path):
        raise HTTPException(status_code=404, detail="結果不存在或已過期")

    original_name = "survival"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            original_name = meta.get('original_name', 'survival')

    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name

    try:
        with ZipFile(zip_path, "w") as zipf:
            zipf.write(xlsx_path, arcname=f"{original_name}_stats.xlsx")
            if os.path.exists(img_path):
                zipf.write(img_path, arcname=f"{original_name}_plot.png")
            if os.path.exists(txt_path):
                zipf.write(txt_path, arcname=f"{original_name}_report.txt")

        return FileResponse(zip_path, filename=f"{original_name}_result.zip", media_type="application/zip")
    finally:
        pass

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
                if not os.path.exists(xlsx_path): continue

                meta_path = os.path.join(RESULT_DIR, f"{task_id}.meta")
                original_name = "survival"
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                        original_name = meta.get('original_name', 'survival')
                
                folder = original_name
                zipf.write(xlsx_path, arcname=f"{folder}/{original_name}_stats.xlsx")
                img_path = os.path.join(RESULT_DIR, f"{task_id}.png")
                if os.path.exists(img_path): zipf.write(img_path, arcname=f"{folder}/{original_name}_plot.png")
                txt_path = os.path.join(RESULT_DIR, f"{task_id}.txt")
                if os.path.exists(txt_path): zipf.write(txt_path, arcname=f"{folder}/{original_name}_report.txt")

        return FileResponse(zip_path, filename="All_Survival_Results.zip", media_type="application/zip")
    except Exception as e:
        print(f"Zip Error: {e}")
        raise HTTPException(status_code=500, detail="建立壓縮檔失敗")