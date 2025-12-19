import re
import os

def sanitize_filename(name: str, max_length: int = 50) -> str:
    # 1️⃣ 只保留檔名（防路徑穿越）
    name = os.path.basename(name)

    # 2️⃣ 移除副檔名
    name = os.path.splitext(name)[0]

    # 3️⃣ 將空白轉成底線
    name = name.replace(" ", "_")

    # 4️⃣ 移除不安全字元（保留中文、英文、數字、_、-）
    name = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", name)

    # 5️⃣ 壓縮連續底線
    name = re.sub(r"_+", "_", name).strip("_")

    # 6️⃣ 限制長度
    return name[:max_length] if name else "file"
