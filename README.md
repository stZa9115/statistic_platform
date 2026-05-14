# 線上統計檢定平台

芋頭雲粉絲LAB / DR.HSU Taiwan — 課室實踐研究與教師支持系統

---

## 服務架構

| 服務 | 說明 | 對外 Port |
|------|------|-----------|
| `stat_frontend` | nginx 靜態前端 + API Proxy | 8080 |
| `stat_backend` | FastAPI 主後端（t 檢定、ANOVA、存活分析等） | 5581 |
| `hlm_backend` | Flask HLM 後端（多層次模型） | 5001 |

---

## 啟動流程（Docker Compose）

### 前置需求

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 已安裝並執行中

### 啟動

```bash
# 在專案根目錄執行
docker compose up --build -d
```

啟動後開啟瀏覽器：

```
http://localhost:8080
```

### 停止

```bash
docker compose down
```

---

## 本機開發啟動（不用 Docker）

### 後端 — FastAPI

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### 後端 — HLM Flask

```bash
cd backend
python hlm_flask_server.py
```

### 前端

用 VS Code **Live Server** 開啟 `frontend/statisticList.html`，預設在 `http://127.0.0.1:5500`。

> 本機開發時 `frontend/config.js` 中 `laptop = true`，API 會指向 `http://127.0.0.1:5000`。
> HLM 頁面的 API URL 在 Docker 環境下由 nginx proxy 轉導，本機開發時需將 `hlm.html` 內的 `HLM_API_URL` 暫改為 `http://localhost:5001`。

---

## 功能頁面

| 功能 | 路徑 |
|------|------|
| 首頁 | `/statisticList.html` |
| 獨立樣本 t 檢定 | `/subpage/independentTtestnew.html` |
| 成對樣本 t 檢定 | `/subpage/pairedTtest.html` |
| ANOVA | `/subpage/anova.html` |
| Repeated ANOVA | `/subpage/ra.html` |
| 存活分析 | `/subpage/normalSurvival.html` |
| 競爭風險 | `/subpage/competingRisk.html` |
| 重複事件 | `/subpage/recurrentEvents.html` |
| 多層次模型 (HLM) | `/subpage/hlm.html` |

---

## 待辦

- HLM 計算公式補充
- HLM 文字描述說明
- 加入其他統計圖表
