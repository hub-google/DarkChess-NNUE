# DarkChess-NNUE

CPU-only 台灣暗棋研究專案：**NNUE Value + Star1 chance search + self-play + experimental Policy teacher**。

## 目前狀態（2026-10-22）

- [已實作] Python 暗棋核心／隱藏資訊隔離／CPU 有界搜尋；超預算返回標記為 `qd=0` 的合法公開資訊 fallback。
- [已實作] 自我對弈保存 `q`、`v`、`qd`、`pi`（稀疏 Star1 搜尋機率）及合法終局。
- [已實作] 嚴格終局棋譜驗證、binary training cache、Reanalysis、Value NNUE 訓練、Paired SPRT Gate。
- [實驗性] 獨立 498→64→1024 Policy Head 訓練；可在自我對弈探索時以 `POLICY_PRIOR_MIX` 引入 Policy prior，僅會選擇合法動作。
- [待驗證] 正式 15 CPU Runner 長時間產能、完整 HF replay 恢復、外部競賽 Elo、Policy 對實戰的棋力貢獻。
- [尚未完成] Rust/WASM 正式搜尋引擎及瀏覽器同步；**網頁目前不等於正式 Python 引擎的棋力**。

## CPU Actions

| Workflow | 觸發 | 作用 |
| --- | --- | --- |
| `ci.yml` | push / manual | Python correctness tests |
| `pipeline_smoke.yml` | 指定變更 push / manual | 真實 1 局 → cache → Value/Policy 訓練 → Policy 引導 1 局 → SPRT |
| `self_play_trial.yml` | 指定變更 push / manual | 實際兩局完整對弈 + 終局驗證 + Artifact，無 HF 寫入 |
| `benchmark.yml` | 搜尋相關 push / manual | 搜尋與完整棋局 CPU throughput |
| `self_play.yml` | **manual only** | 15 CPU Worker 正式 HF staging |
| `train.yml` | **manual only** | HF 合併 → Value/Policy 訓練 → SPRT → 條件式晉升 |

正式排程目前依先前要求暫停，不會因推送程式自動開始大量運算。啟動 `self_play.yml` / `train.yml` 需要 GitHub Actions 手動 Run workflow，以及可用的 `HF_TOKEN` secret。

## 快速本地測試

```bash
python -m unittest discover -s test -p "test_*.py" -v
SEARCH_MAX_SECONDS=3 SEARCH_NODE_BUDGET_EARLY=500 SEARCH_NODE_BUDGET_MID=1200 SEARCH_NODE_BUDGET_LATE=2500 BATCH_SIZE=1 python -u src/workers/self_play.py
python -u src/training/validate_replay.py --directory output_data --min-games 1
```

安裝需求見 `requirements.txt`。Linux / macOS 使用以上環境變數語法，Windows 請使用 PowerShell 等效設定。

詳細需求與實際流程：`docs/requirements/01_專案目標與架構總覽.md`、`03_訓練資料與分散式儲存.md`、`04_模型訓練與棋力驗證.md`、`05_前端網頁與容錯機制.md`、`07_GitHub_Actions_執行流程與腳本對照.md`。

**研究目標是達到國際頂尖暗棋棋力，沒有任何世界第一的現成驗證。**
