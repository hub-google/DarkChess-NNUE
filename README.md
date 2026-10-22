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
| `hf_production_trial.yml` | push 一次／manual | 單 CPU Worker 產一局、驗證、HF 真正上傳並回讀，**不晉升模型** |
| `arena.yml` | 指定變更 push／manual | Champion 與公開資訊 material baseline 成對對弈並輸出統計 |
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

## HF 資料集與正式訓練

2026-10-22 已在原 HF 帳號、原 ID `hub-google/DarkChess-NNUE-Data` 重建缺失的**私人** dataset，並用 GitHub Action 真實對弈完成「上傳 → 回讀 → 終局重播」驗證；**已不存在的舊 replay 尚未復原**。見 [HF bootstrap 驗收](https://github.com/hub-google/DarkChess-NNUE/actions/runs/39755073316)。

本次另提供 `run_full_production_once.yml`：先跑 CPU correctness tests，再以 `workflow_dispatch` 啟動現有 15-worker `self_play.yml` 的**一次**正式資料生成；`train_after_self_play.yml` 只在生產 Self-play 成功結束時觸發一次 `train.yml`。正式自動 cron 仍停用。正式 training 至少要有 **1000 局有效棋譜**，通過配對 SPRT 才能替換 Champion；不足會失敗而非假裝有進步。

注意：完整生產流程的 Action 可能需要數小時；Action 被接受／啟動不等於已訓練成功，更不代表世界排名已獲證明。
## 網頁端機率搜尋（2026-10-22）

`frontend/src/engine/search.ts` 提供以公開盤面估值的 CPU 受限 Star1-style 翻牌機率搜尋，可結合瀏覽器 NNUE；`App.tsx` 已改為使用 `chooseSearchedMove`，不再使用原本只有簡化走法選擇／隨機翻棋的 UI 策略。JS 搜尋與 Python 的完整 Transposition Table、增量 NNUE、Tablebase **並非完全同一核心**，棋力尚須獨立對戰驗證。正式 NNUE+Policy 訓練仍以 Python 實作為準。

## 外部開源引擎對戰

`external_arena.yml` 會抓取固定 commit `9f7fbe849f5d4a2b3358b1b18a31a2cf1bfdcc5c` 的 [MistyBanqi](https://github.com/brianhliou/misty-banqi) MIT 授權 Rust 引擎，編譯為原生 UCI binary，再以兩局同樣隱藏底牌、交換玩家順序的配對比賽測試 NNUE Champion。這是獨立引擎，跟只對戰自身或 material baseline 不同。公共 FEN 不傳任何真實暗牌，並以 `searchmoves` 強制當局合法根動作。MistyBanqi 與本專案長追、和棋實作仍可能不完全相同，若 UCI 回傳非法動作，測試將**失敗而不是悄悄變更棋規**。兩對／四局為端到端 smoke，不是 Elo 信賴區間，也不是世界第一證據。

## 驗證報告與模型部署

正式 `train.yml` 每次 SPRT 會輸出 `sprt_results.json`（配對局數、逐對比分、LLR、決策與參數），由 Action 保存為 artifact。只有 `ACCEPT_H1` 才會提交新 Champion 權重，並透過 GitHub Actions API 主動派送 `deploy_pages.yml` 對外更新網頁；`UNDECIDED` 或 `ACCEPT_H0` 保留既有權重，不假裝日更進步。正式評估每一步上限改為 0.10 秒，200 pairs，目的使 CPU Job 在 6 小時期限內可控。

### 棋力弱點實測與對照實驗

2026-10-22 已完成第一次 Rust MistyBanqi 配對實戰：現任 NNUE Champion 兩對四局僅獲 **0.5/4 分（12.5%）**，可在 GitHub Actions [39757428996](https://github.com/hub-google/DarkChess-NNUE/actions/runs/39757428996) 核對；四局樣本量不代表真實 Elo。搜尋中的 material bootstrap 曾對炮估值與車同級（5:5），已參考有外部競技數據的角色價值排序調整為 **炮 16、車 9**（王30／士14／象11／馬7／兵4）；這不會直接竄改既有 Champion NNUE。外部競技場現在同底牌比較 Champion／修正後純物質評估／75% Champion+25% material 的 Hybrid，**只作實驗，不會因少量比賽自動更換 Champion**。
