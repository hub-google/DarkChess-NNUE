# DarkChess-NNUE

CPU-only 台灣暗棋研究專案：**NNUE Value + Star1 chance search + self-play + experimental Policy teacher**。

## 目前狀態（2026-10-23）

- [已實作] Python 暗棋核心／隱藏資訊隔離／CPU 有界搜尋；超預算返回標記為 `qd=0` 的合法公開資訊 fallback。
- [已實作] 自我對弈保存 `q`、`v`、`qd`、`pi`（稀疏 Star1 搜尋機率）及合法終局。
- [已實作] 嚴格終局棋譜驗證、binary training cache、Reanalysis、Value NNUE 訓練、Paired SPRT Gate。
- [實驗性] 獨立 498→64→1024 Policy Head 訓練；可在自我對弈探索時以 `POLICY_PRIOR_MIX` 引入 Policy prior，僅會選擇合法動作。
- [已驗證] 自動 Watchdog 成功派送連續模式 4 CPU Worker 正式 Self-play；HF 私人 replay 真實上傳與下載正常。15 Worker 長時測試有 13 名成功，2 名末尾 timeout 的零新檔案誤報已修復。
- [待驗證] 下一輪結束的接力實測、1000 局以上正式 NNUE 訓練與 SPRT 結果、Policy 對實戰的棋力貢獻；外部對戰顯示現任 Champion 尚弱。
- [尚未完成] Rust/WASM 正式搜尋引擎及瀏覽器同步；**網頁目前不等於正式 Python 引擎的棋力**。

## CPU Actions

| Workflow | 觸發 | 作用 |
| --- | --- | --- |
| `ci.yml` | push / manual | Python correctness tests |
| `pipeline_smoke.yml` | 指定變更 push / manual | 真實 1 局 → cache → Value/Policy 訓練 → Policy 引導 1 局 → SPRT |
| `self_play_trial.yml` | 指定變更 push / manual | 實際兩局完整對弈 + 終局驗證 + Artifact，無 HF 寫入 |
| `benchmark.yml` | 搜尋相關 push / manual | 搜尋與完整棋局 CPU throughput |
| `self_play.yml` | 接力 workflow_dispatch | 自動 4 CPU Worker × 55 分鐘；完成立刻派送下一輪；手動 15 Worker |
| `train_after_self_play.yml` | 每輪 Self-play 成功完成 | 距上次正式 train 超過六小時才自動派送訓練 |
| `self_play_watchdog.yml` | 每小時 + 第一次 push / manual | 若無執行中 Self-play，恢復連續 4 Worker 流程 |
| `monthly_activity.yml` | 每月 1 日排程 / manual | 空 Commit 維護 Actions 排程活動 |
| `hf_production_trial.yml` | push 一次／manual | 單 CPU Worker 產一局、驗證、HF 真正上傳並回讀，**不晉升模型** |
| `arena.yml` | 指定變更 push／manual | Champion 與公開資訊 material baseline 成對對弈並輸出統計 |
| `train.yml` | 每日 03:41 UTC / manual | HF 合併 → 未滿千局正常略過 → Value/Policy 訓練 → SPRT → 晉升 |

已啟用工作完成即接力的 GitHub Actions 自我對弈與每小時故障恢復，訓練另以事件觸發（六小時限一次）和每日 cron 備援。私人 HF 資料仍需 `HF_TOKEN` Secret；GitHub 排隊、平台限額或停用仍可能造成中斷。

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

另保留 `run_full_production_once.yml`：先跑 CPU correctness tests，再以 `workflow_dispatch` 啟動現有 15-worker `self_play.yml` 的**一次**加速資料生成。常態任務採 4 Worker × 55 分鐘完成即接力，正式 training 以六小時間隔事件觸發、每天一次備援 cron；至少要有 **1000 局有效棋譜**，不足就正常略過並繼續累積。通過配對 SPRT 的 Challenger 才能取代 Champion。

注意：完整生產流程的 Action 可能需要數小時；Action 被接受／啟動不等於已訓練成功，更不代表世界排名已獲證明。
## 網頁端機率搜尋（2026-10-22）

`frontend/src/engine/search.ts` 提供以公開盤面估值的 CPU 受限 Star1-style 翻牌機率搜尋，可結合瀏覽器 NNUE；`App.tsx` 已改為使用 `chooseSearchedMove`，不再使用原本只有簡化走法選擇／隨機翻棋的 UI 策略。JS 搜尋與 Python 的完整 Transposition Table、增量 NNUE、Tablebase **並非完全同一核心**，棋力尚須獨立對戰驗證。正式 NNUE+Policy 訓練仍以 Python 實作為準。

## 外部開源引擎對戰

`external_arena.yml` 會抓取固定 commit `9f7fbe849f5d4a2b3358b1b18a31a2cf1bfdcc5c` 的 [MistyBanqi](https://github.com/brianhliou/misty-banqi) MIT 授權 Rust 引擎，編譯為原生 UCI binary，再以兩局同樣隱藏底牌、交換玩家順序的配對比賽測試 NNUE Champion。這是獨立引擎，跟只對戰自身或 material baseline 不同。公共 FEN 不傳任何真實暗牌，並以 `searchmoves` 強制當局合法根動作。MistyBanqi 與本專案長追、和棋實作仍可能不完全相同，若 UCI 回傳非法動作，測試將**失敗而不是悄悄變更棋規**。兩對／四局為端到端 smoke，不是 Elo 信賴區間，也不是世界第一證據。

## 驗證報告與模型部署

正式 `train.yml` 每次 SPRT 會輸出 `sprt_results.json`（配對局數、逐對比分、LLR、決策與參數），由 Action 保存為 artifact。只有 `ACCEPT_H1` 才會提交新 Champion 權重，並透過 GitHub Actions API 主動派送 `deploy_pages.yml` 對外更新網頁；`UNDECIDED` 或 `ACCEPT_H0` 保留既有權重，不假裝日更進步。正式評估每一步上限改為 0.10 秒，200 pairs，目的使 CPU Job 在 6 小時期限內可控。

### 棋力弱點實測與對照實驗

2026-10-22 已完成第一次 Rust MistyBanqi 配對實戰：現任 NNUE Champion 兩對四局僅獲 **0.5/4 分（12.5%）**，可在 GitHub Actions [39757428996](https://github.com/hub-google/DarkChess-NNUE/actions/runs/39757428996) 核對；四局樣本量不代表真實 Elo。搜尋中的 material bootstrap 曾對炮估值與車同級（5:5），已參考有外部競技數據的角色價值排序調整為 **炮 16、車 9**（王30／士14／象11／馬7／兵4）；這不會直接竄改既有 Champion NNUE。外部競技場現在同底牌比較 Champion／修正後純物質評估／75% Champion+25% material 的 Hybrid，**只作實驗，不會因少量比賽自動更換 Champion**。

## 網頁訓練統計與自動發布

`publish_training_status.yml` 在 Self-play 或正式訓練 workflow 完成時更新 `frontend/public/training-status.json`；只掃描 HF Dataset `staging` 子樹與單一 canonical replay，而不是歷史資料全根目錄。若統計 JSON 真正改變，GitHub Actions 會 commit 然後主動 dispatch `deploy_pages.yml`，玩家看到的是有 `updatedAt` 時戳的公開彙總，而非直接讀私人 HF Dataset。Self-play 持續接力／watchdog 和 train 事件派送已啟用。

### 擴大外部配對測試

已成功跑過同底牌兩對四局的外部基準，得分率：NNUE Champion 12.5%、修正後純 material 50%、hybrid 25%。因只有四局不採用這組結果直接更換 Champion。已擴大 `external_arena.yml` 至 **每候選方案 20 對、40 局**（三種共 120 局），保留相同 seed、相同 CPU 節點及時間上限的試驗。此舉對照修正後 bootstrap 是否能有較穩定收益，並避免用自我對弈 Elo 取代外部衡量。

## 自我對弈與訓練演進（2026-10-22 歷史設定，最新版見 2026-10-23 修正）

**以下每六小時自動啟動的描述為舊方案，已被下文 2026-10-23 完成即接力＋watchdog 取代。** 原先 `self_play.yml` 設定 `17 */6 * * *` UTC GitHub Actions schedule。每六小時的自動循環使用 4 個 CPU Worker，每人約 55 分鐘；手動模式仍可使用原先 15 個 Worker 與長時段測試。GitHub 可能延遲或取消 cron 工作，當 Repo 未提供執行配額或 HF Token 失效時也不會偷偷假報成功。

```text
GitHub scheduled self_play.yml (4 CPU workers × 55 minutes, every 6h)
    -> public-information Star1 self-play vs champion / archived / material baseline
    -> complete terminal replays -> private HF staging
    -> workflow_run completion triggers train_after_self_play.yml
    -> GitHub workflow_dispatch triggers train.yml
    -> consolidate HF staging with cutoff and dedup
    -> download private HF replay using HF_TOKEN
    -> replay_gate: < 1000 games => successful SKIP; next scheduled cycle accumulates
                    >= 1000 games => validate cache, reanalyse hard positions
    -> train Value NNUE + experimental Policy
    -> paired SPRT (only ACCEPT_H1 can promote)
    -> on promotion: archive old model -> commit new model -> deploy Pages
    -> next 6h scheduled self-play automatically uses latest champion
```

每六小時的 schedule 是自動的循環來源，因此不必在訓練完成時同步啟動另一批 15 Worker，也不會遞迴爆量啟動。前者是一個**架構已串接的自動執行能力**；「棋力是否提升」仍取決於實際 SPRT 結果及外部競技場，沒有數學上的必然進步保證。

`monthly_activity.yml` 每月建立一筆**空 Commit（不增加檔案）**，避免 GitHub 長期沒有儲存庫活動後將 schedule 自動停用。這不會啟動新一輪生產訓練。`autonomous_loop_acceptance.yml` 對 GitHub 的排程契約、HF 真實存取與少於 1,000 局跳過訓練做回歸測試。原先一次性 15 Worker Action 不會被重複派送。

**自动循环实际验收**：`autonomous_loop_acceptance.yml` 已增加一次真实 `train.yml` dispatch（不额外启动 15 Worker），用来验证既有 GitHub Token 权限、HF staging 合并、私人 replay 下载与 `ready=false` 的安全略过行为。低于 1000 局时绝不训练或晋升；达到门槛会照正式流程执行，不以本次验收为由绕过 SPRT。

### 舊版 CPU 配額設計（已由 2026-10-23 方案取代）

由於訓練比產生一小批棋譜更昂貴，最終排程改為**每天四次 4 Worker／55 分鐘自我對弈 + 每天一次 03:41 UTC 訓練**，避免每次爬取都重新跑 3 個 epoch；手動 15 Worker 完成後仍會額外觸發一次訓練。Train Workflow 使用 cutoff 保護同時上傳的 staging；不足 1000 局正常略過。若比賽 512 步尚未終局，完整對局資料不允許偽造和局：Self-play 略過不完整的一局，SPRT 在配對層重抽整組底牌，超出重試額度直接失敗、不晉升。

## 2026-10-23 連續自我對弈修正（不再每六小時乾等）

舊版為每六小時啟動一次（只跑約 55 分鐘），且 `train_after_self_play.yml` 錯誤排除了 `schedule`，所以幾千局 staging 棋譜沒有自動進入正式訓練。另舊 15 Worker 的失敗是最後不完整的一局被 timeout 後，先前已上傳成功的 worker 再次嘗試零檔案 upload 而誤判失敗（13/15 成功）。

現改為 `self_play_watchdog.yml` 啟動 `self_play.yml -f continuous=true`（4 Worker、55 分鐘），**本輪所有 Worker 成功結束便由 `continue-next-cycle` job 立即 `workflow_dispatch` 下一輪**，不依賴六小時 cron；若本輪失敗或未成功派送，watchdog **每小時 23 分 UTC** 檢查有無正在跑／排隊的工作，沒有才恢復。不能宣稱 GitHub runner 100% 零空檔：安裝環境、API 派送與排隊仍有時間，配額或平台停用時需排除平台限制。手動 15 Worker 仍是一次性工作，不會意外放大為 15 Worker 無窮遞迴。

`train_after_self_play.yml` 改為**每次成功完成自我對弈都檢查**，若過去 6 小時無正式 `train.yml` 啟動，就派送一次，保留每日獨立訓練 cron 作為備援；不足 1000 局正常累積，足夠才重訓、SPRT 比較並且只有 H1 才晉升。最近一次公開資料約有 2,902 局（可能有重複待合併），因此需要實際觸發正式訓練確認。已完成的末尾批次與中途 timeout 不會造成虛假的失敗或虛假的勝局。
