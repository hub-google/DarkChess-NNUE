# 07. GitHub Actions 執行流程與腳本對照

本文件記錄早期一次性模式與新版連續模式。**2026-10-23 現行設定：Self-play 一輪成功便派送下一輪，watchdog 每小時恢復意外停止的作業；訓練每六小時事件觸發（僅有新一輪成功時）與每日 cron 備援；正式模型依 SPRT 門檻安全晉升。** 下方歷史章節如寫了「manual only」一律以第十二節的現行控制器為準。

## 一、 目前有效的 Workflow

| Workflow | 目前觸發方式 | 主要用途 |
| --- | --- | --- |
| `ci.yml` | push master / PR master / manual | Python unit tests |
| `pipeline_smoke.yml` | 指定 training 相關檔案 push / manual | 完整跑 1 局 self-play → cache/reanalysis → tiny value/policy train → paired validation → export |
| `self_play_trial.yml` | 搜尋/對弈程式 push / manual | **真實完成 2 局棋，檢查合法終局、教師機率與棋譜筆數，輸出 artifact；不寫 HF** |
| `hf_production_trial.yml` | 單次 push / manual | 一局真實 CPU 對弈 → 終局驗證 → HF staging 上傳 → 從 HF 回讀再驗證；不晉升模型 |
| `arena.yml` | 指定程式 push / manual | Champion vs material baseline 配對競技場，輸出 JSON 與 artifact |
| `benchmark.yml` | 指定搜尋/模型檔案 push / manual | CPU search benchmark，輸出 artifact |
| `self_play.yml` | 4 Worker 完成即接力／manual | 4 個 CPU Worker × 55 分鐘，完賽即 workflow_dispatch 下一輪；手動 15 Worker 可選 |
| `self_play_watchdog.yml` | 每小時／manual | 本輪失敗或接力丟失時，檢查無 active 工作才重新派送 4 Worker |
| `train_after_self_play.yml` | 每輪 Self-play 成功 | 每六小時最多派送一次正式訓練 |
| `train.yml` | 完賽事件派送／每日備援／manual | consolidate → replay 門檻 → cache/reanalysis → train → SPRT → promotion |
| `deploy_pages.yml` | 前端 push／manual／正式晉升後派送 | 前端測試、build、部署 GitHub Pages |
| `publish_training_status.yml` | workflow_run／manual／設定更新 push | 從 HF 讀取安全統計並更新前端 training-status |
| `reset_replay_data.yml` | **manual + confirmation** | 新規則版本時清除舊 replay/staging |

`train.yml.bak` 只是備份檔，不是 GitHub Actions workflow。

---

## 二、 `self_play.yml`：分散式資料生成

### 現況

- Trigger：`workflow_dispatch`（continuous=true 由 watchdog／前輪接力派送）。
- Matrix：continuous=true 為 `worker_id: 1..4`；單次手動模式為 1..15。
- Runner：`ubuntu-latest`
- Python：3.10
- PyTorch：CPU-only wheel
- Job timeout：355 分鐘
- 單次 generation budget：連續模式 3,300 秒（55 分鐘），手動模式 21,000 秒（5 小時 50 分）
- 單步預設 deadline：3 秒，預設搜尋節點限制 early=500、mid=1200、late=2500；在完整棋局 Benchmark 證明可行前不得恢復高額度。
- 若失去深度一的搜尋結果，產生 `qd=0` 合法走法；它只能當結果學習資料、不可當 Policy 教師。

Self-play 不再靠六小時 cron：本輪成功由 `continue-next-cycle` 自行派送，失敗／丟失時 watchdog 每小時恢復。

### 執行流程

```text
GitHub runner (worker 1..15)
  ↓
src/workers/self_play.py
  ↓ 完成一局就先落地 output_data/*.jsonl.gz
src/workers/upload_batch.py
  ↓ 合成 batch_*.jsonl.gz
Hugging Face staging/worker_${WORKER_ID}/
```

重要原則：

- 搜尋不得偷讀 `hid` / `hidden_pieces`。
- 完成的局先持久化，runner 接近截止時間時仍可把已完成資料上傳；**零完成／零有效棋譜則上傳步驟必須報錯，整個 Worker 應為失敗**，不得假成功。
- 有經晉升的 `models/champion.policy` 時，可在探索步將 15% 已遮罩合法走法的 Policy prior 與 Star1 搜尋策略混合；沒有 checkpoint 時保留原本搜尋策略。
- 連續 Worker 數為 4，手動完整測試仍為 15；合併及 reset 腳本相容歷史上傳目錄。

---

## 三、 `train.yml`：訓練與 Champion Gate

### 現況

Trigger：`workflow_dispatch` 與每日 03:41 UTC 排程。每次成功 Self-play 的 `train_after_self_play.yml` 會檢查過去六小時是否啟動正式訓練，若沒有則呼叫 `train.yml`。

### 真實步驟

1. Checkout + Python 3.10 + CPU-only PyTorch。
2. 跑完整 Python unit tests。
3. 設定 replay cutoff。
4. 執行 `src/training/consolidate_buffer.py`。
5. 從 HF 下載單一 `replay_buffer.jsonl.gz` 到 `datasets/`。
6. 執行 `src/training/build_training_cache.py`：
   - 產生 `training_cache/cache_*.npz`
   - 建立 hard-position candidates
   - 使用目前 champion 做小量 reanalysis
   - 寫入 `reanalysis_overrides.npz`
   - 寫入 `manifest.json`
7. 先由 `replay_gate.py` 檢查 replay 至少有 1,000 局（不足正常略過訓練且下一輪繼續累積）；再嚴格驗證 binary cache 的完整合法棋譜門檻，不足或資料損毀才失敗且不晉升。
8. 執行 `src/training/train.py`，從 champion 續訓產生 `models/challenger.nnue`。
8. 執行 `src/training/policy_train.py`，使用搜尋教師 softmax 訓練獨立 CPU Policy Head；沒有新格式教師資料時略過。
9. 執行 `src/training/sprt_validation.py`：
   - 最多 200 pairs
   - depth 6
   - node budget 1,200
   - 每步 max_seconds 0.25
   - Elo bounds -20 / +20
10. 僅在 validation 設定 `passed=true` 時：
   - 封存舊 champion
   - challenger Value NNUE → champion NNUE；若已有 challenger Policy，亦一起保存為 champion Policy（Policy 尚未參與正式搜尋）
   - 保留近期 archive champions
   - export 前端模型
   - commit + push
11. H0、未決或驗證失敗：保留既有 champion。

### 資料並行安全

Training 與 self-play 使用不同 concurrency group，可以同時執行。consolidation 只處理固定 cutoff 前的 staging snapshot；驗證新 replay buffer 成功後，也只刪除這次已納入的確切檔案，不會清掉較新的並行上傳。

---

## 四、 `pipeline_smoke.yml`：端到端小型驗收

這是目前最重要的 pipeline 回歸測試。當 training 相關程式被 push 時會自動跑，亦可手動執行。

流程：

```text
1 complete self-play game
  ↓
binary cache + small reanalysis
  ↓
1-epoch tiny Value challenger + optional Policy challenger
  ↓
1 policy-guided complete self-play game + strict terminal check
  ↓
1-pair validation
  ↓
export champion.bin / champion.json
```

它驗證的不只是單一函式，而是 replay 格式、board replay、cache、reanalysis、training、SPRT、export 之間能真正串起來。

---

### `self_play_trial.yml`：完整對弈實測

與只測一局的 pipeline smoke 分開，固定 CPU-only 裝置與安全預算，實際執行 **2 局**，並用 `validate_replay.py` 重播每一步確認終局與資料正確性，必須累積至少 2 筆有效棋譜才成功。此 workflow 不會接觸 Hugging Face 權杖，也不會改動真實 replay buffer。

## 五、 `benchmark.yml`：CPU 搜尋效能驗收

觸發：

- 手動
- 搜尋、NNUE、tablebase、self-play、benchmark config 或 champion 相關檔案 push 到 master

執行 `benchmarks/search_benchmark.py`，測量：

- NNUE 評估 throughput
- 固定 early / mid / late position 的 search throughput
- full-forward vs hybrid evaluator
- 完整低預算 self-play game throughput

結果寫成 `benchmark_results.json` 並保留為 30 天 artifact。

---

## 六、 `ci.yml`：Python correctness gate

觸發：

- push master
- pull request → master
- manual

執行：

`python -m unittest discover -s test -p "test_*.py" -v`

目前測試包含 board/search、hidden-information isolation、Star1 vs exact search、tablebase、training cache、reanalysis override、replay consolidation、symmetry augmentation、SPRT 等。

---

## 七、 前端與狀態 Workflow

### `deploy_pages.yml`

目前在前端程式或部署 workflow 修改推送時，或有新 Champion 真正通過 SPRT 時自動派送，亦可手動啟動。

執行：

1. Node.js 22
2. `npm ci`
3. Vitest
4. Vite build
5. upload-pages-artifact
6. deploy-pages

前端相關 push 可直接觸發部署，正式訓練晉升會另外透過 `gh workflow run deploy_pages.yml` 明確派送。

### `publish_training_status.yml`

目前 **workflow_run（正式對弈／訓練完成）或 manual**。讀取 HF Dataset 後，只把可公開的安全統計寫入 `frontend/public/training-status.json`，有變更才 commit 並派送 Pages 部署。

---

## 八、 `reset_replay_data.yml`：規則版本切換工具

只可手動執行，且必須輸入：

`DELETE-OLD-RULESET-DATA`

才會真正刪除：

- `replay_buffer.jsonl.gz`
- `staging/worker_1..20`
- `staging/fresh`

它是破壞性維護工具，不屬於日常 training loop。

---

## 九、 目前資料與模型閉環

```text
[self_play.yml — manual, 15 CPU workers]
        ↓
[HF staging/worker_*]
        ↓
[train.yml — manual]
        ↓
consolidate_buffer.py
        ↓
HF replay_buffer.jsonl.gz  (raw source of truth)
        ↓
build_training_cache.py
        ↓
training_cache/*.npz + reanalysis_overrides.npz
        ↓
train.py
        ↓
models/challenger.nnue
        ↓
paired SPRT
   ├─ fail / undecided → keep champion
   └─ ACCEPT_H1
        ↓
archive old champion
        ↓
models/champion.nnue
        ↓
export frontend model
```

上圖保留的是過去的單輪流程；現行的「完賽接力＋每小時 Watchdog 自動恢復」及訓練事件觸發見第十二節。

## 十、一次性的正式訓練鏈接（2026-10-22）

- `hf_bootstrap.yml`：只有原 HF Dataset 無法存取時，使用既有 `hub-google` 帳號在同名位置建立新的私人 Dataset；通過一局 CPU 對弈、HF upload/download 及終局重播驗證，不復原舊資料。
- `run_full_production_once.yml`：首次 push 或手動觸發，先跑 Python correctness tests，再使用 GitHub API dispatch 已有 `self_play.yml`；15 Worker、最多 5 小時 50 分，向 HF staging 上傳合法完整對局。**這個 workflow 不定時重複執行。**
- `train_after_self_play.yml`：只有 `self_play.yml` 成功完成時，才透過 `workflow_dispatch` 啟動正式 `train.yml`。一個生產 Self-play 工作不成功，就不會自動訓練。
- `train.yml`：consolidate 後檢查 `training_cache/manifest.json` 至少 1,000 局有效終局才開始 Value/Policy 訓練。SPRT 改用每步 0.25 秒、200 pairs、depth 6、1,200 節點預算；未通過 H1 不晉升。
- 所有驗收／啟動都只代表 GitHub Actions 真正啟動或完成，不能當作棋力變強的證明；正式勝負需根據可回溯的配對結果。
- `external_arena.yml`：Rust/MIT MistyBanqi 固定版本的獨立引擎；在 GitHub CPU 編譯，以 UCI 公共盤面和本專案 Champion 同底牌交換先後配對對弈，驗收合法走法及可重現結果，並保留 JSON artifact。

- 正式訓練 `SPRT` 產生 `sprt_results.json` 保留於 GitHub Artifact；只有 `ACCEPT_H1` 才更新 Champion，並自動 `gh workflow run deploy_pages.yml`，使通過驗證的新模型真正反映於 GitHub Pages。每步 CPU deadline 設為 0.10 秒以降低 6 小時 Job 超限風險。

## 十一、真正自動循環（2026-10-22）

- `self_play.yml`：每 6 小時 UTC 第 17 分自動啟動，4 Worker × 55 分鐘，現有 `concurrency: self-play` 保護同時只跑一個矩陣；原本 15 Worker 只留給 `workflow_dispatch` 大型作業。
- `train_after_self_play.yml`：自動 Self-play 成功結束後以 `workflow_run` 發送 `train.yml` 的 `workflow_dispatch`，失敗不啟動訓練；每次的自動作業完成後下一輪由 GitHub cron 觸發，不需人工手動執行。
- `train.yml`：登入 HF Private Dataset 下載 replay；`replay_gate.py` 低於 1000 局時以 `ready=false` **成功略過**昂貴訓練，留待下一次排程；到達 1000 局才進行完整資料合法性驗證、CPU Value/Policy 訓練與 SPRT；H1 才會晉升。跨多個排程累積，不會因訓練門檻未達就停止。
- `monthly_activity.yml`：每月建立空 Git commit（內容不變）維持公共 Repo 活動，降低 GitHub Actions 的 60 天無活動停用排程風險。
- `autonomous_loop_acceptance.yml`：測試上述自動化契約以及能否用原本 HF_TOKEN 存取私有 Dataset。**自動運作不保證必然棋力提升**，仍以有統計意義的 H1 與獨立對手評估。

## 十二、2026-10-23 持續執行控制器（現行正式設定，覆蓋上述舊版描述）

- `self_play_watchdog.yml` 每小時 UTC 23 分執行；沒有活躍或排隊的 self_play workflow 就派送新的 `continuous=true` 工作，首次推送此 workflow 自動啟動第一次。若有任何活躍自我對弈，直接略過避免重複。
- `self_play.yml` 以 workflow_dispatch input `continuous=true` 啟用 4 Worker／55 分鐘，成功結束時 `continue-next-cycle` job dispatch 相同的下一輪；手動不帶參數仍保留 15 Worker／5h50。不得把 `workflow_run` 作為唯一迴圈控制，以免失敗／已跳過時永久停止。
- Worker 若在最後不完整棋局遇到 CPU timeout，但早前成功 upload 一批以上完整 replay，應視為正常收尾；完全零有效棋譜、真正的程式崩潰、HF 上傳失敗仍必須報錯，交給 watchdog 下個檢查週期恢復。
- `train_after_self_play.yml` **不排除 schedule**：每次成功完成自我對弈都檢查上次正式訓練的派送時間，不足六小時略過，超過六小時會派送 `train.yml`；每日 03:41 UTC 定時訓練仍為備援。`train.yml` 需驗證 1,000 有效對局、產生 NNUE Challenger、通過 SPRT H1 才晉升。這是連續自我對弈 + 受控訓練，不是宣稱每一輪都必然變強。
- `test_autonomous_loop.py` 契約測試與 `autonomous_loop_acceptance.yml` 驗證工作檔內的迴圈、watchdog、訓練與私人 HF 權限。平台配額、GitHub 延誤或服務停用不可能用 YAML 保證完全零中斷，watchdog 提供自動復原。
