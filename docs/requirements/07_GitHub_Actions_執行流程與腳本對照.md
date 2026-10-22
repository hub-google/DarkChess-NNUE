# 07. GitHub Actions 執行流程與腳本對照

本文件以目前 `master` 內實際存在的 `.github/workflows/*.yml` 為準。**Self-play、training、Pages deploy、training-status publish 的自動排程／事件觸發目前都已依使用者要求註解，只保留手動啟動。** 不應從舊文件誤以為它們仍在背景自動跑。

## 一、 目前有效的 Workflow

| Workflow | 目前觸發方式 | 主要用途 |
| --- | --- | --- |
| `ci.yml` | push master / PR master / manual | Python unit tests |
| `pipeline_smoke.yml` | 指定 training 相關檔案 push / manual | 完整跑 1 局 self-play → cache/reanalysis → tiny value/policy train → paired validation → export |
| `self_play_trial.yml` | 搜尋/對弈程式 push / manual | **真實完成 2 局棋，檢查合法終局、教師機率與棋譜筆數，輸出 artifact；不寫 HF** |
| `hf_production_trial.yml` | 單次 push / manual | 一局真實 CPU 對弈 → 終局驗證 → HF staging 上傳 → 從 HF 回讀再驗證；不晉升模型 |
| `arena.yml` | 指定程式 push / manual | Champion vs material baseline 配對競技場，輸出 JSON 與 artifact |
| `benchmark.yml` | 指定搜尋/模型檔案 push / manual | CPU search benchmark，輸出 artifact |
| `self_play.yml` | **manual only** | 15 個 CPU Worker 產生 replay 並上傳 HF staging |
| `train.yml` | **manual only** | consolidate → cache/reanalysis → train → SPRT → promotion |
| `deploy_pages.yml` | **manual only** | 前端測試、build、部署 GitHub Pages |
| `publish_training_status.yml` | **manual only** | 從 HF 讀取安全統計並更新前端 training-status |
| `reset_replay_data.yml` | **manual + confirmation** | 新規則版本時清除舊 replay/staging |

`train.yml.bak` 只是備份檔，不是 GitHub Actions workflow。

---

## 二、 `self_play.yml`：分散式資料生成

### 現況

- Trigger：`workflow_dispatch`
- Matrix：`worker_id: 1..15`
- Runner：`ubuntu-latest`
- Python：3.10
- PyTorch：CPU-only wheel
- Job timeout：355 分鐘
- 單次 generation budget：21,000 秒（5 小時 50 分）
- 單步預設 deadline：3 秒，預設搜尋節點限制 early=500、mid=1200、late=2500；在完整棋局 Benchmark 證明可行前不得恢復高額度。
- 若失去深度一的搜尋結果，產生 `qd=0` 合法走法；它只能當結果學習資料、不可當 Policy 教師。

自動 `0 */6 * * *` cron 仍留在檔案註解中，**目前不會執行**。

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
- 目前正式 Worker 數固定 15；融合與 reset 腳本仍掃描 1~20，只為相容歷史 staging。

---

## 三、 `train.yml`：訓練與 Champion Gate

### 現況

Trigger：`workflow_dispatch`。原本每日 UTC 19:00（台灣次日 03:00）的 cron 已註解，**目前不會每天自動訓練**。

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
7. 檢查 binary cache 至少有 1,000 局有效對弈（防止空白／過小 buffer 污染 Champion）；不足直接失敗，**不**晉升模型。
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

目前 **manual only**。

執行：

1. Node.js 22
2. `npm ci`
3. Vitest
4. Vite build
5. upload-pages-artifact
6. deploy-pages

舊的 push / workflow_run 自動觸發仍保留在註解內，但現在不會因 champion 更新自動部署。

### `publish_training_status.yml`

目前 **manual only**。讀取 HF Dataset 後，只把可公開的安全統計寫入 `frontend/public/training-status.json`，有變更才 commit。

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

目前不應把這張圖理解成「背景會自動一直跑」；生產 self-play / train / deploy / status publish 都需要手動啟動，只有 CI、pipeline smoke 與 benchmark 依各自 push 條件自動執行。

## 十、一次性的正式訓練鏈接（2026-10-22）

- `hf_bootstrap.yml`：只有原 HF Dataset 無法存取時，使用既有 `hub-google` 帳號在同名位置建立新的私人 Dataset；通過一局 CPU 對弈、HF upload/download 及終局重播驗證，不復原舊資料。
- `run_full_production_once.yml`：首次 push 或手動觸發，先跑 Python correctness tests，再使用 GitHub API dispatch 已有 `self_play.yml`；15 Worker、最多 5 小時 50 分，向 HF staging 上傳合法完整對局。**這個 workflow 不定時重複執行。**
- `train_after_self_play.yml`：只有 `self_play.yml` 成功完成時，才透過 `workflow_dispatch` 啟動正式 `train.yml`。一個生產 Self-play 工作不成功，就不會自動訓練。
- `train.yml`：consolidate 後檢查 `training_cache/manifest.json` 至少 1,000 局有效終局才開始 Value/Policy 訓練。SPRT 改用每步 0.25 秒、200 pairs、depth 6、1,200 節點預算；未通過 H1 不晉升。
- 所有驗收／啟動都只代表 GitHub Actions 真正啟動或完成，不能當作棋力變強的證明；正式勝負需根據可回溯的配對結果。
- `external_arena.yml`：Rust/MIT MistyBanqi 固定版本的獨立引擎；在 GitHub CPU 編譯，以 UCI 公共盤面和本專案 Champion 同底牌交換先後配對對弈，驗收合法走法及可重現結果，並保留 JSON artifact。
