# 07. GitHub Actions 執行流程與腳本對照

本文件以目前 `master` 內實際存在的 `.github/workflows/*.yml` 為準。**Self-play、training、Pages deploy、training-status publish 的自動排程／事件觸發目前都已依使用者要求註解，只保留手動啟動。** 不應從舊文件誤以為它們仍在背景自動跑。

## 一、 目前有效的 Workflow

| Workflow | 目前觸發方式 | 主要用途 |
| --- | --- | --- |
| `ci.yml` | push master / PR master / manual | Python unit tests |
| `pipeline_smoke.yml` | 指定 training 相關檔案 push / manual | 完整跑 1 局 self-play → cache/reanalysis → tiny train → paired validation → export |
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
- 完成的局先持久化，runner 接近截止時間時仍可把已完成資料上傳。
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
7. 執行 `src/training/train.py`，從 champion 續訓產生 `models/challenger.nnue`。
8. 執行 `src/training/sprt_validation.py`：
   - 最多 400 pairs
   - depth 8
   - node budget 20,000
   - Elo bounds -15 / +15
9. 僅在 validation 設定 `passed=true` 時：
   - 封存舊 champion
   - challenger → champion
   - 保留近期 archive champions
   - export 前端模型
   - commit + push
10. H0、未決或驗證失敗：保留既有 champion。

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
1-epoch tiny challenger
  ↓
1-pair validation
  ↓
export champion.bin / champion.json
```

它驗證的不只是單一函式，而是 replay 格式、board replay、cache、reanalysis、training、SPRT、export 之間能真正串起來。

---

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
