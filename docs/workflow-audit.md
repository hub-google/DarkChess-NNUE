# GitHub Actions 盤點（2026-10-23）

清理前有 21 個有效 YAML workflow，另有不會執行的 `train.yml.bak`。清理後保留 9 個；移除 12 個有效 workflow 和舊備份。訓練、棋譜、模型及評估程式未刪除。保留現有 workflow 名稱，避免破壞 workflow_run 的名稱依賴。

## 保留的 9 個

| 檔案 | 用途及觸發 | 保留原因 |
| --- | --- | --- |
| self_play.yml | 手動或上一輪接力；4 Worker 連續對弈，生成 5h40m 後立即接下一輪；手動重型模式 15 Worker | 正式資料生成 |
| self_play_watchdog.yml | 每小時檢查，沒有活躍 self-play 才派送；可手動執行 | 失敗或漏接力時恢復；不會每小時重啟活躍工作 |
| train_after_self_play.yml | self-play 成功完成後檢查，距前次 train 至少六小時才派送 | 事件驅動訓練橋接 |
| train.yml | 每日 11:41 台灣時間及手動／橋接派送；合併私人 HF 棋譜、千局門檻、Value/Policy 訓練、SPRT、晉升、部署 | 正式訓練主流程與排程備援 |
| publish_training_status.yml | self-play／train 完成、指定變更或手動；彙總 HF 公開統計，變更才提交並部署 | 網頁顯示訓練進度 |
| deploy_pages.yml | frontend／models 變更或手動派送；測試、建置、發布 | 唯一網頁部署入口 |
| monthly_activity.yml | 每月 1 日 11:43 台灣時間，建立空 commit | 防長期無儲存庫活動導致排程停用 |
| ci.yml | master push、PR、手動；Python 測試；PR／手動加前端測試及建置 | 統一正確性測試；push 前端測試已在 Pages 執行 |
| pipeline_smoke.yml | 訓練程式／測試變更或手動；小規模完整對弈、終局驗證、cache、Value/Policy 訓練、SPRT、匯出 | 單元測試之外的整合測試，不會晉升正式模型 |

## 已刪除的 12 個

| 檔案 | 原本用途 | 刪除理由／替代方式 |
| --- | --- | --- |
| arena.yml | Champion 與 material baseline 成對對戰 | 研究工具，並非正式晉升流程；保留 arena.py，正式晉升仍由 train 的 SPRT 執行 |
| benchmark.yml | CPU 搜尋效能測量及 artifact | 不需每次搜尋修改自動開一個 workflow；保留 benchmarks/search_benchmark.py |
| external_arena.yml | 編譯固定版 MistyBanqi，測 Champion/material/hybrid 三種候選 | 研究對照，正式訓練無依賴；保留外部引擎轉接器、測試與評估程式，可按需本地執行 |
| frontend_search_ci.yml | 前端測試與建置 | 與 Pages 重複；PR／手動測試移入 ci.yml |
| self_play_trial.yml | 兩局真實對弈與終局驗證 | pipeline_smoke 已涵蓋真實棋局、Policy 對弈；補上一般棋局嚴格終局驗證 |
| autonomous_loop_acceptance.yml | 全部 Python 測試、HF 連線、千局門檻與正式 train 派送 | 一次性驗收；契約測試仍由 CI 跑，HF 實際讀寫在正式流程執行；避免改 workflow 又派送正式訓練 |
| release_acceptance.yml | Python／前端／真實棋局測試後派送 Pages 及兩個 smoke | 一次性驗收與重複派送，已由 CI、pipeline_smoke、Pages 覆蓋 |
| hf_bootstrap.yml | 恢復缺失 HF dataset，產一局、上傳並回讀 | 舊恢復用途，dataset 已存在；hf_bootstrap.py 保留供修復使用 |
| hf_diagnostics.yml | HF token／dataset 唯讀診斷 | 臨時排查入口；hf_diagnostics.py 保留 |
| hf_production_trial.yml | 產一局，上傳 HF staging 後下載驗證 | 一次性 HF 整合驗收，正式 self-play／train 已執行上傳和下載 |
| reset_replay_data.yml | 確認字串後清空 HF 舊棋規 replay 與 staging | 舊棋規遷移工具，常態訓練不需要；刪除 workflow 不會刪除任何 HF 資料 |
| run_full_production_once.yml | 跑全部測試後派送一次 15 Worker | 一次性啟動包裝；直接手動 self_play.yml 即可 |

另刪除 `train.yml.bak`：舊 TypeScript 流程且跳過正式 SPRT，不會被 Actions 執行；舊版仍可由 Git 歷史取得。

## 按需研究及診斷

先安裝 requirements.txt。以下程式保留，無需獨立 workflow：

```bash
python -u benchmarks/search_benchmark.py
python -u src/training/arena.py --candidate models/champion.nnue --baseline material --pairs 2 --depth 3 --node-budget 250 --max-seconds 0.15 --output arena_results.json
python -u src/training/hf_diagnostics.py
```

HF 診斷需現有 HF_TOKEN。外部對戰請先 clone https://github.com/brianhliou/misty-banqi 並 checkout `9f7fbe849f5d4a2b3358b1b18a31a2cf1bfdcc5c`，再執行：

```bash
cargo build --release -p banqi-engine --manifest-path /path/to/misty-banqi/Cargo.toml
python -u src/training/external_arena.py --misty /path/to/misty-banqi/target/release/banqi-engine --champion models/champion.nnue --pairs 20 --node-budget 400 --max-seconds 0.15 --candidate-mode champion --output external_arena_champion.json
```

material／hybrid 可透過 candidate-mode 選擇，並使用不同 output 檔名。外部對戰不等於正式模型晉升。

刪除定義後，歷史執行紀錄仍可能出現在 Actions 畫面；它們是舊紀錄，不代表仍有排程。此次保留歷史 log 和 artifact 供追溯。

## 2026-10-24 五輪 15+1 HF 單寫入架構

目前 `self_play.yml` 以五階段 reusable workflow `self_play_wave.yml` 運作，每階段 matrix 15 個 CPU Worker 約 50 分鐘，完成後產生獨立 GitHub Artifacts；該波唯一的 `hf-writer` Job (worker 16) 收集所有檔案、重播合法棋譜、去重、以內容雜湊命名唯一批次，單次提交 HF 私人 Dataset staging/worker_16 並下載核對雜湊。HF Writer 失敗時 Artifact 保留 14 天，可重新執行失敗上傳；前 15 個 CPU Worker 不持有 HF_TOKEN、也不對 HF 提交。五輪跑完結束此 RUN 再派送新 RUN；每小時 watchdog 僅偵測停頓，訓練保持每日排程與六小時事件門檻。
