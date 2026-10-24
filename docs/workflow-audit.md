# GitHub Actions 稽核與端到端驗收（2026-10-24）

## 唯一的正式閉環

1. `self_play.yml`：每個 workflow_dispatch Run 執行五個依序接力的 Wave；每個 Wave 啟動 15 個 CPU 對弈 Worker。每小時 :23（UTC）的 watchdog 僅在沒有新版 Run 時恢復啟動，另有每月的 Actions 活躍度維護。
2. `self_play_wave.yml`：可重用子流程；只讓 Worker 16 存取 `HF_TOKEN`。15 個 Worker 的完整棋譜透過 GitHub Artifacts 移交，Worker 16 驗證合法終局與唯一 ID、集中上傳私人 HF Dataset、重新下載比對 SHA-256。失敗保留 14 天的原始 Artifacts，並在同一 Job 再嘗試一次完整七次重試循環。
3. `publish_training_status.yml`：排程／完成事件彙總 HF 的已整合與待整合棋譜，透過唯一棋局 ID 去重，更新網站統計並觸發 GitHub Pages；另以有序 HF writer 鎖補傳先前失敗的 Artifacts，成功後留下 HF receipts 避免重複補傳。
4. `train.yml`：每日 03:41 UTC 定時訓練，也可由五波結束後的接力流程派送；對 HF staging 做快照、合併／驗證 Replay Buffer、設定最低 1000 局門檻、訓練 Challenger，通過成對 SPRT 才更新 Champion 及網站模型。普通 push 只執行完整 Python 回歸測試，不啟動耗時訓練。
5. `deploy_pages.yml`：只有前端、模型相關變更或明確派送才建置／部署網頁。前端在載入模型時檢查二進位長度與 SHA-256。

正式 workflow：以上 **5 份**。已刪除的歷史 workflow 仍可能保留在 GitHub Actions 的歷史清單中，但不代表仍在排程。

## HF 資料安全

- 新批次：`staging/worker_16/YYYYMMDD/batch_<timestamp>_<digest>.jsonl.gz`
- 彙整後：`replay_buffer.jsonl.gz`
- 補傳成功憑證：`receipts/recovery/<run_id>/<wave>.json`
- 任何 HF 網路／驗證失敗都不得刪除尚未核實的原始 Artifact。
- 一個 Run 五輪中若某 Worker 失敗，其餘輪次仍繼續；但沒有任何成功的 HF Writer 時不得無限制高速接力。
- GitHub Token 推送不一定啟動其他推送式 Actions：正式晉升和統計更新要明確派送 `deploy_pages.yml`。
- 不應從前端直接存取 HF 私人 Dataset 或公開 HF Token。

## 2026-10-24 已實際完成的驗收

- 自動啟動新版 15 個獨立 Worker，第一輪 15/15 成功。
- Worker 16 從全部 GitHub Artifacts 彙整 421 局完整合法棋譜，上傳約 8.4 MB 到 HF，再下載驗證成功。執行 Run：`40077451349`。
- 第一輪結束後自動啟動第二輪的 Worker。
- 訓練 Run `40078183067`：Python 測試、HF Replay Buffer 合併、回讀、最低局數檢查、NNUE Challenger 訓練已通過，SPRT 實際執行中。
- 模型晉升及五輪完成、下一個 Run 接力，只能在各自執行結束後判定，不可將單元測試當成生產成功。

## 實際排查順序

1. GitHub Actions 的 `Continuous self-play and recovery`：檢查五波各 15 Worker，以及每波唯一 `worker 16 HF writer` Job 中的 `[HF Writer] VERIFIED` 行。
2. HF 私有 Dataset 的 `staging/worker_16/` 與 `replay_buffer.jsonl.gz`：確認新增棋局、去重及備份。
3. `Train and promote NNUE`：確認值網路訓練成功、SPRT 通過才執行 `Promote validated challenger`。未通過就保留現有 Champion，不偽稱進化。
4. `Publish training status`：確認 `recover-failed-hf`（需要時）與 `publish`；出現錯誤時保留原始 Artifact 以便下個排程補傳。
5. `deploy_pages.yml`：檢查前端測試、建置與部署結果，再於公開網站核對模型版本及局數。

GitHub Actions: https://github.com/hub-google/DarkChess-NNUE/actions

網站: https://hub-google.github.io/DarkChess-NNUE/
