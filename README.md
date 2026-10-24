# DarkChess-NNUE

CPU-only 台灣暗棋，NNUE／Star1 搜尋、自我對弈與 SPRT 棋力驗證。

## 2026-10-24：15 個 Worker + 第 16 個專屬 HF 上傳 Job，五輪一 RUN

`self_play.yml` 一個 RUN 執行五輪 `self_play_wave.yml`，每輪 **15 個 CPU Worker 並行產生約 50 分鐘合法棋譜**。每個 Worker 使用 GitHub Artifact 保存完整棋譜，沒有 HF_TOKEN；同輪 **第 16 個 HF Writer** 下載 15 個 Artifacts，逐局驗證終局、依遊戲 ID 去重，合併為一份內容雜湊命名的 `.jsonl.gz`，單獨提交到私人 HF `hub-google/DarkChess-NNUE-Data/staging/worker_16/`，再下載校對 SHA-256。成功才進入下一輪，完成五輪才結束此 RUN 並派送下一個 RUN。

HF 寫入最多重試 7 次；失敗不刪除原始 Artifact（保留 14 天，供重跑上傳 Job 使用），不假報成功。分開的 `train.yml` 每日自動檢查資料與訓練、SPRT 通過才晉升 Champion；五輪成功後也有六小時節流派送。每輪工作約一小時，五輪設計目標遠低於六小時；GitHub 的六小時硬限制其實是單個 Job，而非整個 RUN，平台排隊仍可能延遲。每小時 watchdog 只有在對弈停止時才重新派送五輪 RUN。

舊版研究架構及需求文件留存 `docs/`。
