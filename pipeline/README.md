# V2 正式候選架構
release-v2 為發布前驗證分支，正式 master 未替換。五個正式流程：01 原生 CPU 自我對弈（15 台，各 16 局、預設 25ms，獨立裁判、完整 artifact、HF immutable batch 回讀）；02 固定 CPU 訓練；03 25ms/100ms 各 256 組同底牌換邊（1024 局），獨立重算得分、排除訓練底牌、同一權重及程式版本；04 雙門檻通過且 master 未改變才允許晉升；05 原生/WASM/瀏覽器及正式網址驗收。所有任務單輪、手動觸發，不使用週期排程、不自動重跑。01 可在同一次 run 接續 02–04；allow_promotion 預設 false。研究流程保留在 research-v2，正式候選不再有重複研究 workflows。
現階段 240 局固定資料與 96 次更新為探索性已測配方，正式預設配方須依消融實證審核；不因未達顯著而刪除歷史資料。每批新資料以不可變 archive 保存於 HF native_v2/batches，舊 buffer 不被覆寫或清歷史。保留兩類資料，後續採樣依實驗決定。
發布須先確認舊資料消融、Policy/MCTS、正式候選 workflow 驗收及 100ms 引擎對照，才允許由凍結 master 正常快轉。release-v2 的 05 僅建置與本地 Chromium 實機，不發布；master 05 才可 deploy-pages，發布後由獨立 job 開實際網址驗證模型 SHA、WASM 對弈與重新開局。
