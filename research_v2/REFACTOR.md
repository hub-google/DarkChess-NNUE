# 專案清理（研究分支）
- 刪除未被正式網頁、Node 共用引擎或 Python 訓練引用的兩套舊 TypeScript 棋盤、舊啟發式自我對弈、R2 Worker/client 以及它們專屬的 root npm/TS 配置。
- 刪除一次性修復觸發器、已停用 push/schedule 專屬 jobs、舊回填腳本及自動刪除 Actions 歷史的邏輯。原版本留在凍結 Git commit，不重寫歷史。
- 正式 train/self-play/deploy 只允許 master；研究分支不寫正式 HF、不升級冠軍。自我對弈預設單輪，連續需明確輸入。
- 正式訓練不再訓練/晉升未被正式引擎使用的 Policy。保留 Python policy 實驗工具及測試作未來對照。
- 修正限時首翻搜尋：按深度迭代，超時返回上次完整結果或合法 fallback，不再讓 BudgetExceeded 直接逃出。
- 保留 Python 獨立裁判、難局面重新分析、實驗搜尋、正式 NNUE 權重及 HF 棋譜；這些尚有依賴或科學比較用途。
- 尚未替換正式搜尋架構。V2 研究需繼續實測，不能將清理或單元測試視為棋力提升。
