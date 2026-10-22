# DarkChess-NNUE 網頁端

這是 GitHub Pages 的 React + TypeScript + Vite 暗棋公開對弈介面，與 Python 正式 Self-play／NNUE 訓練／SPRT 引擎**不是同一套搜尋程式**。

## 實際功能

- `src/engine/board.ts`：瀏覽器暗棋棋規與簡化走法評分。
- `src/engine/nnue.ts`：讀取 `public/models/champion.bin`、`champion.json` 的 NNUE Value 權重。
- `src/components/Board.tsx`、`Sidebar.tsx`：棋盤和互動元件。
- `src/components/TrainingStatus.tsx`：顯示靜態已發布 `public/training-status.json`，**不是即時 Hugging Face 資料**。

目前不包含正式 Python Star1 搜尋，也不使用最新的 `models/champion.policy`。後續需先完成搜尋核心同源 WASM 和相同計算額度的棋力驗證。

## 本地開發

```bash
npm ci
npm test
npm run build
npm run dev
```

前端部署以 `.github/workflows/deploy_pages.yml` 手動執行，未恢復自動推送部署。後端資料與訓練請參考儲存庫根目錄 `README.md` 與 `docs/requirements/`。
