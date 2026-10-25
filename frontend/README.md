# 暗棋網站

網站採 React、TypeScript 與 Vite，提供玩家與現任暗棋模型對弈。

搜尋決策與正式自我對弈共用 frontend/src/engine/search.ts 的 DarkChessEngine 類別。網站透過背景執行緒搜尋，不會在瀏覽器執行私人的分散式訓練。

**所有規則、模型、棋譜格式、訓練與故障恢復說明統一維護於[倉庫主 README](../README.md)**。

~~~bash
npm ci
npm test -- --run
npm run build
npm run dev
~~~
