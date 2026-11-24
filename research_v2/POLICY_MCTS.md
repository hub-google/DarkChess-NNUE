# Policy/MCTS 探索
同一冠軍 NNUE，UCT 隨機機率節點、PUCT 合法動作先驗，無底牌存取。先驗由 192 局原生新棋譜訓練、48 局保留驗證，與 value 訓練相同分組，保留局不進訓練。固定每手 25ms，expectiminimax vs UCT、expectiminimax vs PUCT、UCT vs PUCT 各 64 組獨立底牌，總計 384 局，換邊與 Python 裁判。比較分別採 alpha .05/3；沒有通過門檻就不替換已驗證的正式候選引擎。不以未達顯著證明方法永遠無效。搜尋樹最深節點不能冒充完整 minimax 深度。所有資料與模型均為研究 artifact，無正式寫入。
