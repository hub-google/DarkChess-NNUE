# 研究競技場原型
目前只供 TypeScript 舊引擎及搜尋設定比較，未接入正式晉升。
固定底牌與首翻、雙方換邊、隨機化配對內的執行順序。引擎輸入移除真實底牌，只有裁判持有。每手相同 wall-time 上限與高節點上限；記錄實際 CPU / wall time、深度零與最大超時。未完成或非法局直接標記實驗無效，不剔除或當成和局。
先編譯原有 shared-build，再執行：
```sh
node arena/match.cjs --pairs 16 --ms 25 --seed 19660217 --output arena/aa.json
python arena/pair_stats.py arena/aa.json --output arena/aa_stats.json
python -m unittest discover -s arena -p 'test_*.py'
```
預設相同模型與搜尋設定為 A/A。16 組只是流程測試，不是棋力證據。實際棋力研究需事前決定樣本數、獨立底牌種子、顯著水準與最低有意義差距。不同引擎的時間限制與超時分布必須可比較。統計採固定樣本的配對五項分布、精確條件符號隨機化檢定及保守區間，不使用原本把同一配對當獨立局的 LLR。保守區間可能需要大量配對，之後再驗證更有效率的序貫方案。固定樣本不可中途依成績停止。模擬 A/A 只驗證統計函式，不代表真實引擎 A/A 已通過。
