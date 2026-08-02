@echo off
chcp 65001 >nul
echo ===================================================
echo   鑷嫊閮ㄧ讲鑷?GitHub (灏囪Ц鐧?GitHub Pages 鏇存柊)
echo ===================================================

echo [1/3] 灏囨墍鏈夎畩鏇村姞鍏?Git...
git add .

echo [2/3] 鎻愪氦璁婃洿...
git commit -m "Auto-deploy via DEPLOY.BAT"

echo [3/3] 鎺ㄩ€佽嚦 GitHub...
git push origin master

echo.
echo ===================================================
echo   宸叉帹閫佸畬鎴愶紒
echo   璜嬭嚦 GitHub 鏌ョ湅 Actions 纰鸿獚 Pages 鏇存柊閫插害銆?echo ===================================================
pause
