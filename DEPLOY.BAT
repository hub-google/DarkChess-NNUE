@echo off
echo ===================================================
echo   自動部署至 GitHub (將觸發 GitHub Pages 更新)
echo ===================================================

echo [1/3] 將所有變更加入 Git...
git add .

echo [2/3] 提交變更...
git commit -m "Auto-deploy via DEPLOY.BAT"

echo [3/3] 推送至 GitHub...
git push origin master

echo.
echo ===================================================
echo   已推送完成！
echo   請至 GitHub 查看 Actions 確認 Pages 更新進度。
echo ===================================================
pause
