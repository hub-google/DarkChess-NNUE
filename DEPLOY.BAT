@echo off
echo ===================================================
echo   Auto-deploying to GitHub (Triggers GitHub Pages)
echo ===================================================

echo [1/3] Adding changes to Git...
git add .

echo [2/3] Committing changes...
git commit -m "Auto-deploy via DEPLOY.BAT"

echo [3/3] Pushing to GitHub...
git push origin master

echo.
echo ===================================================
echo   Deployment completed successfully!
echo   Check GitHub Actions for Pages update status.
echo ===================================================
pause
