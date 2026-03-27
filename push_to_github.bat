@echo off
echo ====================================================
echo   TerraGuard Landslide Prediction - GitHub Sync
echo ====================================================

REM Initialize Git if not already
if not exist .git (
    echo [1/5] Initializing Git repository...
    git init
) else (
    echo [1/5] Git repository already initialized.
)

REM Add all files
echo [2/5] Adding all project files (ML models, Frontend, Backend)...
git add .

REM Commit changes
echo [3/5] Committing project...
git commit -m "Complete Landslide Prediction Project: Retrained with new dataset and Light Mode UI"

REM Set remote origin
echo [4/5] Setting remote origin...
git remote add origin https://github.com/IshitaSahai/-landslide-prediction- 2>nul
if %errorlevel% neq 0 (
    git remote set-url origin https://github.com/IshitaSahai/-landslide-prediction-
)
git branch -M main

REM Push to GitHub
echo [5/5] Pushing to GitHub (main)...
echo IMPORTANT: A login window may appear. Please follow GitHub's instructions.
git push -u origin main

echo.
echo ====================================================
echo   SYNC COMPLETE! Please check your GitHub repository.
echo ====================================================
pause
