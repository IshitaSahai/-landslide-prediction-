@echo off
echo ====================================================
echo   TerraGuard Landslide Prediction - GitHub Sync (V2)
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
git commit -m "Complete Project with ML models and Light Mode UI" 2>nul

REM Set remote origin
echo [4/5] Setting remote origin...
git remote add origin https://github.com/IshitaSahai/-landslide-prediction- 2>nul
if %errorlevel% neq 0 (
    git remote set-url origin https://github.com/IshitaSahai/-landslide-prediction-
)
git branch -M main

REM Push to GitHub (Force)
echo [5/5] Pushing to GitHub (Enforcing project files)...
echo This will override any existing files on GitHub with your local project.
git push -u origin main -f

echo.
echo ====================================================
echo   SYNC COMPLETE! Please check your GitHub repository.
echo ====================================================
pause
