@echo off
echo ========================================================
echo Pushing Indian Stock Market Prediction to GitHub...
echo Target: https://github.com/DeepKale16/Indian_Stock_Market_Prediction
echo ========================================================
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Pushed successfully! Check your GitHub profile for 5th Nov contribution!
) else (
    echo [NOTE] If repository was not found, make sure you created:
    echo        https://github.com/new with name "Indian_Stock_Market_Prediction"
)
pause
