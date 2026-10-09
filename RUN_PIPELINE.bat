@echo off
REM AortaVision 3D: ML Metrics Improvement Pipeline
REM Quick-start script for Windows (VS Code terminal or cmd.exe)
REM
REM Usage:
REM   RUN_PIPELINE.bat          (full run, 10 repeats, 50 Optuna trials)
REM   RUN_PIPELINE.bat quick    (quick test, 1 repeat, no Optuna)
REM   RUN_PIPELINE.bat partial  (2 repeats, 20 trials)
REM   RUN_PIPELINE.bat debug    (verbose logging)

setlocal enabledelayedexpansion

cd /d "%~dp0" || exit /b 1

set MODE=%1
if "%MODE%"=="" set MODE=full

if /i "%MODE%"=="quick" (
  echo === Quick Smoke Test (1 repeat, 2 inner folds, no Optuna) ===
  echo Expected runtime: 2-5 minutes
  echo Purpose: Validate all components without heavy computation
  echo.
  python model\notebooks\improve_metrics.py --repeats 1 --inner-folds 2 --trials 0
  goto end
)

if /i "%MODE%"=="partial" (
  echo === Partial Run (2 repeats, 3 inner folds, 20 Optuna trials) ===
  echo Expected runtime: 30-60 minutes
  echo Purpose: Quick assessment of method ranking
  echo.
  python model\notebooks\improve_metrics.py --repeats 2 --inner-folds 3 --trials 20
  goto end
)

if /i "%MODE%"=="debug" (
  echo === Debug Mode (1 repeat, 2 inner folds, verbose logging) ===
  echo Expected runtime: 2-5 minutes
  echo Purpose: Troubleshoot issues with detailed logging
  echo.
  python model\notebooks\improve_metrics.py --repeats 1 --inner-folds 2 --trials 0 -v
  goto end
)

if /i "%MODE%"=="full" (
  echo === Full Nested Repeated-CV Pipeline ===
  echo Parameters: 10 repeats, 5 outer folds, 3 inner folds, 50 Optuna trials
  echo Expected runtime: 6-12 hours (depends on CPU cores^)
  echo Outputs: metrics/improvement_comparison.csv, metrics/improvement_report.md
  echo.
  set /p "confirm=Proceed? (y/n): "
  if /i "!confirm!"=="y" (
    python model\notebooks\improve_metrics.py --repeats 10 --inner-folds 3 --trials 50
  ) else (
    echo Cancelled.
    exit /b 1
  )
  goto end
)

echo Usage: RUN_PIPELINE.bat [quick^|partial^|debug^|full]
echo.
echo   quick   - Smoke test all components (2-5 min^)
echo   partial - 2 repeats, 20 trials (30-60 min^)
echo   debug   - Verbose logging for troubleshooting
echo   full    - Full pipeline (default, 6-12 hours^)
exit /b 1

:end
echo.
echo === Pipeline Complete ===
echo Review results:
echo   - metrics\improvement_report.md (summary ^& decisions^)
echo   - metrics\improvement_comparison.csv (all-methods detail^)
