@echo off
title JALSAFAYOO AI - Floater Detection Dashboard
cd /d "%~dp0"

echo.
echo ============================================================
echo   JALSAFAYOO AI
echo   Intelligent Water Surface Floater Detection System
echo ============================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [1/2] Creating virtual environment...
    python -m venv venv
    echo [2/2] Installing dependencies...
    venv\Scripts\pip install -r requirements.txt
) else (
    echo Virtual environment found.
)

if not exist "best.pt" (
    echo.
    echo WARNING: best.pt not found in project root.
    echo Place your YOLO model file here before running inference.
    echo.
)

echo Starting server at http://localhost:5000
echo Press Ctrl+C to stop.
echo.

venv\Scripts\python.exe app.py
pause
