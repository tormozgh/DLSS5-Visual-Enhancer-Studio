@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
title DLSS 5 Visual Enhancer — NVIDIA RTX Studio
cls

set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"

:: Check if portable embedded python is present; otherwise fall back to system python
if exist "%~dp0bin\python-3.13.15-embed-amd64\python.exe" (
    set "PYTHON_EXE=%~dp0bin\python-3.13.15-embed-amd64\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%~dp0main_gui.py"
if errorlevel 1 pause
