@echo off
REM KeibaAI nightly auto-collect and retrain (called by Task Scheduler).
REM ASCII only on purpose: cmd.exe reads .bat as system codepage, not UTF-8.
cd /d "%~dp0"
set PYTHONUTF8=1
"C:\Python314\python.exe" keiba.py auto >> "data\auto_run.log" 2>&1
