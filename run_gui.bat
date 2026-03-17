@echo off
REM Запуск GUI Chaboss Cluster из корня проекта (Windows).
REM Использование: run_gui.bat [адрес_мастера]
REM Пример: run_gui.bat 127.0.0.1:60051

cd /d "%~dp0"
if exist ".myvenv\Scripts\activate.bat" call .myvenv\Scripts\activate.bat
if "%1" neq "" set CLUSTER_MASTER_ADDR=%1
python -m ui.main_window
