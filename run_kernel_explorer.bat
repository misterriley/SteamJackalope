@echo off
echo Starting Jackalope Kernel Explorer...
set PATH=%~dp0venv310\Scripts;%PATH%
venv310\Scripts\streamlit.exe run tools/kernel_explorer.py
pause
