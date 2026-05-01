@echo off
cd /d %~dp0
echo %~dp0
set PYTHONPATH=../../..
"../../../venv/Scripts/python.exe" imagej_bridge.py %1 %2 %3 %4 %5 %6 %7