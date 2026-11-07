@echo off
REM Gera o executavel em "WEPES BOOT GENERATOR V1"
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name "WEPES Boot Generator" --icon "%~dp0assets\icon.ico" ^
  --add-data "%~dp0assets\icon.ico;assets" ^
  --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtWebEngineWidgets ^
  --exclude-module PySide6.QtQml --exclude-module PySide6.QtQuick --exclude-module PySide6.Qt3DCore ^
  --exclude-module PySide6.QtMultimedia --exclude-module PySide6.QtPdf --exclude-module PySide6.QtCharts ^
  --exclude-module tkinter --exclude-module matplotlib --exclude-module scipy ^
  --distpath "WEPES BOOT GENERATOR V1" --workpath build --specpath build ^
  WEPES_Boot_Generator.py
