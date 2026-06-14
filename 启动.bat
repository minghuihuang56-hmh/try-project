@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo    Hotel Booking Analysis Agent
echo ============================================
echo.
echo Starting Streamlit server...
echo.
start http://localhost:8501
python -m streamlit run app.py
pause
