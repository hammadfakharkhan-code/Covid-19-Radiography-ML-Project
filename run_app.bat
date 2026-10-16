@echo off
cd /d "%~dp0"
echo Starting PneumoScan AI Web Application...
streamlit run app.py --server.port 8501 --server.headless false
pause
