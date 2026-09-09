@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "GRADIO_ANALYTICS_ENABLED=False"
set "NO_PROXY=localhost,127.0.0.1,::1,%NO_PROXY%"
rem Windows 证书库导出的 CA 包,供 Python 访问 HuggingFace 时做 TLS 校验
rem (适用于存在 SSL 检查代理、certifi 无法验证证书链的机器)
if exist "%SCRIPT_DIR%\.win_roots.pem" set "REQUESTS_CA_BUNDLE=%SCRIPT_DIR%\.win_roots.pem"
echo 使用 .venv 启动 WebUI,首次启动可能需要耐心等待20秒
".venv\Scripts\python.exe" webui.py
pause
