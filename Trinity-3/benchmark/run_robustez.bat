@echo off
echo 🧪 EJECUTANDO TESTS DE ROBUSTEZ AURORA TRINITY-3
echo ================================================

cd /d "%~dp0"

REM Activar entorno virtual si existe
if exist "..\\.venv\\Scripts\\activate.bat" (
    call "..\\.venv\\Scripts\\activate.bat"
    echo ✅ Entorno virtual activado
)

REM Ejecutar tests de robustez
python test_robustez.py

REM Esperar input del usuario
echo.
echo Presiona cualquier tecla para salir...
pause >nul
