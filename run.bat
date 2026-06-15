@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================
echo   CityMind - Urban Intelligence System
echo ============================================
echo.
echo   1. Run Terminal Simulation (20 steps)
echo   2. Open GUI
echo   3. Exit
echo.
set /p choice="Enter choice (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo Running terminal simulation...
    python main.py
    if errorlevel 1 (
        echo.
        echo [ERROR] Terminal simulation failed.
    )
    pause
) else if "%choice%"=="2" (
    echo.
    echo Launching GUI...
    python -c "import customtkinter" >nul 2>nul
    if errorlevel 1 (
        echo [INFO] customtkinter is not installed.
        set /p install_ctk="Install it now? (y/n): "
        if /i "!install_ctk!"=="y" (
            python -m pip install customtkinter
        ) else (
            echo Cannot launch GUI without customtkinter.
            pause
            goto :eof
        )
    )
    python main.py --gui
    if errorlevel 1 (
        echo.
        echo [ERROR] GUI launch failed.
        echo Make sure dependencies are installed:
        echo    pip install customtkinter
    )
    pause
) else if "%choice%"=="3" (
    echo Exiting.
) else (
    echo Invalid option. Please choose 1, 2, or 3.
    pause
)

endlocal
