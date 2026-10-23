@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

rem Copy kernel-metadata.example.json to kernel-metadata.json and set your Kaggle id first.
if not exist kernel-metadata.json (
    echo kernel-metadata.json not found. Copy kernel-metadata.example.json and edit it.
    popd
    exit /b 1
)

call kaggle kernels push -p .
set "KAGGLE_PUSH_EXIT_CODE=%ERRORLEVEL%"

if not "%KAGGLE_PUSH_EXIT_CODE%"=="0" (
    echo.
    echo Kaggle push failed with exit code %KAGGLE_PUSH_EXIT_CODE%.
)

popd
pause
exit /b %KAGGLE_PUSH_EXIT_CODE%
