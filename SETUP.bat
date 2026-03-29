@echo off
echo ============================================
echo  CHILD FACE PREDICTOR - SETUP
echo ============================================
echo.
echo Installing required packages...
pip install deepface opencv-python numpy matplotlib Pillow flask insightface onnxruntime-gpu
echo.
echo Setup complete! Run START.bat to launch.
pause
