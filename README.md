# Face Detection Pre-Processing

This project pre-processes face detection data from Kaggle.

## Environment Setup

### Prerequisites
- **Python 3.11 or 3.12 recommended** (required for DeepFace/TensorFlow)
- **Note**: 
  - Python 3.14 is not supported by TensorFlow (required for DeepFace)
  - If using Python 3.14, the code will automatically fall back to OpenCV DNN face detection
  - For DeepFace features (face analysis, advanced detection), use Python 3.11 or 3.12
  - For basic face detection, Python 3.14 works with OpenCV DNN fallback

### Windows
Run the setup script:
```bash
setup_env.bat
```

### Linux/Mac
Run the setup script:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

### Manual Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   
   **Windows (Command Prompt):**
   ```bash
   venv\Scripts\activate.bat
   ```
   Or use the helper script:
   ```bash
   activate_venv.bat
   ```
   
   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   Or use the helper script:
   ```powershell
   .\activate_venv.ps1
   ```
   
   **Note**: If PowerShell gives an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   
   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Plotting and visualization
- seaborn: Statistical data visualization
- opencv-python: Computer vision and image processing
- kagglehub: Kaggle dataset download utility

## Activating the Environment

After setup, activate the virtual environment:

**Easiest method** - Use the helper script:
- **Command Prompt**: Double-click `activate_venv.bat` or run it from terminal
- **PowerShell**: Run `.\activate_venv.ps1`

**Manual activation**:
- **Command Prompt**: `venv\Scripts\activate.bat`
- **PowerShell**: `.\venv\Scripts\Activate.ps1`
- **Linux/Mac**: `source venv/bin/activate`

You'll know it's activated when you see `(venv)` at the start of your command prompt.

## Usage
After activating the environment, run:
```bash
python pre_processing.py
```