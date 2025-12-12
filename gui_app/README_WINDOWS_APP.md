# Microalgae Counter Windows App

This guide explains how to run and package the Microalgae Counter application on Windows.

## Prerequisites

1.  **Install Python 3.10 or newer**: Download from [python.org](https://www.python.org/). Ensure you check "Add Python to PATH" during installation.
2.  **Install Visual C++ Redistributable**: Required for OpenCV and PyTorch.

## Installation

1.  Open a Command Prompt (cmd) or PowerShell in this folder (`gui_app`).
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have a GPU, you might want to install the CUDA version of PyTorch separately, but for this app, the CPU version installed by default is usually sufficient and easier to package.*

## Running the App

To test the application before packaging:

```bash
python main.py
```

## Packaging as .exe (Standalone Application)

To create a single `.exe` file that you can share with others:

1.  Run PyInstaller:
    ```bash
    pyinstaller --noconfirm --onedir --windowed --name "MicroalgaeCounter" --add-data "inference.py;." main.py
    ```
    *   `--onedir`: Creates a folder with the exe and dependencies (starts faster than --onefile).
    *   `--windowed`: Hides the console window.
    *   `--add-data`: Ensures `inference.py` is included.

2.  **Important**: You must manually copy your model file (`best.pt`) into the `dist/MicroalgaeCounter` folder after building, or load it manually in the app.

3.  The output will be in the `dist/MicroalgaeCounter` folder. You can zip this folder and share it. The user just needs to run `MicroalgaeCounter.exe`.

## Troubleshooting

*   **Model not found**: Ensure `best.pt` is in the same folder as the `.exe` or use the "Load Model" button to find it.
*   **Slow startup**: The first time the app runs, it might take a few seconds to unpack libraries.
