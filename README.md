# OptiGrid AI Smart Traffic Control System

OptiGrid is a Streamlit application that analyzes traffic images for four lanes—North, South, East, and West—and recommends which lane should receive a green light. It uses YOLOv8 to detect people and vehicles, a MobileNetV2 model to classify weather, and a configurable score to select a lane and estimate its green-light timer.

## Features

- Upload a separate traffic image for each of the four lanes.
- Detect people, bicycles, cars, motorcycles, buses, and trucks with YOLOv8.
- Classify weather as clear or bad weather (fog and heavy rain are grouped together).
- Adjust traffic load, YOLO confidence, and base timer in the sidebar.
- Optionally override detected weather or inject an emergency lane.
- View annotated images, lane scores, and the final lane and timer recommendation.

## Requirements

- Python 3.10 or later
- The model weight files in the project root:
  - `yolov8n.pt`
  - `weather_mobilenet.pth`

## Installation and Run (Windows PowerShell)

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell prevents virtual environment activation, run the environment's Python directly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL in the terminal (usually `http://localhost:8501`) and may open it in your browser.

## Using the App

1. Upload one JPG, JPEG, or PNG image for each lane.
2. Set any desired options in the sidebar.
3. Select **Analyse Traffic**.
4. Review the detection results and the recommended green lane.

The app requires an image for all four lanes before analysis. Uploaded images are written to `temp_lanes/` during processing; this folder is generated at runtime.

## Project Files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface, traffic detection, scoring, and lane selection |
| `weather_model.py` | Loads the weather classifier and predicts weather from an image |
| `requirements.txt` | Python package dependencies |
| `yolov8n.pt` | YOLOv8 object-detection weights |
| `weather_mobilenet.pth` | MobileNetV2 weather-classification weights |
| `images/` | Sample lane images |
| `temp_lanes/` | Runtime image files created by the app |

Keep both model weight files in the project root when running the app; the code loads them using relative paths.
