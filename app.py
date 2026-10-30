import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from PIL import Image

from weather_model import predict_weather

LANES = ['North', 'South', 'East', 'West']

# YOUR ORIGINAL MULTIPLIER (UNCHANGED)
WEATHER_MULTIPLIER = {
    'clear': 1.0,
    'bad_weather': 1.5
}

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="OptiGrid AI Traffic System", layout="wide")
st.title("🚦 OptiGrid - AI Smart Traffic Control System")

st.info("📸 Upload traffic images for each lane and analyse traffic dynamically.")

# -----------------------------
# SIDEBAR (YOUR CONTROLS)
# -----------------------------
st.sidebar.header("⚙ Control Panel")

weather_override = st.sidebar.selectbox(
    "Weather Override",
    ["Auto Detect", "clear", "bad_weather"]
)

emergency_lane = st.sidebar.selectbox(
    "Inject Emergency",
    ["None"] + LANES
)

traffic_multiplier = st.sidebar.slider(
    "Traffic Load Multiplier",
    0.5, 2.0, 1.0, 0.1
)

base_timer = st.sidebar.slider(
    "Base Timer",
    10, 120, 30
)

conf_thresh = st.sidebar.slider(
    "YOLO Confidence",
    0.10, 0.90, 0.35, 0.05
)

# -----------------------------
# YOLO LOAD (UNCHANGED)
# -----------------------------
@st.cache_resource
def load_yolo():
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    return model

model = load_yolo()

# -----------------------------
# IMAGE UPLOAD UI (FRIEND FEATURE)
# -----------------------------
st.subheader("📂 Upload Lane Images")

upload_cols = st.columns(4)
uploaded = {}

for col, lane in zip(upload_cols, LANES):
    with col:
        f = st.file_uploader(
            f"{lane} Lane",
            type=["jpg", "jpeg", "png"],
            key=f"upload_{lane}"
        )
        uploaded[lane] = f

        if f:
            st.image(f, use_container_width=True)

# Check if all images uploaded
if not all(uploaded[l] for l in LANES):
    st.warning("⚠️ Please upload images for all four lanes.")
    st.stop()

# Analyse button
if not st.button("🚀 Analyse Traffic"):
    st.stop()

# -----------------------------
# LOAD IMAGES FROM UPLOAD
# -----------------------------
sources = {}
temp_paths = {}

os.makedirs("temp_lanes", exist_ok=True)

for lane in LANES:
    img_pil = Image.open(uploaded[lane]).convert("RGB")

    path = f"temp_lanes/{lane.lower()}.jpg"
    img_pil.save(path)

    sources[lane] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    temp_paths[lane] = path

analyses = {}

# -----------------------------
# MAIN PROCESSING LOOP
# (YOUR ORIGINAL LOGIC)
# -----------------------------
for lane in LANES:

    frame = sources[lane]

    vehicles = []
    pedestrians = 0

    results = model(frame, verbose=False)[0]

    classes = {0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck'}

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in classes and conf > conf_thresh:
            bbox = box.xyxy[0].cpu().numpy()
            vtype = classes[cls]

            vehicles.append({'type': vtype, 'bbox': bbox})

            if vtype == 'person':
                pedestrians += 1

    # WEATHER (UNCHANGED)
    detected_weather = predict_weather(temp_paths[lane])

    if weather_override != "Auto Detect":
        weather = weather_override
    else:
        weather = detected_weather

    emergency = True if emergency_lane == lane else False

    # YOUR ORIGINAL SCORE LOGIC
    score = (len(vehicles) * traffic_multiplier + pedestrians)
    score *= WEATHER_MULTIPLIER[weather]

    if emergency:
        score += 100

    analyses[lane] = {
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "weather": weather,
        "emergency": emergency,
        "score": score
    }

# -----------------------------
# DECISION LOGIC (UNCHANGED)
# -----------------------------
scores = [(analyses[l]['score'], l) for l in LANES]
active_lane = max(scores)[1]

timer_assigned = min(
    base_timer + int(len(analyses[active_lane]['vehicles']) * 3),
    120
)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
st.markdown("---")
st.subheader("🛣 Detection Results")

cols = st.columns(4)

for col, lane in zip(cols, LANES):

    with col:

        img = sources[lane].copy()

        for v in analyses[lane]['vehicles']:

            x1,y1,x2,y2 = v['bbox'].astype(int)

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        status = "🟢 GREEN" if lane == active_lane else "🔴 RED"

        st.image(img,use_container_width=True)

        st.markdown(f"""
        **{lane}**  
        Status: {status}  
        Vehicles: {len(analyses[lane]['vehicles'])}  
        Pedestrians: {analyses[lane]['pedestrians']}  
        Weather: {analyses[lane]['weather']}  
        Emergency: {analyses[lane]['emergency']}  
        Score: {round(analyses[lane]['score'],2)}
        """)

# -----------------------------
# FINAL DECISION
# -----------------------------
st.markdown("---")
st.subheader("🚦 Final Decision")

st.success(f"""
### GREEN LANE: {active_lane}
### Timer Assigned: {timer_assigned} seconds
""")

# -----------------------------
# SCORE CHART
# -----------------------------
st.markdown("---")
st.subheader("📊 Traffic Score Analysis")

df = pd.DataFrame({
    "Lane": LANES,
    "Score": [analyses[l]['score'] for l in LANES]
})

st.bar_chart(df.set_index("Lane"))