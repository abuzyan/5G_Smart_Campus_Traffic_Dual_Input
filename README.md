# 5G Smart Campus Traffic and Mobility Analytics
## Updated scope: Phase 1–3 with two input modes

This project supports:
- **Live laptop webcam:** WebRTC camera stream with YOLOv8 detection, ByteTrack tracking and line-crossing counts.
- **Uploaded video:** Upload MP4, AVI, MOV or MKV; process frames; view counts; play/download the annotated video.

Supported COCO classes: person, bicycle, car, motorcycle, bus and truck.

> This is a local prototype. It does not yet implement real 5G transmission, MEC deployment, congestion estimation, prediction or alerts.

## 1. Requirements

- Windows 10/11
- Python 3.11 recommended (Python 3.12 may also work)
- Laptop webcam for live-camera mode
- Internet connection for first-time YOLO model download
- Chrome or Edge browser

## 2. Setup in Windows Command Prompt (CMD)

Extract the ZIP. Open CMD in the `5G_Smart_Campus_Traffic_Dual_Input` folder.

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.11 is not installed, install Python 3.11 or 3.12 and use the matching launcher command.

## 3. Run

```cmd
streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## 4. Live camera mode

1. In the dashboard, choose **Live Camera**.
2. Click **START** in the camera widget.
3. Allow camera permission in Chrome/Edge.
4. The video should show detection boxes and tracking IDs.
5. Objects crossing the red horizontal line are counted once per tracking ID.
6. Use **Reset live camera counts** to reset the count.

If the camera does not start:
- Allow camera access for `localhost`.
- Close Windows Camera, Zoom, Teams or any app using the webcam.
- Refresh the page and press START again.

## 5. Uploaded video mode

1. Choose **Upload Video**.
2. Upload a short MP4/AVI/MOV/MKV traffic clip.
3. Press **Process uploaded video**.
4. Wait while the frames are processed.
5. Review the category-wise crossing count and unique tracked-object total.
6. Play or download the annotated output video.

For the first test, use a short 10–60 second clip. Processing time depends on video length, resolution and laptop hardware. Very large videos are resized to a maximum width of 960 pixels during processing.

## 6. How counting works

- A horizontal virtual line is placed at 55% of the video height.
- The centre point of a tracked object is checked from frame to frame.
- If the centre crosses the line, its tracking ID is counted once.
- **Total line crossings** and **unique tracked objects** are reported separately.
- An object that appears in the video but never crosses the line will not increase the line-crossing count.

Tracking IDs may change if an object disappears for a long time or tracking is interrupted. This is normal for a prototype.

## 7. Important testing note

A laptop webcam pointed at the user is useful for testing pedestrian detection. For vehicle detection/counting, use a safe view of a road or upload a traffic video. Small toy vehicles may not be detected reliably by the pretrained COCO model.

## 8. Folder structure

```text
5G_Smart_Campus_Traffic_Dual_Input/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    └── traffic_processor.py
```

## 9. Troubleshooting

### `ModuleNotFoundError`
Activate the virtual environment and run:

```cmd
pip install -r requirements.txt
```

### Uploaded video cannot be opened
Try an MP4 video encoded with H.264 or an AVI video. Some phone videos use codecs that OpenCV cannot read.

### First run is slow
Ultralytics downloads `yolov8n.pt` the first time. Keep the internet connection active until the download finishes.

### Port 8501 is busy

```cmd
streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502`.

### Camera permission issue
Use Chrome/Edge, allow camera access for localhost, close other camera applications, and reload the dashboard.

## 10. Later phases (not included)

- Phase 4: traffic density / congestion estimation and traffic prediction.
- Phase 5: alerts, advanced analytics and 5G/MEC integration.
