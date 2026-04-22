---
title: Smart Traffic Violation Detector
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Smart Traffic Violation Detector - Production Grade

An end-to-end, realistic, and production-level system designed to parse video and image data, explicitly track motorcycles, and heuristically flag traffic violations (Triple Riding & Missing Helmets) utilizing YOLOv8 and advanced Intersection-Over-Union bounding logic.

## Upgraded Architecture

- `training/`: Contains PyTorch training scripts for fine-tuning the model.
- `backend/`: FastAPI core hosting highly robust object tracking mechanics and synchronous video parsing.
- `frontend/`: A professional analytical Streamlit dashboard.

## 1. Setting Up the Environment
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. The project uses pre-trained weights (`yolov8n.pt` and `best.pt`). If you wish to re-train the model, ensure your dataset is formatted in YOLO format and run:
   ```bash
   cd training
   python train.py
   ```

## 2. Launching

Run everything locally. The backend utilizes specific metrics checking spatial overlaps against motorcycles matching personnel boundaries individually.

Start Backend:
```bash
python -m uvicorn backend.main:app --reload
```

Start the Dashboard:
```bash
streamlit run frontend/app.py
```

## 3. Deployments

- **Render/Railway (Backend):** Use the preconfigured `Dockerfile` to instantiate the application seamlessly. A `Procfile` is also provided for Heroku.
- **Streamlit Cloud:** Mount this repository directly onto Streamlit Cloud, it will natively detect `frontend/app.py` based on standard routing.
