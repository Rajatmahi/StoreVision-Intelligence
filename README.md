<div align="center">

# # 🛍️ StoreVision Intelligence

AI-Powered Retail Analytics & CCTV Intelligence Platform

**AI-powered retail analytics that turns CCTV footage into live business intelligence.**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-EE4C2C?style=for-the-badge)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)

</div>

---

## 📌 Project Overview

**StoreVision-Intelligence** StoreVision Intelligence is a real-time retail analytics platform that transforms CCTV events into actionable business intelligence.

The platform combines:

- Computer Vision based shopper tracking
- Queue monitoring and abandonment analytics
- Zone heatmaps and dwell analysis
- Conversion funnel tracking
- POS transaction correlation
- Real-time anomaly detection
- Interactive Streamlit dashboard
- FastAPI analytics backend

The system enables store managers to understand customer behavior, optimize staffing, reduce checkout congestion, and measure store performance without requiring additional hardware.**zero manual counting and zero hardware beyond an existing CCTV camera**.

> Built as a hackathon project demonstrating the power of combining Computer Vision, a REST API backend, and a live BI dashboard in a single, cohesive Python stack.

---

## ✨ Features

## ✨ Features

### 🎥 CCTV Analytics
- Visitor Detection & Tracking
- Entry / Exit Monitoring
- Zone-wise Shopper Movement
- Dwell Time Analytics
- Queue Detection

### 📈 Retail Intelligence
- Revenue Tracking
- Average Basket Size
- Conversion Rate Analytics
- Converted Visitor Tracking
- Queue Abandonment Metrics

### 🔥 Heatmaps
- Zone Popularity Analysis
- Traffic Distribution
- Store Layout Insights

### 🚨 Anomaly Detection
- Queue Build-up Alerts
- Low Conversion Alerts
- Revenue Anomalies
- Store Activity Monitoring

### 📊 Dashboard
- Real-Time KPI Cards
- Funnel Visualization
- Heatmaps
- Alert Feed
- POS Correlation Metrics

### 🤖 Computer Vision Pipeline
- **YOLOv8 Person Detection** — detects only humans, ignoring all other objects
- **ByteTrack Multi-Object Tracking** — assigns persistent IDs to each shopper across frames, surviving occlusion and partial visibility
- **Virtual Line Crossing** — a configurable counting line generates `ENTRY` and `EXIT` events the moment a person crosses it
- **Zone Analytics** — define rectangular store zones (e.g. Cosmetics, Skincare); automatically detect when visitors enter or leave each zone
- **Dwell Time Measurement** — calculates exact millisecond-level time each visitor spends inside every zone

### ⚡ FastAPI Backend
- Schema-validated event ingestion endpoint (`POST /events/ingest`)
- Per-store metrics aggregation (`GET /stores/{store_id}/metrics`)
- Duplicate-safe SQLite writes using `event_id` as a UUID primary key
- Auto-initialising database — no migrations required

### 📊 Live Analytics Dashboard
- **6 KPI cards** — Unique Visitors, Entries, Exits, Zone A, Zone B, Avg Dwell Time
- **Zone popularity bar chart** — ranked by visit frequency
- **Hourly traffic line chart** — ENTRY vs EXIT trends over time
- **Dwell time histogram** — distribution by zone
- **Live events table** — latest 20 raw events with real-time refresh
- **Auto-refresh every 5 seconds** — no page reload, fully live

---

## 📊 Dashboard KPIs

- Visitors
- Revenue
- Conversion Rate
- Average Basket Size
- Queue Abandonment %
- Queue Depth

Additional Panels:

- Active Alerts
- Conversion Funnel
- Zone Heatmap
- POS Correlation Summary
- Top Zones Analysis

## 🏗️ Architecture

```
📷 CCTV Video
      │
      ▼
  YOLOv8 Detection          ← Detects people in every frame
      │
      ▼
  ByteTrack Tracking         ← Assigns stable IDs across frames
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
Entry/Exit Analytics          Zone Analytics + Dwell Time
(line_crossing.py)            (zone_analytics.py)
      │                                  │
      └──────────────┬───────────────────┘
                     │  HTTP POST /events/ingest
                     ▼
              FastAPI Backend
              (app/main.py)
                     │
                     ▼
           SQLite Database
         (data/store_intelligence.db)
                     │
                     ▼
        Streamlit Dashboard
        (dashboard/app.py)
```

For the full interactive Mermaid diagram with component explanations, see [`docs/architecture.md`](docs/architecture.md).

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Computer Vision** | YOLOv8 Nano (`ultralytics`) | Real-time person detection |
| **Object Tracking** | ByteTrack | Persistent multi-person tracking |
| **Video I/O** | OpenCV (`cv2`) | Frame-level video processing |
| **Backend** | FastAPI + Uvicorn | Async REST API with schema validation |
| **Data Validation** | Pydantic v2 | Type-safe event models |
| **Database** | SQLite | Zero-config persistent storage |
| **Dashboard** | Streamlit | Interactive web UI |
| **Charts** | Plotly Express | Interactive analytics charts |
| **Data Processing** | Pandas | DataFrame-based aggregations |
| **Language** | Python 3.13 | Unified stack — no context switching |

---

## 📁 Folder Structure

```text
StoreVision-Intelligence/
│
├── app/                               # FastAPI backend services
│   ├── main.py                        # API entry point & route registration
│   ├── database.py                    # SQLite connection & initialization
│   ├── models.py                      # Pydantic models
│   ├── ingestion.py                   # Event ingestion endpoints
│   ├── metrics.py                     # KPI & business metrics engine
│   ├── funnel.py                      # Conversion funnel analytics
│   ├── anomalies.py                   # Alert & anomaly detection
│   └── health.py                      # Health check endpoint
│
├── pipeline/                          # Computer Vision analytics pipeline
│   ├── detect.py                      # Person detection using YOLO
│   ├── video_detect.py                # Frame-by-frame video processing
│   ├── track_people.py                # Multi-object tracking (ByteTrack)
│   ├── line_crossing.py               # Entry / Exit event generation
│   ├── zone_analytics.py              # Zone visit & dwell-time analytics
│   └── cross_camera.py                # Cross-camera visitor tracking
│
├── dashboard/
│   └── app.py                         # Streamlit analytics dashboard
│
├── data/
│   ├── store_intelligence.db          # SQLite database
│   ├── pos_transactions.csv           # POS transaction data
│   ├── store_layout.json              # Zone coordinates & store layout
│   ├── test_video.mp4                 # Sample CCTV footage
│   ├── tracked_output.mp4             # Tracking output video
│   ├── line_crossing_output.mp4       # Entry/Exit analytics output
│   └── zone_analytics_output.mp4      # Zone analytics output
│
├── docs/
│   ├── architecture.md                # System architecture
│   ├── DESIGN.md                      # Design decisions
│   ├── CHOICES.md                     # Technical choices & tradeoffs
│   └── plan.md                        # Development roadmap
│
├── tests/                             # Future unit/integration tests
│
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker image definition
├── docker-compose.yml                 # Multi-service deployment
├── .gitignore
└── README.md
```


---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- A CCTV video file placed at `data/test_video.mp4`

### 1. Clone the repository
```bash
git clone https://github.com/your-username/StoreVision-Intelligence.git
cd StoreVision-Intelligence
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> YOLOv8 will automatically download the `yolov8n.pt` model weights (~6 MB) on first run.

---

## 🚀 Running the System

Open **three separate terminal windows** in the project root, with the virtual environment activated in each.

### Terminal 1 — FastAPI Backend
```bash
uvicorn app.main:app --reload
```
API will be live at: **http://127.0.0.1:8000**  
Interactive API docs: **http://127.0.0.1:8000/docs**

### Terminal 2 — AI Pipeline
Choose the analytics mode you want to run:

```bash
# Zone analytics with dwell time (recommended for full demo)
python pipeline/zone_analytics.py

# Entry/Exit line crossing only
python pipeline/line_crossing.py
```

### Terminal 3 — Live Dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard will open automatically at: **http://localhost:8501**

---

## 🔌 API Endpoints

### `GET /`
Health check — confirms the API is running.
```json
{ "message": "Store Intelligence API running" }
```

### `GET /health`
Liveness probe for monitoring / Docker health checks.
```json
{ "status": "healthy" }
```

### `POST /events/ingest`
Ingest a batch of visitor events from the CV pipeline.

**Request body:**
```json
[
  {
    "event_id": "550e8400-e29b-41d4-a716-446655440000",
    "store_id": "STORE_001",
    "camera_id": "CAM_ENTRY_01",
    "visitor_id": "VIS_28",
    "event_type": "ENTRY",
    "timestamp": "2026-05-30T10:00:00+00:00",
    "zone_id": null,
    "dwell_ms": 0,
    "is_staff": false,
    "confidence": 0.9,
    "metadata": {}
  }
]
```

**Response:**
```json
{ "status": "success", "events_received": 1, "duplicates_skipped": 0 }
```

### `GET /stores/{store_id}/metrics`
Returns aggregated KPIs for a specific store.

**Example:** `GET /stores/STORE_001/metrics`
```json
{
  "store_id": "STORE_001",
  "unique_visitors": 42,
  "entries": 38,
  "exits": 35,
  "staff_events": 3
}
```

---

## 📊 Dashboard Features

| Widget | Description |
|---|---|
| **Unique Visitors** | Count of distinct `visitor_id` values seen |
| **Total Entries** | Count of `ENTRY` events |
| **Total Exits** | Count of `EXIT` events |
| **Zone A Visitors** | Unique visitors who entered the Cosmetics zone |
| **Zone B Visitors** | Unique visitors who entered the Skincare zone |
| **Avg Dwell Time** | Average seconds spent inside any zone |
| **Zone Bar Chart** | Zone popularity ranked by total visits |
| **Traffic Line Chart** | ENTRY vs EXIT volume per minute over time |
| **Dwell Histogram** | Distribution of dwell times, coloured by zone |
| **Events Live Feed** | Last 20 raw events refreshed every 5 seconds |

---

## 🔮 Future Improvements

## 🔮 Future Improvements

- Multi-Camera Support
- Real-Time RTSP Streams
- Customer Re-identification
- Predictive Queue Forecasting
- AI-generated Store Reports
- PostgreSQL Migration
- Cloud Deployment
- Mobile Dashboard

### Computer Vision
- [ ] **Re-identification (Re-ID)** — recognise the same shopper across multiple cameras using appearance embeddings
- [ ] **Staff detection** — classify employees vs customers using uniform colour profiles
- [ ] **Queue detection** — flag zones with more than N people for staffing alerts
- [ ] **Anomaly detection** — alert when visitor counts deviate from historical baselines

### Backend
- [ ] **PostgreSQL migration** — swap SQLite for a production-grade database
- [ ] **WebSocket events** — push events to the dashboard in real-time without polling
- [ ] **Multi-store support** — route events from multiple stores to isolated data partitions
- [ ] **Authentication** — JWT-based API key management per camera/store

### Dashboard
- [ ] **Heatmap overlay** — visualise foot traffic density on a store floorplan image
- [ ] **Conversion funnel** — track the path from Entry → Zone A → Zone B → Exit
- [ ] **AI-generated summaries** — natural language daily briefings using an LLM
- [ ] **Export to PDF** — one-click daily report generation for store managers

### Infrastructure
- [ ] **Docker Compose** — containerise all three services for one-command deployment
- [ ] **RTSP stream support** — replace file input with live camera feeds
- [ ] **Cloud deployment** — deploy to GCP / AWS with managed database

---

## 📄 License

This project was built as a hackathon submission. Feel free to use, modify, and build on it.

---

<div align="center">

Built with ❤️ using Python · YOLOv8 · FastAPI · Streamlit

</div>
