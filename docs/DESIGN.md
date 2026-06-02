# Store Intelligence System — Design Document

> **Hackathon Submission** · Technical design, architectural decisions, tradeoffs, and the reasoning behind every technology choice.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [YOLOv8 — Detection Layer](#2-yolov8--detection-layer)
3. [ByteTrack — Tracking Layer](#3-bytetrack--tracking-layer)
4. [FastAPI — Backend Layer](#4-fastapi--backend-layer)
5. [SQLite — Storage Layer](#5-sqlite--storage-layer)
6. [Streamlit — Dashboard Layer](#6-streamlit--dashboard-layer)
7. [Event Flow](#7-event-flow)
8. [AI-Assisted Decisions](#8-ai-assisted-decisions)
9. [Design Tradeoffs](#9-design-tradeoffs)
10. [Known Limitations](#10-known-limitations)
11. [Group-Entry & Motion-Based State Machine](#11-group-entry--motion-based-state-machine)
12. [Billing Queue Detection & Analytics](#12-billing-queue-detection--analytics)
13. [Graceful Occlusion Recovery](#13-graceful-occlusion-recovery)

---

## 1. System Architecture

The system is intentionally designed as a **linear, decoupled pipeline** where each layer has a single responsibility and communicates with the next layer through a well-defined interface.

```
┌─────────────────────────────────────────────────────────────────┐
│                   COMPUTER VISION LAYER                         │
│                                                                 │
│  Video Frames  →  YOLOv8 Detection  →  ByteTrack IDs           │
│       ↓                                                         │
│  Business Logic (line_crossing.py / zone_analytics.py)         │
│       ↓  Event JSON (HTTP POST)                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      API LAYER                                  │
│                                                                 │
│  FastAPI  →  Pydantic Validation  →  SQLite Write               │
│       ↑  Metrics Query (HTTP GET)                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                   PRESENTATION LAYER                            │
│                                                                 │
│  Streamlit reads SQLite  →  Pandas aggregation  →  Plotly UI    │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | How it's applied |
|---|---|
| **Single Responsibility** | Each Python file does exactly one thing. `database.py` only handles DB connections. `metrics.py` only aggregates data. |
| **Loose Coupling** | The CV pipeline sends HTTP JSON. It doesn't import anything from `app/`. These two halves could run on entirely separate machines. |
| **Fail-Safe Defaults** | Every layer handles errors without crashing downstream layers. The dashboard shows "No data" instead of a traceback. The pipeline continues processing if the API is unreachable. |
| **Incremental Complexity** | The system was built layer by layer, each step verifiable in isolation before wiring it to the next. |

---

## 2. YOLOv8 — Detection Layer

### What it does
YOLOv8 (You Only Look Once, version 8) is a single-pass convolutional neural network that processes an entire video frame in one forward pass and outputs a list of bounding boxes, class labels, and confidence scores.

### Why YOLOv8 was chosen

| Criterion | YOLOv8 | Alternative: Faster R-CNN | Alternative: MediaPipe |
|---|---|---|---|
| **Speed** | 45+ FPS on CPU (Nano) | ~5 FPS on CPU | ~30 FPS |
| **Accuracy (mAP)** | 37.3 (Nano) — 53.9 (Large) | ~40 mAP | Limited to poses |
| **Built-in tracking** | ✅ Yes (`model.track()`) | ❌ No | ❌ No |
| **Ease of use** | `pip install ultralytics` | Complex setup | Moderate |
| **Active maintenance** | ✅ Monthly updates | Stale | ✅ Active |

**YOLOv8 Nano (`yolov8n.pt`)** was specifically chosen for its sub-100ms per-frame inference on CPU hardware. Retail CCTV cameras do not always have GPU access — the Nano model ensures the system runs on commodity hardware that a physical store would actually have.

### Configuration decisions

```python
model.track(frame, classes=[0], conf=0.3, ...)
```

- **`classes=[0]`** — restricts detection to humans only. This eliminates false positives from shopping carts, bags, mannequins, and clothing displays — all of which a retail environment is full of.
- **`conf=0.3`** — a deliberately low confidence threshold. In a tracking context, a low-confidence detection is better than no detection at all. ByteTrack (see section 3) handles low-confidence boxes gracefully, using them only for re-association rather than for creating new tracks.

---

## 3. ByteTrack — Tracking Layer

### What it does
ByteTrack is a multi-object tracking algorithm that assigns a persistent integer ID to each detected person and maintains that ID across video frames even when the person is partially occluded, facing away from the camera, or temporarily out of frame.

### Why ByteTrack was chosen

| Criterion | ByteTrack | Alternative: SORT | Alternative: DeepSORT |
|---|---|---|---|
| **Occlusion handling** | ✅ Excellent (uses low-conf detections) | ❌ Poor | ✅ Good |
| **Re-ID model required** | ❌ No | ❌ No | ✅ Yes (extra GPU model) |
| **ID switches per sequence** | Low | High | Medium |
| **Complexity** | Low | Very Low | High |
| **Integration** | Built into `ultralytics` | Separate install | Separate install |

### The key insight: "Every Detection Box Matters"

Most trackers discard detection boxes below a confidence threshold (e.g., 0.5). If someone turns their back to the camera, YOLO's confidence might drop from 0.85 to 0.32. A standard tracker would drop that box — losing the track — and re-assign a new ID when they turn around.

ByteTrack keeps two sets of boxes:
- **High-confidence** (`>0.5`) → used to create new tracks and update existing high-confidence tracks
- **Low-confidence** (`0.3–0.5`) → used *only* to re-associate with lost tracks, never to create new ones

This means a person who ducks behind a shelf for a second reappears with their **same Track ID** — which is exactly what our entry/exit and zone dwell-time logic depends on.

### `persist=True`

```python
model.track(frame, persist=True, ...)
```

Without this flag, the tracker's internal memory (the Kalman filter state for each track) is reset every call. With `persist=True`, the tracker state lives in the model object and survives across frames. This is the single most important flag for retail analytics — without it, every frame starts fresh and IDs are completely meaningless.

### Appearance-based Re-Identification (Re-ID)

A pure motion-tracker like ByteTrack will eventually lose a person if they exit the camera view for an extended period (e.g., leaving the store and returning 5 minutes later). To solve this without heavy deep-learning Re-ID models, we implemented a custom **Appearance Signature Cache** inside `pipeline/tracker.py`.

When a visitor exits, we extract an HSV color histogram of their torso and save it with a 10-minute TTL. When a new person enters, we compare their color histogram against the cache using Cosine Similarity. If the similarity is above 0.8 and their bounding box size hasn't drastically changed (to differentiate adults and children), we map their new `track_id` back to their original `visitor_id` and emit a `REENTRY` event. This prevents double-counting in the conversion metrics.

---

## 4. FastAPI — Backend Layer

### What it does
FastAPI serves as the central data gateway. It accepts visitor event payloads from the CV pipeline, validates them, and writes them to the database. It also exposes analytics aggregation endpoints that the dashboard queries.

### Why FastAPI was chosen

| Criterion | FastAPI | Alternative: Flask | Alternative: Django REST |
|---|---|---|---|
| **Performance** | Async, ~10× faster than Flask | Sync | Sync |
| **Schema validation** | Built-in (Pydantic) | Manual | django-rest-framework |
| **Auto API docs** | ✅ `/docs` (Swagger) | ❌ | ✅ |
| **Type hints** | Native | Optional | Optional |
| **Learning curve** | Low | Very Low | High |

### Schema-first design

The `Event` Pydantic model in `app/models.py` acts as the **contract** between the CV pipeline and the database. If `line_crossing.py` accidentally sends a string where `dwell_ms` expects an integer, FastAPI returns a `422 Unprocessable Entity` — the bad data never touches the database.

```python
class Event(BaseModel):
    event_id: str          # UUID — primary key, prevents duplicates
    store_id: str
    camera_id: str
    visitor_id: str        # VIS_<track_id>
    event_type: str        # ENTRY | EXIT | ZONE_ENTER | ZONE_EXIT
    timestamp: str         # ISO 8601 UTC
    zone_id: Optional[str] # ZONE_A | ZONE_B | null
    dwell_ms: int          # milliseconds
    is_staff: bool
    confidence: float
    metadata: dict
```

### Duplicate-safe ingestion

The CV pipeline processes the same video file in multiple test runs. Without protection, the database would accumulate duplicate events. The `event_id` is a UUID generated at event-creation time — SQLite's `INSERT OR IGNORE` ensures the same event is only written once, and the API response tells the caller how many duplicates were silently skipped.

---

## 5. SQLite — Storage Layer

### What it does
SQLite is a serverless, file-based relational database. All events are stored in a single file: `data/store_intelligence.db`.

### Why SQLite was chosen

| Criterion | SQLite | Alternative: PostgreSQL | Alternative: MongoDB |
|---|---|---|---|
| **Setup** | Zero config | Requires server | Requires server |
| **Dependencies** | Built into Python | `psycopg2` driver | `pymongo` driver |
| **Performance** | Excellent for <1M rows | Excellent at scale | Good |
| **SQL queries** | ✅ Full SQL | ✅ Full SQL | ❌ No SQL |
| **Hackathon suitability** | ✅ Perfect | Overkill | Overkill |

### Designed to be swappable

The entire database interaction is isolated in `app/database.py`. The connection string, table creation, and query functions are the only things that would change in a production migration to PostgreSQL. The FastAPI routes, Pydantic models, and Streamlit dashboard would require **zero changes**.

```python
# To migrate to PostgreSQL, only this line changes:
conn = sqlite3.connect("data/store_intelligence.db")
# Becomes:
conn = psycopg2.connect("postgresql://user:pass@host:5432/db")
```

### Schema design decisions

- **`event_id TEXT PRIMARY KEY`** — UUIDs prevent duplicate ingestion even if the pipeline reruns
- **`is_staff INTEGER`** — SQLite has no boolean type; `0/1` integers are the standard pattern
- **No foreign keys** — kept intentionally for hackathon simplicity; a production schema would reference a `stores` and `cameras` table

---

## 6. Streamlit — Dashboard Layer

### What it does
Streamlit transforms the SQLite data into an interactive, auto-refreshing analytics dashboard accessible in any web browser, with no frontend code required.

### Why Streamlit was chosen

| Criterion | Streamlit | Alternative: Grafana | Alternative: React + D3 |
|---|---|---|---|
| **Setup** | `pip install streamlit` | Requires server + config | Weeks of dev time |
| **Python native** | ✅ Yes | ❌ No | ❌ No |
| **SQLite support** | ✅ Direct | Needs plugin | Manual |
| **Time to dashboard** | ~1 hour | ~4 hours | Weeks |
| **Hackathon suitability** | ✅ Perfect | Moderate | Impractical |

### Auto-refresh implementation

```python
time.sleep(5)
st.rerun()
```

This is the simplest possible live-refresh implementation. Every 5 seconds Streamlit re-executes the entire script from line 1 — re-running the SQLite query, recalculating KPIs, and redrawing all charts. It is not WebSocket-based real-time, but it is perfectly sufficient for a hackathon demo where events are being generated at a rate of ~10/minute.

### Defensive data loading

The dashboard uses a layered approach to handle bad data without crashing:

1. `try/except` around the SQL query — handles missing DB file or table
2. `errors="coerce"` in `pd.to_datetime()` — converts unparseable timestamps to `NaT`
3. `dropna(subset=["timestamp"])` — removes corrupted rows before any chart is drawn
4. `if df.empty` guard — shows a friendly info message instead of a crash when no data exists

---

## 7. Event Flow

The following diagram traces a single real-world moment — **a shopper walking from the store entrance into the Cosmetics section** — through the entire system.

```
T=0:000ms  Camera records frame 240
           ↓
T=0:012ms  YOLOv8 detects person at (x=412, y=280, w=80, h=180)
           confidence=0.87, class=person
           ↓
T=0:015ms  ByteTrack assigns Track ID = 28
           (matches previous frame's Track ID 28 — same person)
           ↓
T=0:016ms  line_crossing.py:
           prev_x = 430 (right side of frame)
           curr_x = 412 (crossed line_x=500 from right to left)
           → Generates EXIT event
           ↓
T=0:017ms  zone_analytics.py:
           is_inside_zone(412, 370, ZONE_A_COORDS) → True
           Track ID 28 not in visitor_sessions → ZONE_ENTER event
           visitor_sessions[28] = {"zone": "ZONE_A", "enter_time_ms": T}
           ↓
T=0:020ms  send_event() builds JSON payload:
           {
             "event_id": "f47ac10b-58cc-...",
             "visitor_id": "VIS_28",
             "event_type": "ZONE_ENTER",
             "zone_id": "ZONE_A",
             "dwell_ms": 0,
             "timestamp": "2026-05-30T10:00:00.020Z"
           }
           ↓
T=0:022ms  HTTP POST to http://127.0.0.1:8000/events/ingest
           ↓
T=0:024ms  FastAPI receives payload
           Pydantic validates all fields ✅
           ↓
T=0:025ms  database.py executes:
           INSERT OR IGNORE INTO events VALUES (...)
           → 1 row written, 0 duplicates
           ↓
T=0:026ms  API returns: {"status": "success", "events_received": 1}
           ↓
T=5:000ms  Streamlit dashboard auto-refresh triggers
           Pandas reads SQLite → Zone A visitors KPI increments by 1
           Zone bar chart re-renders → ZONE_A bar grows taller
```

---

## 8. AI-Assisted Decisions

Several key design decisions in this system were non-obvious and required computer-vision domain knowledge:

### Decision 1: Feet-based zone detection
**Problem:** Using the bounding box centroid for zone detection causes false positives when a shopper *leans* into a zone without stepping into it (e.g., reaching across a shelf).

**Solution:** We use the **bottom-centre of the bounding box** — the approximate foot position — for all zone collision detection.

```python
y_feet = box[1] + box[3] / 2   # y_center + half-height = bottom of box
is_inside_zone(x_center, y_feet, zone_coords)
```

This is a standard technique in production retail analytics systems. A person standing in Zone B while reaching into Zone A is correctly counted as a Zone B visitor.

### Decision 2: Low confidence threshold for tracking
**Standard tracking:** `conf=0.5` — drop low-confidence detections  
**Our approach:** `conf=0.3` — keep low-confidence detections for ByteTrack

**Reasoning:** In a tracking context, a slightly wrong bounding box is far better than no bounding box. ByteTrack uses low-confidence detections exclusively for re-association with lost tracks — it never creates a new track from a low-confidence detection. This means we get the benefit of occlusion recovery without the risk of ghost detections.

### Decision 3: Counting line at frame centre
**Problem:** The "correct" counting line position depends on the physical store layout and camera angle. In a real deployment, this would be calibrated by a technician using the actual doorway pixel coordinates.

**Solution:** For the hackathon demo, we place the line at `frame_width // 2`. This works for the test video where foot traffic crosses the centre of the frame, and it is trivially configurable by changing one variable.

### Decision 4: Neighbourhood blob vs. single-pixel accumulation for heatmap
**Problem:** If a person appears in 200 consecutive frames, a single-pixel accumulator produces a dot, not a spatially meaningful hotspot.

**Solution:** We add `+1.0` to a `40×40 pixel square` neighbourhood centred on each detection. This creates smooth, spatially diffuse hotspots that correspond to the physical area a person occupies, not just their mathematical centroid.

### Decision 5: `persist=True` for cross-frame tracking
**Problem:** Calling `model.track()` without `persist=True` resets the Kalman filter state on every call, making every frame independent.

**Solution:** `persist=True` stores the tracker's internal state inside the model object, surviving across loop iterations. This is the foundational enabler for all dwell-time and zone-transition logic.

### Decision 6: NMS Tuning for Group Entry (`iou=0.75`)
**Problem:** When 2–4 people enter side-by-side, YOLO's default Non-Maximum Suppression (NMS) threshold (`iou=0.5`) merges overlapping bounding boxes into one — counting a group as a single person.

**Solution:** We raise the NMS IoU threshold to `iou=0.75`. This allows YOLO to retain separate bounding boxes for people who are physically adjacent. A higher IoU threshold means boxes must overlap *more* before being merged, so two close-but-distinct people are preserved as separate detections.

```python
model.track(frame, iou=0.75, ...)  # was iou=0.5 (default)
```

**Tradeoff accepted:** At `iou=0.75`, there is a marginally higher chance of retaining a duplicate detection of the *same* person. ByteTrack's ID consistency and our `counted_ids` set together guarantee only one ENTRY event fires per unique `track_id`.

### Decision 7: Smoothed Trajectory State Machine for Line Crossing
**Problem:** Raw frame-to-frame comparison (`prev_x < line_x and curr_x >= line_x`) breaks when a person is partially occluded *exactly at* the counting line. They may disappear for 2–3 frames and reappear on the other side — skipping the line-crossing event entirely.

**Solution:** We replaced the 1-frame diff with a **Smoothed Trajectory State Machine**:

1. **Trajectory History:** Every detected `x_center` is appended to a per-track `deque` of length 5. The *smoothed position* is the average of the last 5 readings.
2. **Anchor Side:** Each track is assigned an initial anchor side (`OUTSIDE` or `INSIDE`) based on where it first appeared relative to the counting line.
3. **State Transition:** An `ENTRY` fires when a track whose anchor is `OUTSIDE` has a smoothed x ≥ `line_x`. An `EXIT` fires for the reverse. The anchor is then flipped.

This means even if detection drops for 3 frames exactly at the door, the smoothed average of the surrounding frames will correctly resolve to the inside — triggering the `ENTRY` event.

```
Frame 1: x=30 (OUTSIDE)  → history=[30]   smoothed=30
Frame 2: x=45 (OUTSIDE)  → history=[30,45] smoothed=37.5
Frame 3: MISSED (occluded)
Frame 4: MISSED (occluded)
Frame 5: x=80 (INSIDE)   → history=[30,45,80] smoothed=51.6 ≥ line_x → ENTRY ✅
```

---

## 9. Design Tradeoffs

Every architectural decision involves a tradeoff. This section documents them explicitly.

### Tradeoff 1: YOLOv8 Nano vs. YOLOv8 Large

| | Nano | Large |
|---|---|---|
| **mAP** | 37.3 | 53.9 |
| **CPU latency** | ~25ms/frame | ~250ms/frame |
| **GPU latency** | ~2ms/frame | ~8ms/frame |
| **Chosen for** | Runs on store hardware | Better crowded-scene detection |

**Decision:** Nano for hackathon. A production system would use Large or X on a GPU.

### Tradeoff 2: SQLite vs. PostgreSQL

| | SQLite | PostgreSQL |
|---|---|---|
| **Setup time** | 0 minutes | ~30 minutes |
| **Concurrent writers** | 1 (serialised) | Unlimited |
| **Max practical rows** | ~10M | Billions |
| **Horizontal scaling** | ❌ | ✅ |
| **Chosen for** | Hackathon speed | Production scale |

**Decision:** SQLite for hackathon. The `database.py` isolation makes migration trivial.

### Tradeoff 3: Polling refresh vs. WebSockets

| | `st.rerun()` every 5s | WebSocket push |
|---|---|---|
| **Latency** | Up to 5 seconds | <100ms |
| **Complexity** | 2 lines of code | Separate async server |
| **Backend changes** | None | Requires FastAPI WebSocket routes |
| **Chosen for** | Hackathon demo | Production live view |

**Decision:** Polling is acceptable when events are generated at 10–20/minute. A live store with hundreds of events per minute would warrant WebSocket push.

### Tradeoff 4: File video vs. RTSP stream

| | `data/test_video.mp4` | RTSP live stream |
|---|---|---|
| **Reproducibility** | ✅ Identical results | ❌ Varies |
| **Hardware required** | None | IP camera + network |
| **Demo reliability** | ✅ Perfect | Risk of network issues |
| **Latency** | Offline processing | Real-time |
| **Chosen for** | Hackathon demo | Production deployment |

**Decision:** Pre-recorded video for reproducible, reliable demos. Switching to RTSP is a one-line change: `cv2.VideoCapture("rtsp://camera-ip/stream")`.

### Tradeoff 5: Single counting line vs. door polygon

| | Centre line | Door polygon |
|---|---|---|
| **Accuracy** | Good for aligned cameras | Excellent for any angle |
| **Configuration** | 1 variable | 4 coordinates |
| **Handles camera angle** | ❌ | ✅ |
| **Chosen for** | Simplicity | Real deployment |

---

## 10. Known Limitations

| Limitation | Root Cause | Production Solution |
|---|---|---|
| IDs reset if pipeline restarts | ByteTrack state is in-memory | Persist tracker state to Redis between restarts |
| Zone boundaries are fixed pixels | No camera calibration | Use homography matrix to map pixel zones to real-world floor coordinates |
| Single camera | Architecture is per-camera | Separate pipeline process per camera, shared database |
| No Re-ID across cameras | ByteTrack is per-stream | Add appearance embedding model (e.g., OSNet) for cross-camera Re-ID |
| Dwell time resets on ID switch | Each Track ID is treated as unique | Cluster close IDs by spatial proximity and timing |
| No staff filtering | All detected people counted | Train a classifier on uniform colours / ID badges |

---

## 11. Group-Entry & Motion-Based State Machine

### The Group-Entry Challenge
In retail analytics, counting accuracy deteriorates when 2–4 shoppers enter the store simultaneously. Traditional tracking pipelines fail in three major scenarios:
1. **Overlap & Occlusion:** Shoppers physically block each other at the threshold, causing YOLO to detect them as a single group box or ByteTrack to drop one individual's track.
2. **Deferred Detections (Spawn near Line):** An occluded group member might only be first detected *after* crossing the counting line. If the system uses a simple spatial heuristic (`outside` if `x < line_x`), their initial side is registered as `INSIDE`, completely missing their `ENTRY` event.
3. **Pacing / Jitter:** Multiple people walking closely can cause bounding box centroids to jitter back and forth across the door line, leading to double-counting.

### robust Group-Entry Handling Strategy
To ensure **one ENTRY event per individual customer**, we designed and implemented a **Motion-Based Deferred Anchor State Machine** combined with optimized model parameters:

```
                  Counting Line (line_x)
                           │
       [OUTSIDE]           │           [INSIDE]
                           │
  Track 1 ──(Start)────────┼───────────► (ENTRY fires)
                           │
  Track 2 ─────────────────┼──(Start)──► (ENTRY fires!)
             [Occluded at  │   [Spawned inside buffer]
              the doorway] │   Deferred anchor set to OUTSIDE
                           │   due to rightward velocity vector (+dx)
```

#### A. Motion-Based Deferred Anchor Initialization
Instead of assigning a static `OUTSIDE`/`INSIDE` anchor immediately on the very first frame a track is seen, we defer the decision if the track spawns inside a configurable **doorway buffer zone** (`door_buffer_pixels = 80`):
- **1-Frame Deferral:** When a track first appears within the buffer zone, we only update its trajectory history. No anchor side is set.
- **Directional Velocity Check:** Once the track accumulates at least 2 frames of trajectory history, we calculate its movement direction:
  $$\Delta x = x_{\text{latest}} - x_{\text{first}}$$
  - **Moving Right ($\Delta x > 0$):** The track is moving deeper into the store. We initialize its anchor side to `OUTSIDE`. On the same frame, since its current position is past the line, it immediately triggers a correct `ENTRY` event.
  - **Moving Left ($\Delta x < 0$):** The track is moving out of the store. We initialize its anchor side to `INSIDE`. If it continues to cross, an `EXIT` event is triggered.
- **Spatial Fallback:** If a track spawns far away from the doorway buffer zone, we immediately initialize its anchor side using the simple spatial rule (`OUTSIDE` if $x < \text{line\_x}$ else `INSIDE`), eliminating any latency for normal visitors.

#### B. YOLO & ByteTrack Parameter Optimization
- **NMS Tuning (`iou=0.75`):** We raised the NMS IoU threshold from `0.5` to `0.75`. This prevents YOLO from merging overlapping group members into a single bounding box while maintaining separate tracker IDs.
- **Kalman Filtering (`trajectory_history_len=5`):** Trajectories are smoothed across 5 frames. This filtering reduces physical jitter at the threshold, preventing false double-counting.

---

## 12. Billing Queue Detection & Analytics

### Queue State Machine Heuristics
Billing queue analytics tracks the state of each shopper as they approach checkout, detect line build-up, and spot queue abandonment.

```
       [Entering Zone]
              │
              ▼
       ┌──────────────┐
       │   BILLING    │◄─────────── (Session start)
       └──────┬───────┘
              │
              │  Dwells >= 10s
              ▼
       ┌──────────────┐
       │   IN QUEUE   ├────────────► [Store Exit] (Successful purchase)
       └──────┬───────┘
              │
              │  Exits BILLING zone
              ▼
       ┌──────────────┐
       │ PENDING ABAND│
       └──────┬───────┘
              │
              ├──────────────────────────────┐
              │ Enter Skincare/Cosmetics     │ Exit Store directly
              ▼                              ▼
      [QUEUE_ABANDON]                  [Clear pending]
```

#### 1. Queue Formation (Dwell Time Threshold)
- A shopper is not counted in the queue immediately upon stepping into the `"BILLING"` zone coordinates to avoid false joins from passers-by.
- A **dwell threshold of 10 seconds** is enforced. When a shopper's cumulative time in `"BILLING"` reaches $\ge 10,000$ ms, a `BILLING_QUEUE_JOIN` event is emitted.

#### 2. Queue Depth & Growth
- **Queue Depth Calculation:** In real-time, the backend computes the live queue depth by counting unique visitors who have emitted `BILLING_QUEUE_JOIN` but have not yet emitted `BILLING_QUEUE_ABANDON`, `PURCHASE`, or `EXIT`.
- **Growth Anomalies:** If queue depth exceeds $2$, a `QUEUE_BUILDUP` warning is raised. If it exceeds $5$, a `QUEUE_SPIKE` critical alert is dispatched.

#### 3. Queue Abandonment Detection
- When a shopper in the `IN QUEUE` state exits the `"BILLING"` zone, we mark their state as `PENDING_ABANDON`.
- If the shopper enters a shopping zone (`"SKINCARE"` or `"COSMETICS"`) while `PENDING_ABANDON` is active, we trigger a `BILLING_QUEUE_ABANDON` event.
- If the shopper crosses the exit line to exit the store, we clear the `PENDING_ABANDON` flag, categorizing it as a completed purchase checkout.

---

## 13. Graceful Occlusion Recovery

### The Occlusion Challenge
In busy retail store layouts, visitors are frequently blocked from camera view by high shelving, promotional aisle displays, or other shoppers walking close by. When a visitor's track is lost, standard trackers immediately forget the track state. When they reappear, they are assigned a brand new `track_id`, which artificially inflates customer counts and fragments journey sessions.

### In-Store Recovery Architecture
We implemented an active **Lost-Track Signature Cache** combined with **HSV-based local Re-Identification** to recover lost tracks seamlessly:

```
    [Track 28 is active] ────► [Occluded by Display] ────► [Active ID goes missing]
             │                                                     │
      Update signature                                    Register into lost cache
      every frame (HSV)                                   with 30-second TTL
                                                                   │
                                                                   ▼
    [Track 45 starts inside] ◄─── Cosine Match (sig) ◄──── [Reappears in Skincare]
             │                        similarity > 0.82
             ▼
    Map Track 45 ──► Track 28
    Carry over session states (Billing Queue, counted, zone enter times)
```

#### A. Active Torso Signature Caching
- As long as a track is detected, we periodically extract its torso HSV histogram signature via `update_active_signature` and keep it fresh in `active_signatures`.

#### B. Lost Track Registration
- When a track goes missing from the active YOLO list, we register it in `lost_track_cache` mapping `lost_track_id -> {time, signature, last_bbox, last_zone}` with a **30-second Time-To-Live (TTL)**.

#### C. Local Re-ID Comparison
- When a new track ID is spawned inside the store, we compare its torso HSV histogram signature against the lost cache using cosine similarity.
- If **cosine similarity exceeds 0.82**, height difference is $<30\%$, and they reappear in a spatially contiguous zone, the new track is matched.
- The new ID is mapped back to the lost track ID inside `id_mapping`.
- The visitor session data, billing queue states, and counted flags are carried over, ensuring the journey metrics continue seamlessly.

#### D. Offline Exit Recovery (Preventing Missed Exits)
- If a lost track expires without recovery, and its last known coordinates were near the exit line boundary (within 80 pixels), we gracefully emit an offline `EXIT` event. This guarantees that customer counts balance correctly even under heavy exit occlusions.

## 14. Arbitrary Polygon Zones
- Store layouts are configured using either axis-aligned bounding boxes `[x1, y1, x2, y2]` or arbitrary polygons (lists of 2D coordinates `[[x1, y1], [x2, y2], ...]`) in `data/store_layout.json`.
- The pipeline utilizes `cv2.pointPolygonTest` to perform precise containment checks for coordinates of visitor tracks. This allows support for complex zone geometries, non-rectangular store shapes, and perspective-distorted CCTV feeds.

## 15. POS Transaction Correlation
- Sales transactions are stored in `data/pos_transactions.csv`.
- The metrics and funnel calculations correlate customer session data with POS data by matching visitors who were in the `"BILLING"` zone within 5 minutes prior to a recorded transaction timestamp.
- Matches are counted as converted visitors, and are used to compute the store's conversion rate, total transactions, and stage-by-stage drop-off percentages.

## 16. Cross-Camera Visitor Deduplication
- Multiple CCTV camera streams (e.g. entry, floor, billing) track shoppers locally.
- To prevent double counting and maintain unified shopper journey metrics, the system implements a shared `CrossCameraMatcher` in `pipeline/cross_camera.py` reading and writing to `data/cross_camera_registry.json`.
- When a new shopper is detected on any camera, their torso HSV signature is matched against recent visitors from other cameras using cosine similarity (threshold >= 0.82) and a 5-minute temporal window constraint.
- If a match is found, the global visitor ID is mapped to the local track ID, preventing duplicate `ENTRY` events and maintaining visitor_id continuity.

---

*This document is intended for technical judges evaluating the Store Intelligence System hackathon submission. All design decisions reflect the constraints of a time-boxed hackathon while maintaining a clear path to production deployment.*
