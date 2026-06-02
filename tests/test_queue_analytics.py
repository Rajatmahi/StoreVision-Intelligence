import pytest
import os
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db, get_db_connection
from pipeline.tracker import VisitorTracker

os.environ["DB_PATH"] = "data/test_queue_analytics.db"
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events")
    conn.commit()
    conn.close()
    yield
    if os.path.exists("data/test_queue_analytics.db"):
        os.remove("data/test_queue_analytics.db")

def test_api_queue_depth_calculations():
    # 1. Ingest a join event
    join_event = {
        "event_id": "join-1",
        "store_id": "STORE_QUEUE_TEST",
        "camera_id": "CAM",
        "visitor_id": "VIS_QUEUE_1",
        "event_type": "BILLING_QUEUE_JOIN",
        "timestamp": "2026-03-03T14:00:00Z",
        "zone_id": "BILLING",
        "dwell_ms": 10000,
        "is_staff": False,
        "confidence": 0.9,
        "metadata": {}
    }
    client.post("/events/ingest", json=[join_event])
    
    # Verify queue depth is 1
    res = client.get("/stores/STORE_QUEUE_TEST/metrics")
    assert res.status_code == 200
    assert res.json()["queue_depth"] == 1
    
    # 2. Ingest another join event
    join_event2 = {**join_event, "event_id": "join-2", "visitor_id": "VIS_QUEUE_2"}
    client.post("/events/ingest", json=[join_event2])
    
    # Verify queue depth is 2
    res = client.get("/stores/STORE_QUEUE_TEST/metrics")
    assert res.json()["queue_depth"] == 2
    
    # 3. Ingest an abandon event for visitor 1
    abandon_event = {
        "event_id": "abandon-1",
        "store_id": "STORE_QUEUE_TEST",
        "camera_id": "CAM",
        "visitor_id": "VIS_QUEUE_1",
        "event_type": "BILLING_QUEUE_ABANDON",
        "timestamp": "2026-03-03T14:01:00Z",
        "zone_id": "BILLING",
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.9,
        "metadata": {}
    }
    client.post("/events/ingest", json=[abandon_event])
    
    # Verify queue depth is 1
    res = client.get("/stores/STORE_QUEUE_TEST/metrics")
    assert res.json()["queue_depth"] == 1
    
    # 4. Ingest a purchase event for visitor 2
    purchase_event = {
        "event_id": "purchase-2",
        "store_id": "STORE_QUEUE_TEST",
        "camera_id": "CAM",
        "visitor_id": "VIS_QUEUE_2",
        "event_type": "PURCHASE",
        "timestamp": "2026-03-03T14:02:00Z",
        "zone_id": "BILLING",
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.9,
        "metadata": {}
    }
    client.post("/events/ingest", json=[purchase_event])
    
    # Verify queue depth is 0
    res = client.get("/stores/STORE_QUEUE_TEST/metrics")
    assert res.json()["queue_depth"] == 0

def test_tracker_queue_state_initialization():
    tracker = VisitorTracker()
    assert hasattr(tracker, "joined_billing_queue")
    assert hasattr(tracker, "billing_queue_pending_abandon")
    assert isinstance(tracker.joined_billing_queue, dict)
    assert isinstance(tracker.billing_queue_pending_abandon, dict)
