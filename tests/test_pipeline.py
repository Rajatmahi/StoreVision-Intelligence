# PROMPT: Generate FastAPI tests for POST /events/ingest checking idempotency (sending the same event twice) and graceful degradation for missing payload fields using pytest and httpx.
# CHANGES MADE: Added setup/teardown for test DB to avoid polluting the main DB. Adapted the models to match the actual Event schema.

import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

# Use a test database
os.environ["DB_PATH"] = "data/test_store_intelligence.db"

@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    yield
    if os.path.exists("data/test_store_intelligence.db"):
        os.remove("data/test_store_intelligence.db")

client = TestClient(app)

def test_ingest_idempotency():
    event = {
        "event_id": "test-uuid-1",
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_test_1",
        "event_type": "ENTRY",
        "timestamp": "2026-03-03T14:22:10Z",
        "zone_id": None,
        "dwell_ms": 0,
        "is_staff": False,
        "confidence": 0.95,
        "metadata": {}
    }
    
    # First insert
    response = client.post("/events/ingest", json=[event])
    assert response.status_code == 200
    data = response.json()
    assert data["events_received"] == 1
    assert data["duplicates_skipped"] == 0
    
    # Second insert (duplicate)
    response2 = client.post("/events/ingest", json=[event])
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["events_received"] == 0
    assert data2["duplicates_skipped"] == 1

def test_ingest_validation_error():
    # Missing required field "event_type"
    event = {
        "event_id": "test-uuid-2",
        "store_id": "STORE_BLR_002",
        # missing other fields
    }
    response = client.post("/events/ingest", json=[event])
    assert response.status_code == 200
    data = response.json()
    assert data["malformed_skipped"] == 1
    assert data["events_received"] == 0
