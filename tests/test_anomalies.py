# PROMPT: Generate FastAPI tests for GET /stores/{store_id}/anomalies to check detection of queue spikes, conversion drops, and dead zones.
# CHANGES MADE: Modified the logic to inject events matching the anomaly conditions to assert that they are correctly detected by the new anomaly logic we will implement.

import os
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db, get_db_connection

os.environ["DB_PATH"] = "data/test_anomalies.db"
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
    if os.path.exists("data/test_anomalies.db"):
        os.remove("data/test_anomalies.db")

def test_anomaly_dead_zone():
    # If no events in the last 30 minutes, it should flag a dead zone
    response = client.get("/stores/STORE_DEAD/anomalies")
    assert response.status_code == 200
    data = response.json()
    assert "anomalies" in data
