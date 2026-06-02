import uuid
import datetime
import requests

API_URL = "http://127.0.0.1:8000/events/ingest"

def send_event(
    store_id: str,
    camera_id: str,
    event_type: str,
    track_id: int,
    confidence: float,
    is_staff: bool = False,
    zone_id: str = None,
    dwell_ms: int = 0,
    metadata: dict = None
):
    event_id = str(uuid.uuid4())
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    visitor_label = str(track_id)
    print(f"[{event_type}] {visitor_label} (Staff: {is_staff})")
    
    meta = metadata or {}
    # Auto-inject queue/dwell metadata if missing but values are provided
    if event_type in ("BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"):
        if "queue_depth" not in meta:
            # Try to get queue_depth, default to 0 if not passed
            meta["queue_depth"] = meta.get("queue_depth", 0)
    elif event_type in ("ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL"):
        if "zone_id" not in meta and zone_id is not None:
            meta["zone_id"] = zone_id
        if "dwell_ms" not in meta:
            meta["dwell_ms"] = dwell_ms

    event_payload = {
        "event_id": event_id,
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_label,
        "event_type": event_type,
        "timestamp": now_iso,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": float(confidence),
        "metadata": meta
    }
    
    try:
        response = requests.post(API_URL, json=[event_payload])

        if response.status_code == 200:
            print(f"✓ Event sent: {event_type}")
        else:
            print(f"✗ API returned {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("Warning: Could not connect to FastAPI server.")
