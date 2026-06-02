import requests
import datetime
import uuid
import time

API_URL = "http://127.0.0.1:8000/events/ingest"
STORE_ID = "STORE_BLR_002"
CAMERA_ID = "CAM_SIMULATOR"

def send_events(events):
    try:
        response = requests.post(API_URL, json=events)
        print(f"Sent {len(events)} events. Response: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to API")

def create_event(visitor_id, event_type, zone_id=None, dwell_ms=0):
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": STORE_ID,
        "camera_id": CAMERA_ID,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": False,
        "confidence": 0.99,
        "metadata": {}
    }

def simulate():
    print("Starting Simulation...")
    events = []
    
    # 1. Simulate TRAFFIC_SPIKE (25 entries quickly)
    for i in range(25):
        events.append(create_event(f"VIS_SPIKE_{i}", "ENTRY"))
    
    # 2. Simulate OVERCROWDING in COSMETICS (15 active visitors)
    for i in range(15):
        events.append(create_event(f"VIS_CROWD_{i}", "ENTRY"))
        events.append(create_event(f"VIS_CROWD_{i}", "ZONE_ENTER", "COSMETICS"))

    # 3. Simulate LONG_DWELL (One visitor stayed for 20 minutes)
    events.append(create_event("VIS_SLEEPER", "ENTRY"))
    events.append(create_event("VIS_SLEEPER", "ZONE_ENTER", "SKINCARE"))
    events.append(create_event("VIS_SLEEPER", "ZONE_EXIT", "SKINCARE", dwell_ms=1200000)) # 20 mins

    # 4. Populate Funnel (Entry -> Zone -> Billing Queue -> Purchase)
    for i in range(10): # 10 entered
        vid = f"VIS_FUNNEL_{i}"
        events.append(create_event(vid, "ENTRY"))
        
        if i < 8: # 8 visited zones
            events.append(create_event(vid, "ZONE_ENTER", "SKINCARE"))
            events.append(create_event(vid, "ZONE_EXIT", "SKINCARE", dwell_ms=15000))
            
        if i < 6: # 6 joined queue
            events.append(create_event(vid, "BILLING_QUEUE_JOIN", "BILLING"))
            
        if i < 4: # 4 purchased
            events.append(create_event(vid, "PURCHASE", "BILLING"))
            events.append(create_event(vid, "EXIT"))
            
        if i >= 4 and i < 6: # 2 abandoned
            events.append(create_event(vid, "BILLING_QUEUE_ABANDON", "BILLING"))
            events.append(create_event(vid, "EXIT"))
            
    send_events(events)
    print("Simulation complete! Check the dashboard.")

if __name__ == "__main__":
    simulate()
