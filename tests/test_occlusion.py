import numpy as np
import pytest
from pipeline.tracker import VisitorTracker

def test_active_signature_caching():
    tracker = VisitorTracker()
    # Dummy frame (100x100) and bbox
    frame = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
    bbox = [50, 50, 40, 80]
    
    tracker.update_active_signature(track_id=1, frame=frame, bbox=bbox)
    assert 1 in tracker.active_signatures
    assert 1 in tracker.last_bboxes
    assert len(tracker.active_signatures[1]) == 256

def test_lost_track_registration():
    tracker = VisitorTracker()
    frame = np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8)
    bbox = [50, 50, 40, 80]
    
    tracker.update_active_signature(track_id=2, frame=frame, bbox=bbox)
    tracker.register_lost_track(track_id=2, last_zone="COSMETICS", current_time_ms=1000)
    
    assert 2 not in tracker.active_signatures
    assert 2 in tracker.lost_track_cache
    assert tracker.lost_track_cache[2]["zone"] == "COSMETICS"
    assert tracker.lost_track_cache[2]["time_ms"] == 1000

def test_in_store_recovery():
    tracker = VisitorTracker()
    frame = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)
    bbox = [50, 50, 40, 80]
    
    # Track 3 is registered, then lost in Skincare
    tracker.update_active_signature(track_id=3, frame=frame, bbox=bbox)
    tracker.visitor_sessions[3] = {"zone": "SKINCARE", "enter_time_ms": 1000}
    tracker.register_lost_track(track_id=3, last_zone="SKINCARE", current_time_ms=2000)
    
    # Track 4 appears inside Skincare with identical signature
    is_recovered, lost_id = tracker.check_lost_recovery(
        new_track_id=4, frame=frame, bbox=bbox, current_zone="SKINCARE", current_time_ms=3000
    )
    
    assert is_recovered is True
    assert lost_id == 3
    assert tracker.get_mapped_track_id(4) == 3
    # Check that session carries over
    assert tracker.visitor_sessions[4]["zone"] == "SKINCARE"

def test_reap_expired_lost_tracks_no_exit():
    tracker = VisitorTracker()
    frame = np.full((600, 600, 3), (255, 255, 255), dtype=np.uint8)
    bbox = [100, 50, 40, 80] # Center x = 100, far from line (500)
    
    tracker.update_active_signature(track_id=5, frame=frame, bbox=bbox)
    tracker.register_lost_track(track_id=5, last_zone="COSMETICS", current_time_ms=1000)
    
    events_emitted = []
    def mock_send_event(store_id, camera_id, event_type, track_id, conf, staff):
        events_emitted.append((event_type, track_id))
        
    # Reap at t=40000ms (>30s TTL)
    tracker.reap_expired_lost_tracks(
        current_time_ms=40000, line_x=500, store_id="STORE", camera_id="CAM", send_event_fn=mock_send_event
    )
    
    # Since center_x=100 is far from line=500 (distance 400 > 80), no exit should fire
    assert len(events_emitted) == 0
    assert 5 not in tracker.lost_track_cache

def test_reap_expired_lost_tracks_with_exit():
    tracker = VisitorTracker()
    frame = np.full((600, 600, 3), (255, 255, 255), dtype=np.uint8)
    bbox = [480, 50, 40, 80] # Center x = 480, close to line (500)
    
    tracker.update_active_signature(track_id=6, frame=frame, bbox=bbox)
    tracker.register_lost_track(track_id=6, last_zone=None, current_time_ms=1000)
    
    events_emitted = []
    def mock_send_event(store_id, camera_id, event_type, track_id, conf, staff):
        events_emitted.append((event_type, track_id))
        
    # Reap at t=40000ms
    tracker.reap_expired_lost_tracks(
        current_time_ms=40000, line_x=500, store_id="STORE", camera_id="CAM", send_event_fn=mock_send_event
    )
    
    # Since center_x=480 is within 80 pixels of exit boundary line (500), exit should trigger!
    assert len(events_emitted) == 1
    assert events_emitted[0] == ("EXIT", 6)
    assert 6 not in tracker.lost_track_cache
