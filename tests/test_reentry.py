import numpy as np
import cv2
from pipeline.tracker import VisitorTracker

def test_extract_appearance_signature():
    tracker = VisitorTracker()
    # Create a dummy image 100x100 (blue)
    frame = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
    bbox = [50, 50, 40, 80] # center_x, center_y, width, height
    
    sig = tracker.extract_appearance_signature(frame, bbox)
    assert sig is not None
    assert sig.shape == (256,) # 16x16 flattened

def test_reentry_caching():
    tracker = VisitorTracker(reentry_ttl_sec=10)
    
    # Dummy image 100x100 red
    frame_red = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)
    bbox = [50, 50, 40, 80]
    
    # 1. Register exit
    tracker.register_exit(original_track_id=1, frame=frame_red, bbox=bbox)
    
    # 2. Check reentry with same color
    is_reentry, mapped_id = tracker.check_reentry(new_track_id=2, frame=frame_red, bbox=bbox)
    assert is_reentry is True
    assert mapped_id == 1
    
    # 3. Check mapping logic
    assert tracker.get_mapped_track_id(2) == 1
    
    # 4. Check reentry with different color (blue)
    tracker.register_exit(original_track_id=3, frame=frame_red, bbox=bbox)
    frame_blue = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
    is_reentry2, mapped_id2 = tracker.check_reentry(new_track_id=4, frame=frame_blue, bbox=bbox)
    
    assert is_reentry2 is False
    assert mapped_id2 is None
    assert tracker.get_mapped_track_id(4) == 4

if __name__ == "__main__":
    test_extract_appearance_signature()
    test_reentry_caching()
    print("Re-entry tests passed!")
