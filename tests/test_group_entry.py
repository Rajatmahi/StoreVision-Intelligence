import numpy as np
from pipeline.tracker import VisitorTracker

def test_trajectory_smoothing():
    tracker = VisitorTracker(trajectory_history_len=3)
    
    # 1. Update 1: x=10
    tracker.update_trajectory(1, 10)
    assert tracker.get_smoothed_x(1) == 10.0
    
    # 2. Update 2: x=20
    tracker.update_trajectory(1, 20)
    assert tracker.get_smoothed_x(1) == 15.0 # (10+20)/2
    
    # 3. Update 3: x=30
    tracker.update_trajectory(1, 30)
    assert tracker.get_smoothed_x(1) == 20.0 # (10+20+30)/3
    
    # 4. Update 4: x=40 (pushes out 10)
    tracker.update_trajectory(1, 40)
    assert tracker.get_smoothed_x(1) == 30.0 # (20+30+40)/3

def test_anchor_state_machine_logic():
    tracker = VisitorTracker()
    line_x = 50
    
    # Assume track 1 starts OUTSIDE
    tracker.track_anchor_side[1] = "OUTSIDE"
    
    # Frame 1: well outside
    tracker.update_trajectory(1, 30)
    
    # Frame 2: getting close
    tracker.update_trajectory(1, 45)
    
    # Frame 3: missed detection (no update)
    # Frame 4: missed detection (no update)
    
    # Frame 5: reappears inside!
    tracker.update_trajectory(1, 85)
    
    smoothed = tracker.get_smoothed_x(1)
    
    # Even though we missed the exact line crossing, 
    # the smoothed x is (30+45+85)/3 = 53.33, which is >= 50
    # Our anchor was OUTSIDE, so this correctly triggers an ENTRY!
    assert smoothed >= line_x
    
    # Emulate the loop logic
    if tracker.track_anchor_side[1] == "OUTSIDE" and smoothed >= line_x:
        tracker.track_anchor_side[1] = "INSIDE"
        
    assert tracker.track_anchor_side[1] == "INSIDE"

def test_deferred_motion_anchor_outside():
    tracker = VisitorTracker()
    line_x = 50
    track_id = 10
    
    # 1. Update 1: first position x=48 is close to the line (50)
    tracker.update_trajectory(track_id, 48)
    tracker.initialize_track_anchor(track_id, line_x)
    
    # Anchor side should be deferred (None) because we only have 1 frame
    assert track_id not in tracker.track_anchor_side
    
    # 2. Update 2: second position x=55 (moving right, entry)
    tracker.update_trajectory(track_id, 55)
    tracker.initialize_track_anchor(track_id, line_x)
    
    # Anchor side should now be set to "OUTSIDE" (started outside and entered)
    assert tracker.track_anchor_side[track_id] == "OUTSIDE"

def test_deferred_motion_anchor_inside():
    tracker = VisitorTracker()
    line_x = 50
    track_id = 11
    
    # 1. Update 1: first position x=52 is close to the line (50)
    tracker.update_trajectory(track_id, 52)
    tracker.initialize_track_anchor(track_id, line_x)
    
    # Deferred
    assert track_id not in tracker.track_anchor_side
    
    # 2. Update 2: second position x=45 (moving left, exit)
    tracker.update_trajectory(track_id, 45)
    tracker.initialize_track_anchor(track_id, line_x)
    
    # Anchor side should be set to "INSIDE"
    assert tracker.track_anchor_side[track_id] == "INSIDE"

def test_non_close_track():
    tracker = VisitorTracker()
    line_x = 50
    track_id = 12
    
    # First position x=-50 is far from the line (50), so no deferral
    tracker.update_trajectory(track_id, -50)
    tracker.initialize_track_anchor(track_id, line_x)
    
    # Anchor side should be immediately set to "OUTSIDE"
    assert tracker.track_anchor_side[track_id] == "OUTSIDE"

if __name__ == "__main__":
    test_trajectory_smoothing()
    test_anchor_state_machine_logic()
    test_deferred_motion_anchor_outside()
    test_deferred_motion_anchor_inside()
    test_non_close_track()
    print("Group entry tests passed!")
