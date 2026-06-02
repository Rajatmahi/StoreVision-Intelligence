import numpy as np
from pipeline.tracker import VisitorTracker

def test_axis_aligned_box():
    tracker = VisitorTracker()
    coords = [10, 10, 100, 100] # x1, y1, x2, y2
    
    # Inside point
    assert tracker.is_inside_zone(50, 50, coords) is True
    # Boundary point
    assert tracker.is_inside_zone(10, 10, coords) is True
    assert tracker.is_inside_zone(100, 100, coords) is True
    # Outside point
    assert tracker.is_inside_zone(5, 50, coords) is False
    assert tracker.is_inside_zone(50, 105, coords) is False

def test_polygon_coords():
    tracker = VisitorTracker()
    # A triangular polygon: (0,0), (100, 0), (50, 100)
    coords = [[0, 0], [100, 0], [50, 100]]
    
    # Inside point
    assert tracker.is_inside_zone(50, 50, coords) is True
    # Outside point
    assert tracker.is_inside_zone(10, 80, coords) is False
    assert tracker.is_inside_zone(50, 110, coords) is False
    
    # A diamond polygon
    diamond = [[50, 0], [100, 50], [50, 100], [0, 50]]
    assert tracker.is_inside_zone(50, 50, diamond) is True
    assert tracker.is_inside_zone(10, 10, diamond) is False
