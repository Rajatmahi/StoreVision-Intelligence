import os
import time
import numpy as np
import pytest
from pipeline.cross_camera import CrossCameraMatcher

@pytest.fixture
def temp_registry_path():
    path = "data/test_cross_camera_registry.json"
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_cross_camera_registration_and_matching(temp_registry_path):
    matcher = CrossCameraMatcher(filepath=temp_registry_path, ttl_sec=5)
    
    # 1. Test registration of a signature
    sig1 = np.array([1.0, 0.0, 0.0, 0.0])
    matcher.register_visitor(visitor_id="VIS_1", camera_id="CAM_A", signature=sig1)
    
    # Check that visitor registry contains the visitor
    data = matcher._load()
    assert "VIS_1" in data
    assert data["VIS_1"]["camera_id"] == "CAM_A"
    
    # 2. Test matching from a different camera with same signature
    match_id = matcher.find_match(camera_id="CAM_B", signature=sig1)
    assert match_id == "VIS_1"
    
    # 3. Test that matching on the same camera is avoided
    match_id_same = matcher.find_match(camera_id="CAM_A", signature=sig1)
    assert match_id_same is None
    
    # 4. Test matching with different signature (low similarity)
    sig2 = np.array([0.0, 1.0, 0.0, 0.0])
    match_id_diff = matcher.find_match(camera_id="CAM_B", signature=sig2)
    assert match_id_diff is None

def test_cross_camera_ttl_expiry(temp_registry_path):
    # Short TTL of 1 second
    matcher = CrossCameraMatcher(filepath=temp_registry_path, ttl_sec=1)
    sig = np.array([1.0, 1.0, 1.0, 1.0])
    
    matcher.register_visitor(visitor_id="VIS_1", camera_id="CAM_A", signature=sig)
    
    # Match immediately -> should succeed
    assert matcher.find_match(camera_id="CAM_B", signature=sig) == "VIS_1"
    
    # Wait for TTL to expire
    time.sleep(1.2)
    
    # Match now -> should fail
    assert matcher.find_match(camera_id="CAM_B", signature=sig) is None
