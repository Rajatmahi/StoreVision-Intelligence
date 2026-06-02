"""
tests/test_staff_classifier.py

Validates:
  - All 4 behaviour rules
  - Uniform colour detection (white / black / navy)
  - Sticky flag (once True stays True)
  - Non-staff track is not falsely classified
"""

import numpy as np
import time
import pytest

from pipeline.staff_classifier import (
    StaffClassifier,
    StaffFeatures,
    BehaviourClassifier,
    UniformClassifier,
    STAFF_HSV_PALETTE,
    LONG_PRESENCE_THRESHOLD_SEC,
    ZONE_TRAVERSAL_THRESHOLD,
    TRAVERSAL_RATE_WINDOW_MS,
    TRAVERSAL_RATE_THRESHOLD,
    BACKTRACK_ZONE_THRESHOLD,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_frame_with_torso_color(bgr_color, width=200, height=400):
    """Create a synthetic frame where the torso area is a solid colour."""
    frame = np.full((height, width, 3), bgr_color, dtype=np.uint8)
    return frame

# A bbox that occupies the full frame so the torso crop is non-empty
FULL_FRAME_BBOX = [100, 200, 200, 400]  # [cx, cy, w, h]

# ─── Behaviour rule tests ──────────────────────────────────────────────────────

class TestBehaviourClassifier:
    def _make_feat(self) -> StaffFeatures:
        f = StaffFeatures()
        f.first_seen_ms = 0
        return f

    def test_r1_long_presence(self):
        bc = BehaviourClassifier()
        feat = self._make_feat()
        threshold_ms = LONG_PRESENCE_THRESHOLD_SEC * 1000 + 1
        is_staff, rule = bc.classify(feat, current_ms=threshold_ms)
        assert is_staff, "R1 should trigger after long presence"
        assert rule == "R1_LONG_PRESENCE"

    def test_r1_not_triggered_before_threshold(self):
        bc = BehaviourClassifier()
        feat = self._make_feat()
        short_time_ms = LONG_PRESENCE_THRESHOLD_SEC * 1000 - 1000
        is_staff, rule = bc.classify(feat, current_ms=short_time_ms)
        assert not is_staff

    def test_r2_high_zone_traversal(self):
        bc = BehaviourClassifier()
        feat = self._make_feat()
        feat.first_seen_ms = 0
        # Simulate enough zone entries over 2 minutes (past the 60 s warm-up)
        t = 90_000  # 90 seconds in
        for i in range(ZONE_TRAVERSAL_THRESHOLD):
            feat.zone_visits.append((f"ZONE_{i}", t + i * 1000))
            feat.total_zone_entries += 1
        is_staff, rule = bc.classify(feat, current_ms=t + 10_000)
        assert is_staff, "R2 should trigger on high zone traversal count"
        assert rule == "R2_HIGH_TRAVERSAL"

    def test_r3_high_traversal_rate(self):
        bc = BehaviourClassifier()
        feat = self._make_feat()
        feat.first_seen_ms = 0
        # All zone entries clustered in a 10-min window, after 60 s warm-up
        t_base = 90_000
        for i in range(TRAVERSAL_RATE_THRESHOLD):
            feat.zone_visits.append((f"ZONE_{i}", t_base + i * 60_000))
            feat.total_zone_entries += 1
        current_ms = t_base + (TRAVERSAL_RATE_THRESHOLD - 1) * 60_000 + 1000
        is_staff, rule = bc.classify(feat, current_ms=current_ms)
        assert is_staff, "R3 should trigger on high zone traversal rate"
        assert rule == "R3_HIGH_TRAVERSAL_RATE"

    def test_r4_backtrack_same_zone(self):
        bc = BehaviourClassifier()
        feat = self._make_feat()
        feat.first_seen_ms = 0
        t = 90_000
        # Simulate visiting ZONE_A many times spaced out to avoid R3
        for i in range(BACKTRACK_ZONE_THRESHOLD):
            feat.zone_visits.append(("ZONE_A", t + i * 300_000))
            feat.zone_visit_counts["ZONE_A"] = feat.zone_visit_counts.get("ZONE_A", 0) + 1
            feat.total_zone_entries += 1
        is_staff, rule = bc.classify(feat, current_ms=t + 630_000)
        assert is_staff, "R4 should trigger on same-zone revisits"
        assert rule == "R4_BACKTRACK"

    def test_minimum_warmup_60s(self):
        """No rule fires in the first 60 seconds."""
        bc = BehaviourClassifier()
        feat = self._make_feat()
        feat.first_seen_ms = 0
        # Artificially fill zone visits but we're still in the 60 s window
        for i in range(20):
            feat.zone_visits.append((f"ZONE_{i}", i * 1000))
            feat.total_zone_entries += 1
        is_staff, rule = bc.classify(feat, current_ms=59_999)
        assert not is_staff, "Behaviour rules must not fire in first 60 s"


# ─── Uniform classifier tests ──────────────────────────────────────────────────

class TestUniformClassifier:
    def test_white_apron_detected(self):
        uc = UniformClassifier()
        # White BGR → (255, 255, 255)
        frame = make_frame_with_torso_color((255, 255, 255))
        feat = StaffFeatures()
        # Feed 30+ frames to exceed the warm-up
        result = False
        for _ in range(35):
            result = uc.classify_frame(frame, FULL_FRAME_BBOX, feat)
        assert result, "White uniform should be classified as staff"

    def test_black_uniform_detected(self):
        uc = UniformClassifier()
        # Black BGR → (0, 0, 0)
        frame = make_frame_with_torso_color((0, 0, 0))
        feat = StaffFeatures()
        result = False
        for _ in range(35):
            result = uc.classify_frame(frame, FULL_FRAME_BBOX, feat)
        assert result, "Black uniform should be classified as staff"

    def test_customer_color_not_detected(self):
        uc = UniformClassifier()
        # Bright orange — typical customer clothing, not a staff palette colour
        frame = make_frame_with_torso_color((0, 128, 255))  # BGR orange
        feat = StaffFeatures()
        result = False
        for _ in range(35):
            result = uc.classify_frame(frame, FULL_FRAME_BBOX, feat)
        assert not result, "Orange clothing should NOT be classified as staff"

    def test_warmup_no_classification_before_30_frames(self):
        uc = UniformClassifier()
        frame = make_frame_with_torso_color((255, 255, 255))  # white
        feat = StaffFeatures()
        for _ in range(29):
            result = uc.classify_frame(frame, FULL_FRAME_BBOX, feat)
        assert not result, "Should not classify until >= 30 frames observed"


# ─── Integrated StaffClassifier tests ─────────────────────────────────────────

class TestStaffClassifier:
    def test_sticky_flag(self):
        """Once is_staff=True, it never reverts to False."""
        sc = StaffClassifier()
        frame = make_frame_with_torso_color((255, 255, 255))
        t = 0
        # Force uniform classification by feeding 35 frames
        for i in range(35):
            sc.update(track_id=1, frame=frame, bbox=FULL_FRAME_BBOX, current_ms=i * 1000)
        assert sc.is_staff(1), "Should be staff after white uniform detection"
        
        # Now pass a non-staff frame — flag must not revert
        orange_frame = make_frame_with_torso_color((0, 128, 255))
        for i in range(35, 70):
            sc.update(track_id=1, frame=orange_frame, bbox=FULL_FRAME_BBOX, current_ms=i * 1000)
        assert sc.is_staff(1), "Staff flag must remain True (sticky)"

    def test_customer_not_classified_as_staff(self):
        """A short-duration, low-zone-count, non-uniform person is not staff."""
        sc = StaffClassifier()
        orange_frame = make_frame_with_torso_color((0, 128, 255))
        for i in range(40):
            sc.update(track_id=99, frame=orange_frame, bbox=FULL_FRAME_BBOX,
                      current_ms=i * 1000, current_zone="COSMETICS")
        assert not sc.is_staff(99), "Normal customer should not be classified as staff"

    def test_unknown_track_returns_false(self):
        sc = StaffClassifier()
        assert sc.is_staff(99999) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
