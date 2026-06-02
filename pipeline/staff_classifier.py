"""
staff_classifier.py — Behaviour-based and appearance-based staff identification.

Two independent layers:
  1. UniformClassifier  : HSV torso-colour histogram vs. configurable staff palette.
  2. BehaviourClassifier: Rule engine on per-track movement features.

A track is flagged is_staff=True if EITHER layer triggers.
Once set True, the flag is sticky for the rest of that tracking session.
No hardcoded visitor IDs.  No facial recognition.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─── Configurable thresholds ────────────────────────────────────────────────

# Behaviour rules — adjust per store
LONG_PRESENCE_THRESHOLD_SEC   = 4 * 3600   # 4 hours
ZONE_TRAVERSAL_THRESHOLD       = 8          # distinct zone entries per session
TRAVERSAL_RATE_WINDOW_MS       = 10 * 60 * 1000  # 10 minutes
TRAVERSAL_RATE_THRESHOLD       = 3          # zones per window
BACKTRACK_ZONE_THRESHOLD       = 3          # same zone entered ≥ N times

# Uniform colour matching
UNIFORM_MATCH_FRAME_RATIO      = 0.60       # 60% of observed frames must match palette
STAFF_HSV_PALETTE = [
    # White apron / polo:  low saturation, high value
    {"h": (0,   180), "s": (0,  40),  "v": (190, 255)},
    # Black uniform:  any hue, low value
    {"h": (0,   180), "s": (0,  255), "v": (0,   55)},
    # Navy / dark blue:  blue hue, medium-high sat, low-medium value
    {"h": (100, 130), "s": (80, 255), "v": (30,  140)},
]


# ─── Per-track feature record ────────────────────────────────────────────────

@dataclass
class StaffFeatures:
    first_seen_ms: Optional[int] = None
    zone_visits: list = field(default_factory=list)  # list of (zone_id, enter_ms)
    zone_visit_counts: dict = field(default_factory=dict)  # zone_id → count
    total_zone_entries: int = 0
    uniform_match_frames: int = 0
    total_observed_frames: int = 0
    is_staff: bool = False


# ─── Behaviour Classifier ────────────────────────────────────────────────────

class BehaviourClassifier:
    """
    Evaluates rule-based behaviour features to detect staff.

    Rules (any one sufficient):
      R1 — Long presence:       time_in_store > LONG_PRESENCE_THRESHOLD_SEC
      R2 — High zone traversal: total_zone_entries > ZONE_TRAVERSAL_THRESHOLD
      R3 — High traversal rate: > TRAVERSAL_RATE_THRESHOLD zones in any 10-min window
      R4 — Back-and-forth:      any single zone visited > BACKTRACK_ZONE_THRESHOLD times
    """

    def classify(self, features: StaffFeatures, current_ms: int) -> tuple[bool, Optional[str]]:
        """
        Returns (is_staff, triggered_rule_name | None).
        Only runs if the track has been active for at least 60 s.
        """
        if features.first_seen_ms is None:
            return False, None

        time_in_store_ms = current_ms - features.first_seen_ms

        # Minimum observation window before behaviour rules activate
        if time_in_store_ms < 60_000:
            return False, None

        # R1 — Long presence
        if time_in_store_ms >= LONG_PRESENCE_THRESHOLD_SEC * 1000:
            return True, "R1_LONG_PRESENCE"

        # R2 — High zone traversal count
        if features.total_zone_entries >= ZONE_TRAVERSAL_THRESHOLD:
            return True, "R2_HIGH_TRAVERSAL"

        # R3 — High traversal rate in any recent window
        window_start = current_ms - TRAVERSAL_RATE_WINDOW_MS
        recent_entries = [ms for _, ms in features.zone_visits if ms >= window_start]
        if len(recent_entries) >= TRAVERSAL_RATE_THRESHOLD:
            return True, "R3_HIGH_TRAVERSAL_RATE"

        # R4 — Back-and-forth (same zone repeated)
        for zone_id, count in features.zone_visit_counts.items():
            if count >= BACKTRACK_ZONE_THRESHOLD:
                return True, "R4_BACKTRACK"

        return False, None


# ─── Uniform Colour Classifier ───────────────────────────────────────────────

class UniformClassifier:
    """
    Compares torso HSV histogram dominant bin against the staff colour palette.
    Returns True if the dominant colour falls inside any palette band.
    """

    def extract_torso_dominant_hsv(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """
        Returns mean HSV of the torso region (top 20%–60% of bounding box).
        bbox format: [x_center, y_center, width, height]
        """
        x_c, y_c, w, h = [int(v) for v in bbox]
        x1 = max(0, x_c - w // 2)
        y1 = max(0, y_c - h // 2)
        x2 = min(frame.shape[1], x_c + w // 2)
        y2 = min(frame.shape[0], y_c + h // 2)

        torso_y1 = y1 + int((y2 - y1) * 0.20)
        torso_y2 = y1 + int((y2 - y1) * 0.60)

        if torso_y2 <= torso_y1 or x2 <= x1:
            return None

        roi = frame[torso_y1:torso_y2, x1:x2]
        if roi.size == 0:
            return None

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return hsv_roi.mean(axis=(0, 1))  # [mean_H, mean_S, mean_V]

    def matches_palette(self, mean_hsv: np.ndarray) -> bool:
        """Checks if mean HSV falls inside any staff palette band."""
        h, s, v = float(mean_hsv[0]), float(mean_hsv[1]), float(mean_hsv[2])
        for band in STAFF_HSV_PALETTE:
            if (band["h"][0] <= h <= band["h"][1] and
                band["s"][0] <= s <= band["s"][1] and
                band["v"][0] <= v <= band["v"][1]):
                return True
        return False

    def classify_frame(self, frame: np.ndarray, bbox: list, features: StaffFeatures) -> bool:
        """
        Updates uniform match counters. Returns True if cumulative ratio
        exceeds UNIFORM_MATCH_FRAME_RATIO threshold.
        """
        features.total_observed_frames += 1
        mean_hsv = self.extract_torso_dominant_hsv(frame, bbox)

        if mean_hsv is not None and self.matches_palette(mean_hsv):
            features.uniform_match_frames += 1

        if features.total_observed_frames < 30:
            return False  # Need at least 30 frames before deciding

        ratio = features.uniform_match_frames / features.total_observed_frames
        return ratio >= UNIFORM_MATCH_FRAME_RATIO


# ─── Unified Staff Classifier ─────────────────────────────────────────────────

class StaffClassifier:
    """
    Combines BehaviourClassifier and UniformClassifier.
    Maintains per-track StaffFeatures state.
    Call update() every frame, read is_staff() any time.
    """

    def __init__(self):
        self._features: dict[int, StaffFeatures] = {}
        self._behaviour  = BehaviourClassifier()
        self._uniform    = UniformClassifier()

    def _get_or_create(self, track_id: int, current_ms: int) -> StaffFeatures:
        if track_id not in self._features:
            self._features[track_id] = StaffFeatures(first_seen_ms=current_ms)
        return self._features[track_id]

    def update(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: list,
        current_ms: int,
        current_zone: Optional[str] = None,
    ) -> bool:
        """
        Call once per frame per tracked person.
        Returns True if this track is now classified as staff.
        """
        feat = self._get_or_create(track_id, current_ms)

        # Once staff, always staff (sticky)
        if feat.is_staff:
            return True

        # --- Zone feature update ---
        if current_zone is not None:
            last_zone = feat.zone_visits[-1][0] if feat.zone_visits else None
            if current_zone != last_zone:
                # New zone entry
                feat.zone_visits.append((current_zone, current_ms))
                feat.total_zone_entries += 1
                feat.zone_visit_counts[current_zone] = (
                    feat.zone_visit_counts.get(current_zone, 0) + 1
                )

        # --- Uniform classifier (every frame) ---
        if self._uniform.classify_frame(frame, bbox, feat):
            feat.is_staff = True
            return True

        # --- Behaviour classifier (activates after 60 s) ---
        is_staff_by_behaviour, rule = self._behaviour.classify(feat, current_ms)
        if is_staff_by_behaviour:
            feat.is_staff = True
            return True

        return False

    def is_staff(self, track_id: int) -> bool:
        """Read the current staff classification for a track."""
        feat = self._features.get(track_id)
        return feat.is_staff if feat else False

    def get_features(self, track_id: int) -> Optional[StaffFeatures]:
        return self._features.get(track_id)
