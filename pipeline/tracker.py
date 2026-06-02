import time
import cv2
import numpy as np
from collections import deque
from pipeline.staff_classifier import StaffClassifier
from pipeline.cross_camera import CrossCameraMatcher

class VisitorTracker:
    def __init__(self, camera_id="CAM_MAIN_01", reentry_ttl_sec=600, trajectory_history_len=5): 
        self.camera_id = camera_id
        # State tracking for robust line crossing
        self.trajectory_history = {} # track_id -> deque of recent x_centers
        self.track_anchor_side = {}  # track_id -> "OUTSIDE" or "INSIDE"
        self.trajectory_history_len = trajectory_history_len
        
        self.counted_ids = set()
        
        self.visitor_sessions = {} # track_id -> {"zone": current_zone, "enter_time_ms": current_time_ms}
        self.last_dwell_emitted = {} # track_id -> timestamp ms
        
        self.total_entries = 0
        self.total_exits = 0
        
        # --- STAFF CLASSIFIER ---
        self.staff_classifier = StaffClassifier()
        
        # --- RE-ID STATE ---
        self.exited_visitor_cache = {}
        self.reentry_ttl_sec = reentry_ttl_sec
        self.id_mapping = {}

        # --- QUEUE STATE ---
        self.joined_billing_queue = {}          # track_id -> bool
        self.billing_queue_pending_abandon = {}  # track_id -> bool

        # --- OCCLUSION STATE ---
        self.active_signatures = {}             # track_id -> np.ndarray
        self.lost_track_cache = {}              # track_id -> dict
        self.last_bboxes = {}                   # track_id -> list
        
        # --- CROSS-CAMERA MATCHER ---
        self.cross_camera_matcher = CrossCameraMatcher()

    def update_trajectory(self, track_id, x_center):
        if track_id not in self.trajectory_history:
            self.trajectory_history[track_id] = deque(maxlen=self.trajectory_history_len)
        self.trajectory_history[track_id].append(x_center)
        
    def initialize_track_anchor(self, track_id, line_x, door_buffer_pixels=80):
        """
        Initializes the track anchor side based on position and direction of motion.
        If the track starts near the door, we defer assignment until we have enough frames
        (at least 2) to calculate velocity and determine if they entered or exited.
        """
        if track_id in self.track_anchor_side:
            return
            
        hist = self.trajectory_history.get(track_id)
        if not hist:
            return
            
        first_x = hist[0]
        latest_x = hist[-1]
        
        # Check if the track's first position is close to the counting line
        if abs(first_x - line_x) < door_buffer_pixels:
            if len(hist) < 2:
                # Defer initialization until we have more history
                return
            
            # Compute velocity (dx)
            dx = latest_x - first_x
            if dx > 0:
                # Moving right -> entering -> started OUTSIDE
                self.track_anchor_side[track_id] = "OUTSIDE"
            elif dx < 0:
                # Moving left -> exiting -> started INSIDE
                self.track_anchor_side[track_id] = "INSIDE"
            else:
                # If static, fallback to spatial check on first_x
                self.track_anchor_side[track_id] = "OUTSIDE" if first_x < line_x else "INSIDE"
        else:
            # Not close to the line: use standard spatial check on first_x
            self.track_anchor_side[track_id] = "OUTSIDE" if first_x < line_x else "INSIDE"

    def get_smoothed_x(self, track_id):
        if track_id not in self.trajectory_history or len(self.trajectory_history[track_id]) == 0:
            return None
        hist = self.trajectory_history[track_id]
        return sum(hist) / len(hist)

    def get_current_time_ms(self):
        return int(time.time() * 1000)

    def is_staff_member(self, frame, box, current_ms=None, current_zone=None, track_id=None):
        """
        Returns True if this track is classified as a staff member.
        Delegates to StaffClassifier which combines uniform colour and behaviour rules.
        When called without track_id (legacy), falls back to False.
        """
        if track_id is None or frame is None:
            return False
        ts = current_ms or self.get_current_time_ms()
        return self.staff_classifier.update(
            track_id=track_id,
            frame=frame,
            bbox=box,
            current_ms=ts,
            current_zone=current_zone,
        )

    def is_staff(self, track_id: int) -> bool:
        """Quick read of the sticky staff flag for a track_id."""
        return self.staff_classifier.is_staff(track_id)

    def is_inside_zone(self, x, y, zone_coords):
        if not zone_coords:
            return False
        # Polygon support: coords is a list of points e.g. [[x1, y1], [x2, y2], ...]
        if isinstance(zone_coords, (list, tuple)) and len(zone_coords) > 0 and isinstance(zone_coords[0], (list, tuple)):
            poly = np.array(zone_coords, dtype=np.int32)
            dist = cv2.pointPolygonTest(poly, (float(x), float(y)), False)
            return dist >= 0
        # Bounding box support: coords is [x1, y1, x2, y2]
        if len(zone_coords) == 4 and all(isinstance(v, (int, float)) for v in zone_coords):
            x1, y1, x2, y2 = zone_coords
            return x1 <= x <= x2 and y1 <= y <= y2
        return False

    def get_queue_depth(self):
        """Returns the number of people currently in the billing queue."""
        return sum(1 for val in self.joined_billing_queue.values() if val)

    def get_mapped_track_id(self, track_id):
        """Returns the original track_id if this is a returning visitor."""
        return self.id_mapping.get(track_id, track_id)

    def extract_appearance_signature(self, frame, bbox):
        """Extracts an HSV color histogram from the bounding box (focusing on the torso)."""
        x_c, y_c, w, h = [int(v) for v in bbox]
        # Convert xywh to xyxy
        x1 = max(0, int(x_c - w/2))
        y1 = max(0, int(y_c - h/2))
        x2 = min(frame.shape[1], int(x_c + w/2))
        y2 = min(frame.shape[0], int(y_c + h/2))
        
        # Focus on upper body / torso (top 20% to 60% of bbox) to avoid changing legs/shoes
        torso_y1 = y1 + int((y2 - y1) * 0.2)
        torso_y2 = y1 + int((y2 - y1) * 0.6)
        
        if torso_y2 <= torso_y1 or x2 <= x1:
            return None # Invalid bbox
            
        roi = frame[torso_y1:torso_y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Compute histogram (Hue, Saturation)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def register_exit(self, original_track_id, frame, bbox):
        """Called when a visitor exits. Stores their signature in the cache."""
        signature = self.extract_appearance_signature(frame, bbox)
        if signature is not None:
            self.exited_visitor_cache[original_track_id] = {
                "time": time.time(),
                "signature": signature,
                "bbox": bbox
            }
            
    def check_reentry(self, new_track_id, frame, bbox, similarity_threshold=0.8):
        """Compares a new entry against the exit cache to find returning visitors."""
        now = time.time()
        
        # Clean up expired cache entries
        expired_ids = [vid for vid, data in self.exited_visitor_cache.items() if (now - data["time"]) > self.reentry_ttl_sec]
        for vid in expired_ids:
            del self.exited_visitor_cache[vid]
            
        new_signature = self.extract_appearance_signature(frame, bbox)
        if new_signature is None or len(self.exited_visitor_cache) == 0:
            return False, None
            
        best_match_id = None
        best_similarity = -1
        
        for old_id, data in self.exited_visitor_cache.items():
            old_signature = data["signature"]
            # Compute cosine similarity
            similarity = np.dot(new_signature, old_signature) / (np.linalg.norm(new_signature) * np.linalg.norm(old_signature) + 1e-6)
            
            # Simple spatial check: bbox size shouldn't drastically change (e.g., adult vs child)
            old_h = data["bbox"][3]
            new_h = bbox[3]
            height_ratio = min(old_h, new_h) / max(old_h, new_h)
            
            if similarity > similarity_threshold and height_ratio > 0.7:
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = old_id
                    
        if best_match_id is not None:
            # Match found! Map the new track_id to the old track_id
            self.id_mapping[new_track_id] = best_match_id
            # Remove from cache so they can't re-enter multiple times simultaneously
            del self.exited_visitor_cache[best_match_id]
            return True, best_match_id
            
        return False, None

    def update_active_signature(self, track_id, frame, bbox):
        """Extract and cache the appearance signature for a currently tracked visitor."""
        sig = self.extract_appearance_signature(frame, bbox)
        if sig is not None:
            self.active_signatures[track_id] = sig
            self.last_bboxes[track_id] = bbox

    def register_lost_track(self, track_id, last_zone, last_bbox=None, current_time_ms=None):
        """Move an active signature to the lost track cache when a track is lost."""
        sig = self.active_signatures.get(track_id)
        bbox = last_bbox if last_bbox is not None else self.last_bboxes.get(track_id)
        if sig is not None and bbox is not None:
            ts = current_time_ms or int(time.time() * 1000)
            self.lost_track_cache[track_id] = {
                "time_ms": ts,
                "signature": sig,
                "bbox": bbox,
                "zone": last_zone
            }
            # Clean up active signature
            if track_id in self.active_signatures:
                del self.active_signatures[track_id]
            if track_id in self.last_bboxes:
                del self.last_bboxes[track_id]

    def check_lost_recovery(self, new_track_id, frame, bbox, current_zone, current_time_ms=None, similarity_threshold=0.82):
        """Checks if a newly detected track matches any recently lost track inside the store."""
        now = current_time_ms or int(time.time() * 1000)
        
        new_signature = self.extract_appearance_signature(frame, bbox)
        if new_signature is None or len(self.lost_track_cache) == 0:
            return False, None
            
        best_match_id = None
        best_similarity = -1
        
        # Clean up lost tracks older than 30 seconds
        expired_ids = [vid for vid, data in self.lost_track_cache.items() if (now - data["time_ms"]) > 30000]
        for vid in expired_ids:
            del self.lost_track_cache[vid]
            
        for old_id, data in self.lost_track_cache.items():
            # Temporal check: must have been lost within last 30s
            if (now - data["time_ms"]) > 30000:
                continue
                
            old_signature = data["signature"]
            similarity = np.dot(new_signature, old_signature) / (np.linalg.norm(new_signature) * np.linalg.norm(old_signature) + 1e-6)
            
            # Size check
            old_h = data["bbox"][3]
            new_h = bbox[3]
            height_ratio = min(old_h, new_h) / max(old_h, new_h)
            
            # Zone match check (usually occluded person reappears in same or adjacent zone)
            zone_match = (data["zone"] == current_zone) or (data["zone"] is None) or (current_zone is None)
            
            if similarity > similarity_threshold and height_ratio > 0.7 and zone_match:
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = old_id
                    
        if best_match_id is not None:
            # Map the new track to the lost track ID
            self.id_mapping[new_track_id] = best_match_id
            
            # Carry over active states to the new track mapping
            self.joined_billing_queue[new_track_id] = self.joined_billing_queue.get(best_match_id, False)
            self.billing_queue_pending_abandon[new_track_id] = self.billing_queue_pending_abandon.get(best_match_id, False)
            
            # Restore session to visitor_sessions so they don't trigger new ZONE_ENTER events
            if best_match_id in self.visitor_sessions:
                self.visitor_sessions[new_track_id] = self.visitor_sessions[best_match_id]
                
            # Remove from lost cache
            del self.lost_track_cache[best_match_id]
            
            return True, best_match_id
            
        return False, None

    def reap_expired_lost_tracks(self, current_time_ms, line_x, store_id, camera_id, send_event_fn):
        """Reaps expired lost tracks and performs graceful offline exits if lost close to the door."""
        now = current_time_ms
        expired_ids = []
        for vid, data in self.lost_track_cache.items():
            if (now - data["time_ms"]) > 30000:
                expired_ids.append(vid)
                
                mapped_id = self.get_mapped_track_id(vid)
                
                # Check queue abandonment on reap
                if self.joined_billing_queue.get(vid, False) or self.billing_queue_pending_abandon.get(vid, False):
                    self.joined_billing_queue[vid] = False
                    self.billing_queue_pending_abandon[vid] = False
                    q_depth = self.get_queue_depth()
                    send_event_fn(
                        store_id, camera_id, "BILLING_QUEUE_ABANDON", mapped_id, 0.5, 
                        self.is_staff(vid), zone_id="BILLING", metadata={"queue_depth": q_depth}
                    )
                
                # Exited check: if last known position was close to the line, trigger offline exit
                last_bbox = data["bbox"]
                x_center = last_bbox[0]
                
                # If they were close to the exit boundary and we reap them, trigger EXIT
                if abs(x_center - line_x) < 80:
                    send_event_fn(store_id, camera_id, "EXIT", mapped_id, 0.5, self.is_staff(vid))
                    self.total_exits += 1
                    
        for vid in expired_ids:
            if vid in self.lost_track_cache:
                del self.lost_track_cache[vid]

