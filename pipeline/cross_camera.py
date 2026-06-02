import os
import json
import time
import numpy as np

class CrossCameraMatcher:
    def __init__(self, filepath="data/cross_camera_registry.json", ttl_sec=300):
        self.filepath = filepath
        self.ttl_sec = ttl_sec

    def _load(self) -> dict:
        if not os.path.exists(self.filepath):
            return {}
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self, data: dict):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def register_visitor(self, visitor_id: str, camera_id: str, signature):
        if signature is None:
            return
        # Ensure signature is list of floats for JSON serialization
        if hasattr(signature, "tolist"):
            signature_list = signature.tolist()
        else:
            signature_list = list(signature)

        data = self._load()
        now = time.time()
        
        # Clean up expired entries (older than TTL)
        data = {vid: item for vid, item in data.items() if (now - item["timestamp"]) < self.ttl_sec}
        
        data[visitor_id] = {
            "camera_id": camera_id,
            "timestamp": now,
            "signature": signature_list
        }
        self._save(data)

    def find_match(self, camera_id: str, signature, similarity_threshold=0.82) -> str:
        if signature is None:
            return None
        
        data = self._load()
        now = time.time()
        best_match_id = None
        best_similarity = -1

        new_sig = np.array(signature)
        new_norm = np.linalg.norm(new_sig) + 1e-6

        for vid, item in data.items():
            # Temporal constraint check
            if (now - item["timestamp"]) > self.ttl_sec:
                continue
            
            # Avoid matching on the exact same camera (handled by local tracker/reentry)
            if item["camera_id"] == camera_id:
                continue

            old_sig = np.array(item["signature"])
            old_norm = np.linalg.norm(old_sig) + 1e-6
            
            # Cosine similarity
            similarity = np.dot(new_sig, old_sig) / (new_norm * old_norm)
            
            if similarity > similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = vid

        return best_match_id
