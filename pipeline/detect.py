from IPython.core import interactiveshell
import cv2
import json
import time
from ultralytics import YOLO

from pipeline.tracker import VisitorTracker
from pipeline.emit import send_event



def run_detection(
    video_path,
    output_path,
    camera_id
):
    # Load config
    try:
        with open("data/store_layout.json", "r") as f:
            layout = json.load(f)
            STORE_ID = layout.get("store_id", "STORE_BLR_002")
            zones = {}
            # Try to find the matching camera layout first
            cam_found = False
            for cam in layout.get("cameras", []):
                if cam.get("camera_id") == camera_id:
                    cam_found = True
                    if "zones" in cam:
                        for z in cam["zones"]:
                            zones[z["zone_id"]] = tuple(z["coords"])

            print("=" * 50)
            print(f"CAMERA_ID = {camera_id}")
            print(f"ZONES = {zones}")
            print("=" * 50)               
            # Fallback if specific camera not found
            if not cam_found:
                for cam in layout.get("cameras", []):
                    if cam.get("type") == "zones":
                        for z in cam.get("zones", []):
                            zones[z["zone_id"]] = tuple(z["coords"])
    except Exception:
        STORE_ID = "STORE_BLR_002"
        zones = {}

    CAMERA_ID = camera_id
    
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"FRAME WIDTH = {frame_width}")
    print(f"FRAME HEIGHT = {frame_height}")
    print(
    f"FRAME SIZE = "
    f"{frame_width} x {frame_height}"
)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not zones:
        raise ValueError(
            f"No zones loaded for {camera_id}"
        )
        
    line_x = frame_width // 2
    tracker = VisitorTracker(camera_id=CAMERA_ID)

    print(f"Starting detection on {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", iou=0.75, conf=0.3, classes=[0], verbose=False)
        result = results[0]
        annotated_frame = result.plot()
        
        cv2.line(annotated_frame, (line_x, 0), (line_x, frame_height), (255, 0, 0), 2)
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        color_idx = 0
        occupancies = {z: 0 for z in zones.keys()}
        
        for z_id, coords in zones.items():
            color = colors[color_idx % len(colors)]
            cv2.rectangle(annotated_frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
            cv2.putText(annotated_frame, z_id, (coords[0]+10, coords[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            color_idx += 1
            
        current_time_ms = tracker.get_current_time_ms()
        
        track_ids = []
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xywh.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            
            for track_id, box, conf in zip(track_ids, boxes, confs):
                x_center = box[0]
                y_feet = box[1] + box[3] 


                # ── Zone detection ──────────────────────────────────────────
                current_zone = None
                for z_id, coords in zones.items():
                    if tracker.is_inside_zone(x_center, y_feet, coords):
                        current_zone = z_id
                        occupancies[z_id] += 1
                        break
                if camera_id == "CAM_BILLING_01":
                    print(
                        f"X={int(x_center)} "
                        f"Y={int(y_feet)} "
        f"ZONE={current_zone}"
                        f"TRACK={track_id}"
                        f"CURRENT_ZONE={current_zone}"
                    )  

                # If this track_id is completely new to the tracker, check lost recovery
                if track_id not in tracker.trajectory_history:
                    is_recovered, lost_id = tracker.check_lost_recovery(
                        track_id, frame, box, current_zone, current_time_ms=current_time_ms
                    )
                    if is_recovered:
                        # Carry over anchor side and counted status
                        if lost_id in tracker.track_anchor_side:
                            tracker.track_anchor_side[track_id] = tracker.track_anchor_side[lost_id]
                        if lost_id in tracker.counted_ids:
                            tracker.counted_ids.add(track_id)
                    else:
                        # Not recovered locally: check cross-camera match
                        sig = tracker.extract_appearance_signature(frame, box)
                        if sig is not None:
                            matched_visitor_id = tracker.cross_camera_matcher.find_match(CAMERA_ID, sig)
                            if matched_visitor_id is not None:
                                tracker.id_mapping[track_id] = matched_visitor_id
                                tracker.counted_ids.add(track_id)
                                tracker.track_anchor_side[track_id] = "INSIDE"
                            else:
                                visitor_id_str = str(track_id)
                                tracker.cross_camera_matcher.register_visitor(visitor_id_str, CAMERA_ID, sig)

                # Update smoothed trajectory
                tracker.update_trajectory(track_id, x_center)
                smoothed_x = tracker.get_smoothed_x(track_id)
                
                # Determine initial anchor side using robust motion-based deferred initialization
                tracker.initialize_track_anchor(track_id, line_x)

                # Get mapped ID to handle returning visitors (REENTRY)
                mapped_id = tracker.get_mapped_track_id(track_id)

                # ── Staff classification (needs current_zone for behaviour rules) ─
                # update() accumulates features every frame; returns sticky True once classified
                tracker.is_staff_member(
                    frame, box,
                    current_ms=current_time_ms,
                    current_zone=current_zone,
                    track_id=track_id
                )
                # Read the sticky flag — once True it never reverts
                staff = tracker.is_staff(track_id)

                # Update active signature for in-store occlusion recovery
                tracker.update_active_signature(track_id, frame, box)

                # Line crossing logic using smoothed state transition
                if track_id in tracker.track_anchor_side and track_id not in tracker.counted_ids and smoothed_x is not None:
                    anchor = tracker.track_anchor_side[track_id]
                    if anchor == "OUTSIDE" and smoothed_x >= line_x:
                        is_reentry, matched_id = tracker.check_reentry(track_id, frame, box)
                        if is_reentry:
                            mapped_id = matched_id
                            send_event(STORE_ID, CAMERA_ID, "REENTRY", mapped_id, conf, staff)
                        else:
                            send_event(STORE_ID, CAMERA_ID, "ENTRY", mapped_id, conf, staff)
                        tracker.total_entries += 1
                        tracker.counted_ids.add(track_id)
                        tracker.track_anchor_side[track_id] = "INSIDE"
                    elif anchor == "INSIDE" and smoothed_x <= line_x:
                        send_event(STORE_ID, CAMERA_ID, "EXIT", mapped_id, conf, staff)
                        tracker.total_exits += 1
                        tracker.counted_ids.add(track_id)
                        tracker.register_exit(mapped_id, frame, box)
                        tracker.track_anchor_side[track_id] = "OUTSIDE"
                        # Successful checkout -> clear pending queue abandonment
                        tracker.billing_queue_pending_abandon[track_id] = False

                # ── Zone enter/exit events ──────────────────────────────────
                if current_zone is not None:
                    if track_id not in tracker.visitor_sessions:
                        tracker.visitor_sessions[track_id] = {"zone": current_zone, "enter_time_ms": current_time_ms}
                        tracker.last_dwell_emitted[track_id] = current_time_ms
                        send_event(STORE_ID, CAMERA_ID, "ZONE_ENTER", mapped_id, conf, staff, zone_id=current_zone)
                        
                        # Trigger abandonment if entering Skincare/Cosmetics and pending abandon
                        if current_zone in ("SKIN_CARE", "MAKEUP") and tracker.billing_queue_pending_abandon.get(track_id, False):
                            tracker.billing_queue_pending_abandon[track_id] = False
                            q_depth = tracker.get_queue_depth()
                            send_event(STORE_ID, CAMERA_ID, "BILLING_QUEUE_ABANDON", mapped_id, conf, staff, zone_id="BILLING", metadata={"queue_depth": q_depth})
                    else:
                        previous_zone = tracker.visitor_sessions[track_id]["zone"]
                        if previous_zone != current_zone:
                            dwell_time = current_time_ms - tracker.visitor_sessions[track_id]["enter_time_ms"]
                            send_event(STORE_ID, CAMERA_ID, "ZONE_EXIT", mapped_id, conf, staff, zone_id=previous_zone, dwell_ms=dwell_time)
                            
                            # Mark pending abandon if exiting billing queue
                            if previous_zone == "BILLING" and tracker.joined_billing_queue.get(track_id, False):
                                tracker.joined_billing_queue[track_id] = False
                                tracker.billing_queue_pending_abandon[track_id] = True
                            
                            tracker.visitor_sessions[track_id] = {"zone": current_zone, "enter_time_ms": current_time_ms}
                            tracker.last_dwell_emitted[track_id] = current_time_ms
                            send_event(STORE_ID, CAMERA_ID, "ZONE_ENTER", mapped_id, conf, staff, zone_id=current_zone)
                            
                            # Trigger abandonment if entering Skincare/Cosmetics and pending abandon
                            if current_zone in ("SKIN_CARE", "MAKEUP") and tracker.billing_queue_pending_abandon.get(track_id, False):
                                tracker.billing_queue_pending_abandon[track_id] = False
                                q_depth = tracker.get_queue_depth()
                                send_event(STORE_ID, CAMERA_ID, "BILLING_QUEUE_ABANDON", mapped_id, conf, staff, zone_id="BILLING", metadata={"queue_depth": q_depth})
                        else:
                            # Billing queue joining detection
                            if current_zone == "BILLING":

                                total_dwell = current_time_ms - tracker.visitor_sessions[track_id]["enter_time_ms"]

                                print(
                                    f"[BILLING DEBUG] "
                                    f"TRACK={track_id} "
                                    f"DWELL={total_dwell} "
                                    f"JOINED={tracker.joined_billing_queue.get(track_id, False)}"
                                )

                                if not tracker.joined_billing_queue.get(track_id, False):

                                    if total_dwell >= 10000:

                                        print(
                                            f"[QUEUE JOIN] "
                                            f"TRACK={track_id}"
                                        )

                                        tracker.joined_billing_queue[track_id] = True
                                        tracker.billing_queue_pending_abandon[track_id] = False

                                        q_depth = tracker.get_queue_depth()

                                        send_event(
                                            STORE_ID,
                                            CAMERA_ID,
                                            "BILLING_QUEUE_JOIN",
                                            mapped_id,
                                            conf,
                                            staff,
                                            zone_id="BILLING",
                                            metadata={"queue_depth": q_depth}
                                        )
                            
                            
                else:
                    if track_id in tracker.visitor_sessions:
                        previous_zone = tracker.visitor_sessions[track_id]["zone"]
                        dwell_time = current_time_ms - tracker.visitor_sessions[track_id]["enter_time_ms"]
                        send_event(STORE_ID, CAMERA_ID, "ZONE_EXIT", mapped_id, conf, staff, zone_id=previous_zone, dwell_ms=dwell_time)
                        
                        # Mark pending abandon if exiting billing queue
                        if previous_zone == "BILLING" and tracker.joined_billing_queue.get(track_id, False):
                            tracker.joined_billing_queue[track_id] = False
                            tracker.billing_queue_pending_abandon[track_id] = True

                        del tracker.visitor_sessions[track_id]
                        if track_id in tracker.last_dwell_emitted:
                            del tracker.last_dwell_emitted[track_id]

        # --- Identify lost tracks and reap expired entries ---
        detected_set = set(track_ids)
        active_track_ids = list(tracker.active_signatures.keys())
        for active_id in active_track_ids:
            if active_id not in detected_set:
                last_session = tracker.visitor_sessions.get(active_id)
                last_zone = last_session["zone"] if last_session else None
                tracker.register_lost_track(active_id, last_zone, current_time_ms=current_time_ms)
                
        tracker.reap_expired_lost_tracks(
            current_time_ms=current_time_ms,
            line_x=line_x,
            store_id=STORE_ID,
            camera_id=CAMERA_ID,
            send_event_fn=send_event
        )
                            
        # ── HUD overlay ─────────────────────────────────────────────────────
        cv2.putText(annotated_frame, f"ENTRIES: {tracker.total_entries}", (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(annotated_frame, f"EXITS:   {tracker.total_exits}",   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if 'annotated_frame' in locals():
            out.write(annotated_frame)
        
    cap.release()
    out.release()
    print(f"Finished processing! Saved to {output_path}")

CAMERA_MAPPING = {
    "data/videos/entry_camera.mp4": "CAM_ENTRY_01",
    "data/videos/entry_camera_2.mp4": "CAM_ENTRY_02",
    
    "data/videos/floor_camera_1.mp4": "CAM_ZONES_01",
    "data/videos/floor_camera_2.mp4": "CAM_ZONES_02",
    "data/videos/billing_camera.mp4": "CAM_BILLING_01"
}

if __name__ == "__main__":

    for video_path, camera_id in CAMERA_MAPPING.items():

        output_name = (
            video_path.split("/")[-1]
            .replace(".mp4", "_output.mp4")
        )

        run_detection(
            video_path=video_path,
            output_path=f"data/{output_name}",
            camera_id=camera_id
        )