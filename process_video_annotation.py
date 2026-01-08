import cv2
import torch
import numpy as np
import json
import os
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import argparse
from collections import Counter, defaultdict

def process_video(video_path, output_json_path, device='cuda:0'):
    print(f"Processing video: {video_path}")
    print(f"Output JSON: {output_json_path}")
    print(f"Device: {device}")

    # 1. Initialize YOLOv11 Pose Model
    print("Loading YOLOv11 pose model...")
    pose_model = YOLO('yolo11n-pose.pt')
    pose_model.to(device)

    # 2. Initialize Gazelle Model
    print("Loading Gazelle model...")
    try:
        gazelle_model, gazelle_transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14')
        gazelle_model.to(device)
        gazelle_model.eval()
    except Exception as e:
        print(f"Error loading Gazelle model: {e}")
        return

    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {frame_count} frames")

    # Settings for 30fps and 20 seconds limit
    TARGET_FPS = 30
    MAX_DURATION_SEC = 300
    
    frame_interval = max(1, int(round(fps / TARGET_FPS)))
    output_fps = fps / frame_interval
    max_process_frames = int(fps * MAX_DURATION_SEC)
    
    print(f"Processing at approx {output_fps:.2f} fps (interval: {frame_interval})")
    print(f"Limiting to first {MAX_DURATION_SEC} seconds ({max_process_frames} frames)")

    # Helper function to get face bbox from keypoints
    def get_face_bbox(keypoints_xy, keypoints_conf, img_w, img_h):
        # COCO keypoint indices: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar
        FACE_INDICES = [0, 1, 2, 3, 4]
        CONF_THRESH = 0.3
        
        valid_kps = []
        face_conf_sum = 0
        for i in FACE_INDICES:
            face_conf_sum += keypoints_conf[i]
            if keypoints_conf[i] > CONF_THRESH:
                valid_kps.append(keypoints_xy[i])
        
        if len(valid_kps) <= 1 or face_conf_sum < 1.5:
            return None
            
        kps = np.array(valid_kps)
        min_x, min_y = np.min(kps, axis=0)
        max_x, max_y = np.max(kps, axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        
        # Add padding
        pad_x = w * 0.5
        pad_y = h * 2.0 # More padding on top for forehead/hair
        
        x1 = max(0, min_x - pad_x)
        y1 = max(0, min_y - pad_y)
        x2 = min(img_w, max_x + pad_x)
        y2 = min(img_h, max_y + pad_y)
        
        return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # xywh

    # Helper function for Gazelle inference
    def estimate_gaze(image_rgb, bbox_xywh):
        pil_image = Image.fromarray(image_rgb)
        w, h = pil_image.size
        
        img_tensor = gazelle_transform(pil_image).unsqueeze(0).to(device)
        
        x, y, bw, bh = bbox_xywh
        norm_bbox = (x/w, y/h, (x+bw)/w, (y+bh)/h) # xmin, ymin, xmax, ymax normalized
        
        input_data = {
            "images": img_tensor,
            "bboxes": [[norm_bbox]]
        }
        
        with torch.no_grad():
            output = gazelle_model(input_data)
            
        heatmap = output["heatmap"][0][0].cpu().numpy()
        
        y_idx, x_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        gaze_x = (x_idx / 64.0) * w
        gaze_y = (y_idx / 64.0) * h
        
        return (gaze_x, gaze_y)

    # --- PASS 1: Collect Raw Data ---
    print("Pass 1: Collecting raw detections...")
    raw_tracks = defaultdict(dict) # {person_id: {frame_idx: data}}
    
    process_limit = min(frame_count, max_process_frames)
    
    # We need to keep track of which frames we actually processed to map back later
    processed_frames_indices = []

    for frame_idx in tqdm(range(process_limit)):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval != 0:
            continue
            
        processed_frames_indices.append(frame_idx)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO Tracking
        yolo_results = pose_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device=device,
            conf=0.4,
            iou=0.72
        )
        
        if yolo_results and yolo_results[0].boxes.id is not None:
            res = yolo_results[0]
            track_ids = res.boxes.id.int().cpu().tolist()
            boxes_xywh = res.boxes.xywh.cpu().tolist()
            keypoints = res.keypoints.cpu()
            box_confs = res.boxes.conf.cpu().tolist()
            
            for i, track_id in enumerate(track_ids):
                person_id = int(track_id)
                conf = box_confs[i]
                
                # Filter: Only consider persons below the horizontal center line
                # boxes_xywh[i] is [center_x, center_y, w, h]
                if boxes_xywh[i][1] <= height / 3:
                    continue

                bbox = boxes_xywh[i] # center_x, center_y, w, h
                # Convert to top-left xywh
                tl_x = bbox[0] - bbox[2]/2
                tl_y = bbox[1] - bbox[3]/2
                bbox_tlwh = [tl_x, tl_y, bbox[2], bbox[3]]
                
                kps_xy = keypoints.xy[i].numpy()
                kps_conf = keypoints.conf[i].numpy()
                
                face_bbox = get_face_bbox(kps_xy, kps_conf, width, height)
                
                if face_bbox is None:
                    continue
                
                # Ensure face_bbox is valid
                fx, fy, fw, fh = face_bbox
                fx = max(0, min(fx, width - 1))
                fy = max(0, min(fy, height - 1))
                fw = max(1, min(fw, width - fx))
                fh = max(1, min(fh, height - fy))
                face_bbox = [fx, fy, fw, fh]
                
                # Check intersection
                bx, by, bw, bh = bbox_tlwh
                x1 = max(fx, bx)
                y1 = max(fy, by)
                x2 = min(fx + fw, bx + bw)
                y2 = min(fy + fh, by + bh)
                
                if x2 > x1 and y2 > y1:
                    face_bbox = [x1, y1, x2 - x1, y2 - y1]
                else:
                    continue

                # Estimate Gaze (Raw)
                gaze_point = estimate_gaze(frame_rgb, face_bbox)
                
                raw_tracks[person_id][frame_idx] = {
                    "bbox": bbox_tlwh,
                    "face_bbox": face_bbox,
                    "pose_keypoints": kps_xy.tolist(),
                    "gaze_point": gaze_point,
                    "conf": float(conf),
                    "keypoints_conf": kps_conf.tolist()
                }

    cap.release()

    # --- PASS 2: Smoothing & Target Calculation ---
    print("Pass 2: Smoothing and calculating targets...")
    
    smoothed_tracks = defaultdict(dict) # {person_id: {frame_idx: smoothed_data}}
    
    # 2a. Smooth BBox and Face BBox (Window 5) & Fill Gaps
    for pid, frames_data in raw_tracks.items():
        sorted_frames = sorted(frames_data.keys())
        if not sorted_frames:
            continue
            
        min_f, max_f = sorted_frames[0], sorted_frames[-1]
        
        # Iterate through the full range of frames for this person
        # We step by frame_interval to match the processed frames
        current_f = min_f
        while current_f <= max_f:
            # Define window: [t-2, t-1, t, t+1, t+2]
            # We need to find the actual frame indices that exist in our processed set
            
            window_indices = [
                current_f - 2*frame_interval,
                current_f - frame_interval,
                current_f,
                current_f + frame_interval,
                current_f + 2*frame_interval
            ]
            
            window_frames = [f for f in window_indices if f in frames_data]
            
            if not window_frames:
                current_f += frame_interval
                continue
            
            # Calculate averages
            avg_bbox = np.mean([frames_data[f]['bbox'] for f in window_frames], axis=0).tolist()
            avg_face = np.mean([frames_data[f]['face_bbox'] for f in window_frames], axis=0).tolist()
            
            # For other data, prefer current frame if exists, else nearest neighbor in window
            if current_f in frames_data:
                base_frame = current_f
            else:
                # Find nearest frame in window_frames to current_f
                base_frame = min(window_frames, key=lambda x: abs(x - current_f))
                
            base_data = frames_data[base_frame]
            
            # If current frame is missing (filled), we don't have a gaze point
            # The prompt says "Cancel Gaze point smoothing", so we use raw gaze if available.
            # If filled, gaze_point is None.
            gaze_point = frames_data[current_f]['gaze_point'] if current_f in frames_data else None
            
            smoothed_tracks[pid][current_f] = {
                "bbox": avg_bbox,
                "face_bbox": avg_face,
                "pose_keypoints": base_data['pose_keypoints'],
                "gaze_point": gaze_point, # Raw or None
                "conf": base_data['conf'],
                "keypoints_conf": base_data['keypoints_conf'],
                "is_filled": current_f not in frames_data
            }
            
            current_f += frame_interval

    # 2b. Calculate Gaze Targets (Window 5)
    # First, organize by frame for easy lookup
    frame_to_persons = defaultdict(list)
    for pid, tracks in smoothed_tracks.items():
        for f_idx, data in tracks.items():
            p_entry = data.copy()
            p_entry['id'] = pid
            frame_to_persons[f_idx].append(p_entry)

    # Calculate Instant Targets
    instant_targets = defaultdict(dict) # {pid: {frame_idx: target_id}}
    
    for f_idx, persons in frame_to_persons.items():
        for p in persons:
            if p['gaze_point'] is None:
                continue
                
            gx, gy = p['gaze_point']
            best_target = 0
            min_dist = float('inf')
            
            for other in persons:
                if p['id'] == other['id']:
                    continue
                
                # Use face center if available, else bbox center
                if other['face_bbox']:
                    fx, fy, fw, fh = other['face_bbox']
                    tx, ty = fx + fw/2, fy + fh/2
                else:
                    ox, oy, ow, oh = other['bbox']
                    tx, ty = ox + ow/2, oy + oh/2
                    
                dist = ((gx - tx)**2 + (gy - ty)**2)**0.5
                
                # Simple threshold check? Or just min dist?
                # Let's assume if it's within the frame, it's a candidate.
                if dist < min_dist:
                    min_dist = dist
                    best_target = other['id']
            
            instant_targets[p['id']][f_idx] = best_target

    # Calculate Final Targets using Logic (Window 9)
    final_results = []
    
    # Keep track of the last confirmed target for each person to implement "hold" logic
    last_final_targets = {} 

    for f_idx in processed_frames_indices:
        frame_res = {
            "frame_index": f_idx,
            "timestamp": f_idx / fps,
            "persons": []
        }
        
        if f_idx in frame_to_persons:
            for p in frame_to_persons[f_idx]:
                pid = p['id']
                
                # Window: [t-2, t-1, t, t+1, t+2]
                # range(-3, 4) gives: -3, -2, -1, 0, 1, 2, 3
                window_indices = [
                    f_idx + k * frame_interval for k in range(-3, 4)
                ]
                
                # Collect valid targets in time order
                valid_sequence = []
                for w_idx in window_indices:
                    if pid in instant_targets and w_idx in instant_targets[pid]:
                        tgt = instant_targets[pid][w_idx]
                        if tgt != 0:
                            valid_sequence.append(tgt)
                
                final_target = 0
                if valid_sequence:
                    # Count changes
                    num_changes = 0
                    for i in range(len(valid_sequence) - 1):
                        if valid_sequence[i] != valid_sequence[i+1]:
                            num_changes += 1
                    
                    # Logic: If changes >= 1, maintain previous target (stabilize)
                    if num_changes >= 1:
                        if pid in last_final_targets:
                            final_target = last_final_targets[pid]
                        else:
                            # Fallback if no history: use the first one in the window
                            final_target = valid_sequence[0]
                    else:
                        # If stable (0 changes), use Voting (Mode) - effectively the new value
                        final_target = Counter(valid_sequence).most_common(1)[0][0]
                
                # Update history if we have a valid target
                if final_target != 0:
                    last_final_targets[pid] = final_target
                
                p['gaze_target'] = final_target
                
                frame_res["persons"].append({
                    "id": pid,
                    "bbox": p['bbox'],
                    "pose_keypoints": p['pose_keypoints'],
                    "face_bbox": p['face_bbox'],
                    "gaze_point": p['gaze_point'],
                    "gaze_target": final_target,
                    "conf": p['conf']
                })
        
        final_results.append(frame_res)

    # --- PASS 3: Visualization & Output ---
    print("Pass 3: Generating output video...")
    
    cap = cv2.VideoCapture(video_path)
    output_video_path = output_json_path.replace('.json', '_vis.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))
    
    # Map results by frame index for quick access
    results_map = {res['frame_index']: res['persons'] for res in final_results}
    
    for frame_idx in tqdm(range(process_limit)):
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx not in results_map:
            continue
            
        vis_frame = frame.copy()
        persons = results_map[frame_idx]
        
        # Create lookup for target positions
        persons_dict = {p['id']: p for p in persons}
        
        for person in persons:
            x, y, w, h = [int(v) for v in person["bbox"]]
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            info_lines = [f"ID: {person['id']}"]
            if person["gaze_target"] != 0:
                info_lines.append(f"Tgt: {person['gaze_target']}")

            text_x = x + 5
            text_y = y + 20
            line_height = 15
            
            for line in info_lines:
                if text_y < height: 
                    cv2.putText(vis_frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                text_y += line_height

            if person["face_bbox"]:
                fx, fy, fw, fh = [int(v) for v in person["face_bbox"]]
                cv2.rectangle(vis_frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 1)
            
            # Draw gaze arrow if target exists and is valid
            if person["gaze_target"] != 0 and person["face_bbox"]:
                target_id = person["gaze_target"]
                if target_id in persons_dict:
                    target_person = persons_dict[target_id]
                    if target_person["face_bbox"]:
                        tfx, tfy, tfw, tfh = target_person["face_bbox"]
                        tx, ty = int(tfx + tfw/2), int(tfy + tfh/2)
                        
                        fx, fy, fw, fh = person["face_bbox"]
                        cx, cy = int(fx + fw/2), int(fy + fh/2)
                        cv2.arrowedLine(vis_frame, (cx, cy), (tx, ty), (0, 0, 255), 2, tipLength=0.05)
            
            # Optional: Draw raw gaze point if available (for debugging/verification)
            # if person['gaze_point']:
            #     gx, gy = person['gaze_point']
            #     cv2.circle(vis_frame, (int(gx), int(gy)), 4, (0, 255, 255), -1)

        out.write(vis_frame)

    cap.release()
    out.release()
    
    print(f"Saving results to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for gaze and pose annotation.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to the output JSON file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0 or cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
    else:
        if args.output is None:
            video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
            args.output = f"{video_basename}_annotation.json"
            
        process_video(args.video_path, args.output, args.device)
