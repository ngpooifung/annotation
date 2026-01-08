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

def process_video(video_path, output_json_path, device='cuda:0', target_id=1):
    print(f"Processing video: {video_path}")
    print(f"Output JSON: {output_json_path}")
    print(f"Device: {device}")
    print(f"Target ID for heatmap: {target_id}")

    # 1. Initialize YOLOv12 Pose Model
    print("Loading YOLOv12 pose model...")
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
    MAX_DURATION_SEC = 60
    
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
    def get_gaze_heatmap(image_rgb, bbox_xywh):
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
        return heatmap

    # --- Single Pass: Process & Visualize ---
    print("Processing frames and generating heatmap visualization...")
    
    process_limit = min(frame_count, max_process_frames)
    
    # Initialize Video Writer
    output_video_path = output_json_path.replace('.json', '_heatmap_vis.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))

    # Buffer for Centered Smoothing [t-2, t-1, t, t+1, t+2]
    # Each element: {'frame': frame_img, 'frame_rgb': rgb_img, 'detections': {id: {'face_bbox': ..., 'body_bbox': ...}}}
    frame_buffer = []
    BUFFER_SIZE = 5
    HALF_WINDOW = 2

    for frame_idx in tqdm(range(process_limit)):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval != 0:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Process Current Frame (Get Raw Data)
        current_detections = {}

        # YOLO Tracking
        yolo_results = pose_model.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            verbose=False,
            device=device,
            conf=0.5,
            iou=0.72
        )
        
        if yolo_results and yolo_results[0].boxes.id is not None:
            res = yolo_results[0]
            track_ids = res.boxes.id.int().cpu().tolist()
            boxes_xywh = res.boxes.xywh.cpu().tolist() # center_x, center_y, w, h
            keypoints = res.keypoints.cpu()
            
            for i, tid in enumerate(track_ids):
                # Process each person
                # Check height filter or other quality checks if necessary? 
                # Keeping simple: process all tracked
                
                # Body BBox conversion: center_xywh -> top_left_xywh
                c_bbox = boxes_xywh[i]
                tl_x = c_bbox[0] - c_bbox[2]/2
                tl_y = c_bbox[1] - c_bbox[3]/2
                body_bbox = [tl_x, tl_y, c_bbox[2], c_bbox[3]]
                
                # Face BBox
                kps_xy = keypoints.xy[i].numpy()
                kps_conf = keypoints.conf[i].numpy()
                face_bbox = get_face_bbox(kps_xy, kps_conf, width, height)
                
                # Validate Face BBox if found
                if face_bbox is not None:
                    fx, fy, fw, fh = face_bbox
                    fx = max(0, min(fx, width - 1))
                    fy = max(0, min(fy, height - 1))
                    fw = max(1, min(fw, width - fx))
                    fh = max(1, min(fh, height - fy))
                    face_bbox = [fx, fy, fw, fh]
                    
                    # Ensure face is somewhat inside or related to body (intersection check)
                    # Although get_face_bbox uses keypoints which are usually correct.
                    # Let's trust get_face_bbox but ensure it's not totally out of frame (handled above)
                
                current_detections[tid] = {
                    'body_bbox': body_bbox,
                    'face_bbox': face_bbox
                }

        current_data = {
            'frame': frame,
            'frame_rgb': frame_rgb,
            'detections': current_detections
        }

        # Push to buffer
        frame_buffer.append(current_data)

        # 2. Process Buffered Frame (if buffer is full)
        if len(frame_buffer) >= BUFFER_SIZE:
            center_idx = HALF_WINDOW
            target_data = frame_buffer[center_idx]
            vis_frame = target_data['frame'].copy()
            
            # Identify all unique IDs in the buffer window
            unique_ids = set()
            for d in frame_buffer:
                unique_ids.update(d['detections'].keys())
            
            # Prepare to draw ID 1 last so its heatmap is on top? 
            # Or draw heatmap first so boxes on top?
            # Heatmap modifies image, so draw heatmap first.
            
            # We process target_id specifically
            smoothed_data = {} # store results to draw boxes after heatmap

            for pid in unique_ids:
                # Collect valid boxes from buffer for this pid
                face_bboxes = []
                body_bboxes = []
                
                for d in frame_buffer:
                    if pid in d['detections']:
                        det = d['detections'][pid]
                        if det['face_bbox'] is not None:
                            face_bboxes.append(det['face_bbox'])
                        if det['body_bbox'] is not None:
                            body_bboxes.append(det['body_bbox'])
                
                s_face_bbox = None
                s_body_bbox = None
                
                if face_bboxes:
                    s_face_bbox = np.mean(face_bboxes, axis=0).astype(int).tolist()
                
                if body_bboxes:
                    s_body_bbox = np.mean(body_bboxes, axis=0).astype(int).tolist()
                
                # Refine smoothed face bbox
                if s_face_bbox:
                     fx, fy, fw, fh = s_face_bbox
                     fx = max(0, min(fx, width - 1))
                     fy = max(0, min(fy, height - 1))
                     fw = max(1, min(fw, width - fx))
                     fh = max(1, min(fh, height - fy))
                     s_face_bbox = [fx, fy, fw, fh]
                
                # Store for visualization
                smoothed_data[pid] = {'face': s_face_bbox, 'body': s_body_bbox}

                # VISUALIZATION RULE 1: Target ID Heatmap
                if pid == target_id and s_face_bbox is not None:
                    raw_heatmap = get_gaze_heatmap(target_data['frame_rgb'], s_face_bbox)
                    norm_heatmap = cv2.normalize(raw_heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    norm_heatmap = np.uint8(norm_heatmap)
                    heatmap_resized = cv2.resize(norm_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                    
                    vis_frame = cv2.addWeighted(vis_frame, 0.6, heatmap_color, 0.4, 0)

            # Draw Boxes and Text
            for pid, data in smoothed_data.items():
                s_face = data['face']
                s_body = data['body']
                
                if pid == target_id:
                    if s_face is not None:
                        # Draw Smoothed Face Box (Blue)
                        x, y, w, h = s_face
                        cv2.rectangle(vis_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                        cv2.putText(vis_frame, f"ID: {pid}", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                else:
                    # Other Persons
                    # Rule: Only if they have a valid face detection (s_face is not None)
                    if s_face is not None and s_body is not None:
                        # Draw Smoothed Body Box (Green)
                        bx, by, bw, bh = s_body
                        cv2.rectangle(vis_frame, (int(bx), int(by)), (int(bx+bw), int(by+bh)), (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"ID: {pid}", (int(bx), int(by)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            out.write(vis_frame)
            frame_buffer.pop(0)

    # 3. Flush remaining frames in buffer
    while len(frame_buffer) > 0:
        target_data = frame_buffer[0]
        vis_frame = target_data['frame'].copy()
        
        unique_ids = set()
        for d in frame_buffer:
            unique_ids.update(d['detections'].keys())
            
        smoothed_data = {}
        
        for pid in unique_ids:
            face_bboxes = []
            body_bboxes = []
            for d in frame_buffer:
                if pid in d['detections']:
                    det = d['detections'][pid]
                    if det['face_bbox'] is not None:
                        face_bboxes.append(det['face_bbox'])
                    if det['body_bbox'] is not None:
                        body_bboxes.append(det['body_bbox'])
            
            s_face_bbox = None if not face_bboxes else np.mean(face_bboxes, axis=0).astype(int).tolist()
            s_body_bbox = None if not body_bboxes else np.mean(body_bboxes, axis=0).astype(int).tolist()
            
            if s_face_bbox:
                 fx, fy, fw, fh = s_face_bbox
                 fx = max(0, min(fx, width - 1))
                 fy = max(0, min(fy, height - 1))
                 fw = max(1, min(fw, width - fx))
                 fh = max(1, min(fh, height - fy))
                 s_face_bbox = [fx, fy, fw, fh]

            smoothed_data[pid] = {'face': s_face_bbox, 'body': s_body_bbox}

            if pid == target_id and s_face_bbox is not None:
                raw_heatmap = get_gaze_heatmap(target_data['frame_rgb'], s_face_bbox)
                norm_heatmap = cv2.normalize(raw_heatmap, None, 0, 255, cv2.NORM_MINMAX)
                norm_heatmap = np.uint8(norm_heatmap)
                heatmap_resized = cv2.resize(norm_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                vis_frame = cv2.addWeighted(vis_frame, 0.6, heatmap_color, 0.4, 0)

        for pid, data in smoothed_data.items():
            s_face = data['face']
            s_body = data['body']
            
            if pid == target_id:
                if s_face is not None:
                    x, y, w, h = s_face
                    cv2.rectangle(vis_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                    cv2.putText(vis_frame, f"ID: {pid}", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                if s_face is not None and s_body is not None:
                    bx, by, bw, bh = s_body
                    cv2.rectangle(vis_frame, (int(bx), int(by)), (int(bx+bw), int(by+bh)), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"ID: {pid}", (int(bx), int(by)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(vis_frame)
        frame_buffer.pop(0)

    cap.release()
    out.release()
    print("Done! Heatmap visualization saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for gaze and pose annotation.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to the output JSON file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0 or cpu)")
    parser.add_argument("--target-id", type=int, default=1, help="ID of the person to generate heatmap for")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
    else:
        if args.output is None:
            video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
            args.output = f"{video_basename}_annotation.json"
            
        process_video(args.video_path, args.output, args.device, args.target_id)
