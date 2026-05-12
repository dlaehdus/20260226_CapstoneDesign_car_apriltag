"""
뎊스 기반으로 교차검증

최종적으로 쓸만함
"""


import cv2
import pyrealsense2 as rs
import numpy as np
import math
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

# [기존 설정 유지]
PLATE_WIDTH = 0.22
PLATE_HEIGHT = 0.04
PLATE_DEPTH = 0.02
AXIS = 0.05

GROUP_WIDTH_MULT = 7.0
GROUP_HEIGHT_MULT = 1.2

prev_tvec, prev_rvec = None, None
alpha = 0.3
JUMP_THRESHOLD = 0.8
recognition = 0.5

obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0],
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0],
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0],
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]
], dtype=np.float32)

def rotation_vector_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x, y, z = math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
    else:
        x, y, z = math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0
    return np.degrees([x, y, z])

def get_plate_box_points(w, h, d):
    hw, hh = w / 2, h / 2
    return np.float32([
        [-hw, -hh, 0], [ hw, -hh, 0], [ hw,  hh, 0], [-hw,  hh, 0],
        [-hw, -hh, -d], [ hw, -hh, -d], [ hw,  hh, -d], [-hw,  hh, -d]
    ])

# 모델 및 리얼센스 초기화
model_path = '/home/limdoyeon/realsense_apriltag_2/runs/detect/EV_Plate_Master_v/weights/best.pt'
detection_model = UltralyticsDetectionModel(model_path=model_path, confidence_threshold=recognition, device="cuda:0")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames) # Depth-Color 정렬
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame() # Depth 프레임 가져오기 추가
        if not color_frame or not depth_frame: continue
        
        img_bgr = np.asanyarray(color_frame.get_data())
        results = get_sliced_prediction(img_bgr, detection_model, slice_height=960, slice_width=960, verbose=0)

        detections = []
        for obj in results.object_prediction_list:
            try:
                if not obj.category.name.isdigit(): continue
            except (AttributeError, TypeError): continue

            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            pad = 5
            rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(1280, x2+pad), min(720, y2+pad)
            
            obb_corners = np.array([[rx1, ry1], [rx2, ry1], [rx1, ry2], [rx2, ry2]], dtype=np.float32)
            detections.append({'center': ((rx1+rx2)/2, (ry1+ry2)/2), 'obb_corners': obb_corners, 'w': rx2-rx1, 'h': ry2-ry1})

        valid_groups = []
        visited = [False] * len(detections)
        for i in range(len(detections)):
            if visited[i]: continue
            current_group = [detections[i]]; visited[i] = True; changed = True
            while changed:
                changed = False
                for j in range(len(detections)):
                    if not visited[j]:
                        for member in current_group:
                            dx = abs(member['center'][0] - detections[j]['center'][0])
                            dy = abs(member['center'][1] - detections[j]['center'][1])
                            height_diff = abs(member['h'] - detections[j]['h']) / max(member['h'], 1e-5)
                            if dx < (member['w'] * 3.5) and dy < (member['h'] * 0.8) and height_diff <= 0.2:
                                current_group.append(detections[j]); visited[j] = True; changed = True; break
            if len(current_group) >= 3: valid_groups.append(current_group)
        
        if valid_groups:
            best_group = max(valid_groups, key=lambda g: np.mean([item['h'] for item in g]))
            areas = [item['w'] * item['h'] for item in best_group]
            median_area = np.median(areas)
            current_group = [item for item in best_group if 0.5 * median_area < (item['w'] * item['h']) < 2.0 * median_area]

            if len(current_group) >= 3:
                current_group.sort(key=lambda x: x['center'][0])
                for item in current_group:
                    pts_to_draw = np.array([item['obb_corners'][0], item['obb_corners'][1], item['obb_corners'][3], item['obb_corners'][2]], dtype=np.int32)
                    cv2.polylines(img_bgr, [pts_to_draw], True, (0, 255, 0), 1)

                img_pts_2d = np.array([
                    current_group[0]['obb_corners'][0], current_group[-1]['obb_corners'][1],
                    current_group[0]['obb_corners'][2], current_group[-1]['obb_corners'][3]
                ], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_points, img_pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    # EMA 필터
                    if prev_tvec is None: smoothed_tvec, smoothed_rvec = tvec.copy(), rvec.copy()
                    else:
                        if np.linalg.norm(tvec - prev_tvec) > JUMP_THRESHOLD: smoothed_tvec = tvec.copy()
                        else: smoothed_tvec = alpha * tvec + (1 - alpha) * prev_tvec
                        smoothed_rvec = alpha * rvec + (1 - alpha) * prev_rvec
                    prev_tvec, prev_rvec = smoothed_tvec.copy(), smoothed_rvec.copy()

                    # 좌표 및 각도 출력 (PnP 결과)
                    tx, ty, tz = smoothed_tvec.flatten()
                    roll, pitch, yaw = rotation_vector_to_euler(smoothed_rvec)
                    print(f"XYZ: {tx:.3f}, {ty:.3f}, {tz:.3f} | RPY: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
                    
                    # ------------------------------------------------------------------------------------
                    # [추가] Depth 기반 실좌표 교차 검증 및 출력
                    # ------------------------------------------------------------------------------------
                    center_x = int(np.mean(img_pts_2d[:, 0]))
                    center_y = int(np.mean(img_pts_2d[:, 1]))
                    depth = depth_frame.get_distance(center_x, center_y)
                    if depth > 0:
                        point = rs.rs2_deproject_pixel_to_point(intr, [center_x, center_y], depth)
                        print(f"Real XYZ: {point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}")
                    # ------------------------------------------------------------------------------------

                    # 시각화 유지 (기존과 동일)
                    cv2.drawFrameAxes(img_bgr, K, dist_coeffs, smoothed_rvec, smoothed_tvec, AXIS)
                    box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                    projected_pts, _ = cv2.projectPoints(box_3d, smoothed_rvec, smoothed_tvec, K, dist_coeffs)
                    projected_pts = np.int32(projected_pts).reshape(-1, 2)
                    cv2.drawContours(img_bgr, [projected_pts[:4]], -1, (0, 255, 0), 2)
                    for k in range(4):
                        cv2.line(img_bgr, tuple(projected_pts[k]), tuple(projected_pts[k+4]), (0, 255, 0), 2)
                    cv2.drawContours(img_bgr, [projected_pts[4:]], -1, (0, 255, 0), 2)

        cv2.imshow("OBB-based PnP (Filtered with Depth)", img_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()