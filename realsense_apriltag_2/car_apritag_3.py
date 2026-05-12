"""
디버깅용 코드
"""

import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math
from sahi.models.ultralytics import UltralyticsDetectionModel

# ====================================================================================
# 실제 번호판 크기 및 3D 모델 설정
# ====================================================================================
PLATE_WIDTH = 0.022
PLATE_HEIGHT = 0.04
PLATE_DEPTH = 0.02
AXIS = 0.05           

prev_tvec, prev_rvec = None, None
alpha = 0.3
JUMP_THRESHOLD = 0.8
recognition = 0.4

obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0],   # 좌상단
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0],   # 우상단
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0],   # 좌하단
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]    # 우하단
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

# ====================================================================================
# 모델 및 리얼센스 초기화
# ====================================================================================
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

# 디버깅용 윈도우 생성
win_main = "1. Final Detection (OBB & 3D)"
win_roi = "2. AI Detection & ROI Pad"
win_binary = "3. Binary Thresholding (findContours)"

cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
cv2.namedWindow(win_roi, cv2.WINDOW_NORMAL)
cv2.namedWindow(win_binary, cv2.WINDOW_NORMAL)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        img_bgr = np.asanyarray(color_frame.get_data())
        depth_frame = aligned_frames.get_depth_frame()

        # 화면 1 & 2용 복사본
        img_roi_debug = img_bgr.copy()
        
        results = get_sliced_prediction(img_bgr, detection_model, slice_height=960, slice_width=960, verbose=0)

        detections = []
        binary_list = [] # 화면 3용 리스트

        for obj in results.object_prediction_list:
            try:
                if not obj.category.name.isdigit(): continue
            except: continue

            try: bbox = obj.bbox.xyxy
            except: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)

            # 화면 2: AI 기본 박스 (노란색)
            cv2.rectangle(img_roi_debug, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # Pad 적용 ROI
            pad = -3
            rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(1280, x2+pad), min(720, y2+pad)
            
            # 화면 2: Pad 적용 박스 (파란색)
            cv2.rectangle(img_roi_debug, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
            
            roi = img_bgr[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 화면 3: 이진화 이미지 수집
                binary_list.append(cv2.resize(binary, (60, 80)))

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    main_cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(main_cnt) > 20:
                        rect = cv2.minAreaRect(main_cnt)
                        box_pts = cv2.boxPoints(rect)
                        box_pts[:, 0] += rx1; box_pts[:, 1] += ry1
                        
                        pts = sorted(box_pts, key=lambda x: x[1])
                        top = sorted(pts[:2], key=lambda x: x[0])
                        bottom = sorted(pts[2:], key=lambda x: x[0])
                        obb_corners = np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)
                        
                        detections.append({
                            'center': ((x1+x2)/2, (y1+y2)/2),
                            'obb_corners': obb_corners,
                            'w': x2-x1, 'h': y2-y1
                        })

        # 화면 3 출력 (가로로 병합)
        if binary_list:
            cv2.imshow(win_binary, np.hstack(binary_list))

        # PnP 로직 및 화면 1 시각화
        visited = [False] * len(detections)
        for i in range(len(detections)):
            if visited[i]: continue
            current_group = [detections[i]]; visited[i] = True
            ref_w, ref_h = detections[i]['w'], detections[i]['h']
            
            changed = True
            while changed:
                changed = False
                for j in range(len(detections)):
                    if not visited[j]:
                        for member in current_group:
                            dx = abs(member['center'][0] - detections[j]['center'][0])
                            dy = abs(member['center'][1] - detections[j]['center'][1])
                            if dx < (ref_w * 3.5) and dy < (ref_h * 0.8):
                                current_group.append(detections[j]); visited[j] = True; changed = True; break
            
            if len(current_group) >= 3:
                current_group.sort(key=lambda x: x['center'][0])

                # 화면 1: 숫자별 OBB 그리기
                for item in current_group:
                    cv2.drawContours(img_bgr, [np.int64(item['obb_corners'])], -1, (0, 255, 0), 1)

                img_pts_2d = np.array([
                    current_group[0]['obb_corners'][0],
                    current_group[-1]['obb_corners'][1],
                    current_group[0]['obb_corners'][2],
                    current_group[-1]['obb_corners'][3]
                ], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_points, img_pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    if prev_tvec is None or prev_rvec is None:
                        smoothed_tvec, smoothed_rvec = tvec.copy(), rvec.copy()
                    else:
                        smoothed_tvec = tvec if np.linalg.norm(tvec - prev_tvec) > JUMP_THRESHOLD else alpha * tvec + (1 - alpha) * prev_tvec
                        smoothed_rvec = alpha * rvec + (1 - alpha) * prev_rvec
                    prev_tvec, prev_rvec = smoothed_tvec.copy(), smoothed_rvec.copy()

                    cv2.drawFrameAxes(img_bgr, K, dist_coeffs, smoothed_rvec, smoothed_tvec, AXIS)
                    box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                    projected_pts, _ = cv2.projectPoints(box_3d, smoothed_rvec, smoothed_tvec, K, dist_coeffs)
                    projected_pts = np.int32(projected_pts).reshape(-1, 2)
                    
                    cv2.drawContours(img_bgr, [projected_pts[:4]], -1, (0, 255, 0), 2)
                    for k in range(4): cv2.line(img_bgr, tuple(projected_pts[k]), tuple(projected_pts[k+4]), (0, 255, 0), 2)
                    cv2.drawContours(img_bgr, [projected_pts[4:]], -1, (0, 255, 0), 2)

        # 모든 화면 띄우기
        cv2.imshow(win_roi, img_roi_debug)  # 2번 화면
        cv2.imshow(win_main, img_bgr)      # 1번 화면
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()