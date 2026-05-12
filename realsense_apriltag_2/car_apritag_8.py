"""
미친 모델 디버깅하는 코드
"""

import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import math

# ====================================================================================
# 설정 및 상수
# ====================================================================================
PLATE_WIDTH = 0.22
PLATE_HEIGHT = 0.04
PLATE_DEPTH = 0.02
AXIS = 0.05           

# EMA 필터 설정
prev_tvec, prev_rvec = None, None
alpha = 0.3
JUMP_THRESHOLD = 0.8

# [기존 유지] 실시간 조절 변수
current_scale = 0  
x_offset = 0
y_offset = 0
morph_size = 0 

# [비율 설정] 55:83
TARGET_RATIO = 55 / 83

# YOLO 설정
model_path = '/home/limdoyeon/realsense_apriltag_1/runs/detect/color_yolo26x_sca03_deg40_mAP50945/weights/best.pt'
model = YOLO(model_path)
TARGET_CLASSES = [0, 1, 12, 23, 34, 41, 42, 43, 44, 45] 
CONF_THRESHOLD = 0.435

# 윈도우 이름 설정
win_main = "YOLO PnP Detection"
win_roi = "Debug ROI (Yellow: YOLO, Blue: Padded)"
win_binary = "Binary ROI List"

# ====================================================================================
# 함수 정의
# ====================================================================================
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

# ====================================================================================
# 리얼센스 초기화
# ====================================================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5)

# ====================================================================================
# 메인 루프
# ====================================================================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        img_bgr = np.asanyarray(color_frame.get_data())
        img_roi_debug = img_bgr.copy()
        
        results = model.predict(img_bgr, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = []
        binary_list = [] 

        for box in results.boxes:
            if int(box.cls[0]) not in TARGET_CLASSES: continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            curr_h = (y2 - y1)

            cv2.rectangle(img_roi_debug, (x1, y1), (x2, y2), (0, 255, 255), 1)

            final_h = curr_h + (current_scale * 2)
            final_w = final_h * TARGET_RATIO
            
            new_cx = cx + x_offset
            new_cy = cy + y_offset

            rx1 = int(max(0, new_cx - final_w / 2))
            ry1 = int(max(0, new_cy - final_h / 2))
            rx2 = int(min(1280, new_cx + final_w / 2))
            ry2 = int(min(720, new_cy + final_h / 2))
            
            cv2.rectangle(img_roi_debug, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
            roi = img_bgr[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # [추가] 내부 검은 구멍 채우기 로직
                # 모든 계층을 찾아서 내부의 빈 공간까지 색칠할 수 있게 함
                fill_contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                if fill_contours:
                    # 두께 -1은 내부를 꽉 채운다는 의미 (검은 구멍을 흰색으로 덮어씀)
                    cv2.drawContours(binary, fill_contours, -1, (255), -1)

                # [기존 유지] 모폴로지 연산 (n, m 키 조절)
                if morph_size > 0:
                    kernel = np.ones((morph_size, morph_size), np.uint8)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    binary = cv2.dilate(binary, kernel, iterations=1)

                binary_list.append(cv2.resize(binary, (55, 83)))

                # PnP용 외곽선 추출 (기존 유지)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(main_cnt) > 20:
                        rect = cv2.minAreaRect(main_cnt)
                        box_pts = cv2.boxPoints(rect)
                        box_pts[:, 0] += rx1; box_pts[:, 1] += ry1
                        pts = sorted(box_pts, key=lambda x: x[1])
                        top, bottom = sorted(pts[:2], key=lambda x: x[0]), sorted(pts[2:], key=lambda x: x[0])
                        detections.append({'center': (cx, cy), 
                                           'obb_corners': np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32),
                                           'w': x2-x1, 'h': y2-y1})

        if binary_list:
            cv2.imshow(win_binary, np.hstack(binary_list))

        # PnP 및 시각화 로직 (기존 유지)
        visited = [False] * len(detections)
        for i in range(len(detections)):
            if visited[i]: continue
            current_group = [detections[i]]
            visited[i] = True
            ref_w, ref_h = detections[i]['w'], detections[i]['h']
            
            changed = True
            while changed:
                changed = False
                for j in range(len(detections)):
                    if not visited[j]:
                        for member in current_group:
                            if abs(member['center'][0] - detections[j]['center'][0]) < (ref_w * 3.5) and \
                               abs(member['center'][1] - detections[j]['center'][1]) < (ref_h * 0.8):
                                current_group.append(detections[j]); visited[j] = True; changed = True; break
            
            if len(current_group) >= 3:
                current_group.sort(key=lambda x: x['center'][0])
                for item in current_group:
                    cv2.drawContours(img_bgr, [np.int64(item['obb_corners'])], -1, (0, 255, 0), 1)

                img_pts_2d = np.array([current_group[0]['obb_corners'][0], current_group[-1]['obb_corners'][1],
                                      current_group[0]['obb_corners'][2], current_group[-1]['obb_corners'][3]], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_points, img_pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if success:
                    if prev_tvec is None: smoothed_tvec, smoothed_rvec = tvec.copy(), rvec.copy()
                    else:
                        smoothed_tvec = alpha * tvec + (1 - alpha) * prev_tvec if np.linalg.norm(tvec-prev_tvec) < JUMP_THRESHOLD else tvec.copy()
                        smoothed_rvec = alpha * rvec + (1 - alpha) * prev_rvec
                    prev_tvec, prev_rvec = smoothed_tvec.copy(), smoothed_rvec.copy()

                    cv2.drawFrameAxes(img_bgr, K, dist_coeffs, smoothed_rvec, smoothed_tvec, AXIS)
                    box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                    projected_pts, _ = cv2.projectPoints(box_3d, smoothed_rvec, smoothed_tvec, K, dist_coeffs)
                    projected_pts = np.int32(projected_pts).reshape(-1, 2)
                    cv2.drawContours(img_bgr, [projected_pts[:4]], -1, (0, 255, 0), 2)
                    for k in range(4): cv2.line(img_bgr, tuple(projected_pts[k]), tuple(projected_pts[k+4]), (0, 255, 0), 2)
                    cv2.drawContours(img_bgr, [projected_pts[4:]], -1, (0, 255, 0), 2)

        cv2.imshow(win_roi, img_roi_debug) 
        cv2.imshow(win_main, img_bgr)      
        
        # 키 입력 처리 (모든 키 기능 유지)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('o'): current_scale += 1
        elif key == ord('p'): current_scale -= 1
        elif key == ord('y'): y_offset -= 1
        elif key == ord('h'): y_offset += 1
        elif key == ord('g'): x_offset -= 1
        elif key == ord('j'): x_offset += 1
        elif key == ord('n'): 
            morph_size += 1
            print(f"Morph Size: {morph_size}")
        elif key == ord('m'): 
            morph_size = max(0, morph_size - 1)
            print(f"Morph Size: {morph_size}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()