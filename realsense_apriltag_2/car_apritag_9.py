"""
7번에서 SAHI를 적용해서 멀리서도 작은 번호판을 놓지지않게 함
"""


# ====================================================================================
# 필요 라이브러리
# ====================================================================================
import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO  # 기본 YOLO 유지
import math
# SAHI 관련 라이브러리 추가
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

# ====================================================================================
# 실제 번호판 크기 및 3D 모델 설정
# ====================================================================================
PLATE_WIDTH = 0.22
PLATE_HEIGHT = 0.04
PLATE_DEPTH = 0.02
AXIS = 0.05           

GROUP_WIDTH_MULT = 7.0  
GROUP_HEIGHT_MULT = 1.2  

# ====================================================================================
# 지수 이동 평균 (Exponential Moving Average) 및 설정
# ====================================================================================
prev_tvec, prev_rvec = None, None
alpha = 0.3
JUMP_THRESHOLD = 0.8

# [수정된 부분] YOLO 모델 및 클래스 설정 (기존 변수 유지)
model_path = '/home/limdoyeon/realsense_apriltag_1/runs/detect/color_yolo26x_sca03_deg40_mAP50945/weights/best.pt'
TARGET_CLASSES = [0, 1, 12, 23, 34, 41, 42, 43, 44, 45] # 0~9 숫자 인덱스
CONF_THRESHOLD = 0.435

# SAHI 엔진용 모델 래퍼 설정
detection_model = UltralyticsDetectionModel(
    model_path=model_path, 
    confidence_threshold=CONF_THRESHOLD, 
    device="cuda:0"
)

# ====================================================================================
# 3D 모델 좌표계 및 함수 정의 (기존 유지)
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
# 리얼센스 초기화 (기존 유지)
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
# 메인루프
# ====================================================================================
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        depth_frame = aligned_frames.get_depth_frame()
        img_bgr = np.asanyarray(color_frame.get_data())
        
        # [수정된 부분] SAHI 슬라이싱 추론 적용 (화면 분할 인식)
        results = get_sliced_prediction(
            img_bgr, 
            detection_model, 
            slice_height=720,
            slice_width=500,
            overlap_height_ratio=0.1, 
            overlap_width_ratio=0.2,
            verbose=0
        )

        detections = []
        # SAHI의 결과를 순회하며 기존 로직에 맞게 데이터 추출
        for obj in results.object_prediction_list:
            cls_id = int(obj.category.id)
            
            # 지정된 숫자 클래스 인덱스만 통과
            if cls_id not in TARGET_CLASSES:
                continue
            
            # SAHI 바운딩 박스 좌표 추출
            bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)

            # ====================================================================================
            # 글자 주변 도려내기 및 OBB 계산 (기존 로직 100% 유지)
            # ====================================================================================
            pad = -10
            rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(1280, x2+pad), min(720, y2+pad)
            roi = img_bgr[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

        # ====================================================================================
        # 그룹화와 번호판 사각형 만들고 PnP계산 (기존 로직 100% 유지)
        # ====================================================================================
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
                            dx = abs(member['center'][0] - detections[j]['center'][0])
                            dy = abs(member['center'][1] - detections[j]['center'][1])
                            if dx < (ref_w * 3.5) and dy < (ref_h * 0.8):
                                current_group.append(detections[j])
                                visited[j] = True
                                changed = True
                                break
            
            if len(current_group) >= 3:
                current_group.sort(key=lambda x: x['center'][0])

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
                    # EMA 지터 방지 필터
                    if prev_tvec is None or prev_rvec is None:
                        smoothed_tvec, smoothed_rvec = tvec.copy(), rvec.copy()
                    else:
                        if np.linalg.norm(tvec - prev_tvec) > JUMP_THRESHOLD:
                            smoothed_tvec = tvec.copy()
                        else:
                            smoothed_tvec = alpha * tvec + (1 - alpha) * prev_tvec
                        smoothed_rvec = alpha * rvec + (1 - alpha) * prev_rvec

                    prev_tvec, prev_rvec = smoothed_tvec.copy(), smoothed_rvec.copy()

                    # [기존 유지] 데이터 터미널 출력
                    tx, ty, tz = smoothed_tvec.flatten()
                    roll, pitch, yaw = rotation_vector_to_euler(smoothed_rvec)
                    print(f"XYZ: {tx:.3f}, {ty:.3f}, {tz:.3f} | RPY: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
                    
                    center_x = int(np.mean(img_pts_2d[:, 0]))
                    center_y = int(np.mean(img_pts_2d[:, 1]))
                    if depth_frame:
                        depth = depth_frame.get_distance(center_x, center_y)
                        if depth > 0:
                            point = rs.rs2_deproject_pixel_to_point(intr, [center_x, center_y], depth)
                            print(f"Real XYZ: {point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}")
                    
                    # 시각화 (기존 유지)
                    cv2.drawFrameAxes(img_bgr, K, dist_coeffs, smoothed_rvec, smoothed_tvec, AXIS)
                    box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                    projected_pts, _ = cv2.projectPoints(box_3d, smoothed_rvec, smoothed_tvec, K, dist_coeffs)
                    projected_pts = np.int32(projected_pts).reshape(-1, 2)
                    
                    cv2.drawContours(img_bgr, [projected_pts[:4]], -1, (0, 255, 0), 2)
                    for k in range(4):
                        cv2.line(img_bgr, tuple(projected_pts[k]), tuple(projected_pts[k+4]), (0, 255, 0), 2)
                    cv2.drawContours(img_bgr, [projected_pts[4:]], -1, (0, 255, 0), 2)

        cv2.imshow("YOLO PnP Detection", img_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()