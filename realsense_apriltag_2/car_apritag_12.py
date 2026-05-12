"""
1번에서
한 번호판의 숫자는 모두 같은 크기니까 만약 오검출된 숫자는 혼자 면적이 다를거아니야 그러니까 한 번호판에 인식된 것중에 크기가 다른걸 제거해줘
"""

# ====================================================================================
# 필요 라이브러리
# ====================================================================================
import cv2
import pyrealsense2 as rs
import numpy as np
from sahi.predict import get_sliced_prediction
import math

# ====================================================================================
# 실제 번호판 크기 및 3D 모델 설정
# ====================================================================================
PLATE_WIDTH = 0.22
PLATE_HEIGHT = 0.04
PLATE_DEPTH = 0.02
AXIS = 0.05           

GROUP_WIDTH_MULT = 7.0  
GROUP_HEIGHT_MULT = 1.2  

# [추가] 면적 이상치 필터링 임계값 (중앙값의 0.5배 미만 혹은 2.0배 초과시 제거)
AREA_RATIO_THRESHOLD_LOW = 0.2
AREA_RATIO_THRESHOLD_HIGH = 4.0

# ====================================================================================
# 지수 이동 평균 (Exponential Moving Average)
# ====================================================================================
prev_tvec, prev_rvec = None, None
alpha = 0.3
JUMP_THRESHOLD = 0.8
recognition = 0.5

# ====================================================================================
# 3D 모델 좌표계의 정의
# ====================================================================================
obj_points = np.array([
    [-PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0],   # 좌상단
    [ PLATE_WIDTH/2, -PLATE_HEIGHT/2, 0],   # 우상단
    [-PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0],   # 좌하단
    [ PLATE_WIDTH/2,  PLATE_HEIGHT/2, 0]    # 우하단
], dtype=np.float32)

# ====================================================================================
# 회전 벡터 데이터를 3축 각도 Roll, Pitch, Yaw로 변화
# ====================================================================================
def rotation_vector_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x, y, z = math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])
    else:
        x, y, z = math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0
    return np.degrees([x, y, z])

# ====================================================================================
# 컴퓨터 속에 가상의 3D 번호판 상자 설계도를 그리는 함수
# ====================================================================================
def get_plate_box_points(w, h, d):
    hw, hh = w / 2, h / 2
    return np.float32([
        [-hw, -hh, 0], [ hw, -hh, 0], [ hw,  hh, 0], [-hw,  hh, 0],
        [-hw, -hh, -d], [ hw, -hh, -d], [ hw,  hh, -d], [-hw,  hh, -d]
    ])

# ====================================================================================
# 모델 및 리얼센스 초기화
# ====================================================================================
from sahi.models.ultralytics import UltralyticsDetectionModel
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
        
        results = get_sliced_prediction(img_bgr, detection_model, slice_height=960, slice_width=960, verbose=0)

        detections = []
        for obj in results.object_prediction_list:
            try:
                name = obj.category.name
                if not name.isdigit(): continue
                if name in ['1', '4', '6', '9']: continue
            except (AttributeError, TypeError): continue
            
            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)

            pad = 5
            rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(1280, x2+pad), min(720, y2+pad)
            roi = img_bgr[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                if contours:
                    main_cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(main_cnt)
                    if area > 20:
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
                            'w': x2-x1, 'h': y2-y1,
                            'area': area  # 면적 데이터 저장
                        })

        # ====================================================================================
        # 그룹화 및 면적 필터링 적용
        # ====================================================================================
        visited = [False] * len(detections)
        for i in range(len(detections)):
            if visited[i]: continue
            current_group = [detections[i]]
            visited[i] = True
            ref_w = detections[i]['w']
            ref_h = detections[i]['h']
            
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
            
            # [수정 로직] 번호판 후보 그룹 내에서 면적 이상치 제거
            if len(current_group) >= 3:
                # 1. 그룹 내 숫자들의 면적 리스트 추출
                areas = [det['area'] for det in current_group]
                median_area = np.median(areas) # 중앙값 계산
                
                # 2. 중앙값 대비 너무 크거나 작은 숫자는 필터링
                filtered_group = [
                    det for det in current_group 
                    if median_area * AREA_RATIO_THRESHOLD_LOW <= det['area'] <= median_area * AREA_RATIO_THRESHOLD_HIGH
                ]
                
                # 필터링 후에도 숫자가 3개 이상 남아야 유효한 번호판으로 인정
                if len(filtered_group) >= 3:
                    current_group = filtered_group
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
                        if prev_tvec is None or prev_rvec is None:
                            smoothed_tvec, smoothed_rvec = tvec.copy(), rvec.copy()
                        else:
                            if np.linalg.norm(tvec - prev_tvec) > JUMP_THRESHOLD:
                                smoothed_tvec = tvec.copy()
                            else:
                                smoothed_tvec = alpha * tvec + (1 - alpha) * prev_tvec
                            smoothed_rvec = alpha * rvec + (1 - alpha) * prev_rvec

                        prev_tvec, prev_rvec = smoothed_tvec.copy(), smoothed_rvec.copy()
                        tx, ty, tz = smoothed_tvec.flatten()
                        roll, pitch, yaw = rotation_vector_to_euler(smoothed_rvec)
                        
                        center_x = int(np.mean(img_pts_2d[:, 0]))
                        center_y = int(np.mean(img_pts_2d[:, 1]))
                        if depth_frame:
                            depth = depth_frame.get_distance(center_x, center_y)
                            if depth > 0:
                                point = rs.rs2_deproject_pixel_to_point(intr, [center_x, center_y], depth)
                        
                        cv2.drawFrameAxes(img_bgr, K, dist_coeffs, smoothed_rvec, smoothed_tvec, AXIS)
                        box_3d = get_plate_box_points(PLATE_WIDTH, PLATE_HEIGHT, PLATE_DEPTH)
                        projected_pts, _ = cv2.projectPoints(box_3d, smoothed_rvec, smoothed_tvec, K, dist_coeffs)
                        projected_pts = np.int32(projected_pts).reshape(-1, 2)
                        
                        cv2.drawContours(img_bgr, [projected_pts[:4]], -1, (0, 255, 0), 2)
                        for k in range(4):
                            cv2.line(img_bgr, tuple(projected_pts[k]), tuple(projected_pts[k+4]), (0, 255, 0), 2)
                        cv2.drawContours(img_bgr, [projected_pts[4:]], -1, (0, 255, 0), 2)

        cv2.imshow("OBB-based PnP Detection (Numbers Only)", img_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()