"""
주석 없는 깔끔버전
"""
# ====================================================================================
# 필요 라이브러리
# ====================================================================================

# Open cv 이미지 처리를 담당함 예를들어 필터링, 마스킹, 그리기, 컴퓨터 비전 알고리즘Pnp의 라이브러리
import cv2
# Realsense 카메라 라이브러리 RGB 및 Depth 스트림을 가져오고 카메라의 렌저 정보 즉 내부 파라미터를 제어
import pyrealsense2 as rs
# 행렬 연산용, 모든 좌표 데이터를 배열 형태로 처리하기 위해 사용
import numpy as np
# SAHI(Slicing Aided Hyper Inference). 이미지를 쪼개서 검출하여 아주 작은 번호판 글자도 놓치지 않게 해줍니다.
from sahi.predict import get_sliced_prediction
# 삼각함수 연산용, 회전행렬을 우리가 아는 각도 단위로 변환할때 사용
import math


# ====================================================================================
# 실제 번호판 크기 및 3D 모델 설정
# ====================================================================================

# 실제 번호판의 가로 m단위
PLATE_WIDTH = 0.11
# 실제 번호판의 세로 m단위
PLATE_HEIGHT = 0.02

# 카메라의 이미지상에 그릴 초록상자의 두께 m단위, 나중에 위치각도 계산에 들어가지 않음
PLATE_DEPTH = 0.02

# 화면에 그릴 3D 그림 좌표축 길이 설정 m단위
AXIS = 0.05           


# 글자들을 하나의 번호판으로 묶을 때 사용할 거리 가중치
# 글자의 가로 길이에 7을 곱한 범위 안에 다른글자가 있으면 같은 번호판으로 인식함
GROUP_WIDTH_MULT = 7.0  
# 세로는 1.2배로 좁게 잡아 줄바꿈이 된 다른 물체와 섞이지 않게함
GROUP_HEIGHT_MULT = 1.2  


# ====================================================================================
# 지수 이동 평균 (Exponential Moving Average)
# ====================================================================================
# 박스가 파르르 떨리는 지터(Jitter) 현상을 방지하여 훨씬 부드럽게 보이게 합니다

# 이전 프레임의 위치/회전 값을 저장 (필터용)
prev_tvec, prev_rvec = None, None
# 현재 값을 80%, 이전 값을 20% 섞어서 부드럽게 만듦 (지터 방지)  , 반응 속도 (0.1~1.0, 높을수록 빠름)
alpha = 0.3
# 좌표가 0.8m 이상 갑자기 튀면 무시함 (오류 방지)
JUMP_THRESHOLD = 0.8

# 인식률 0.3이면 30%만 글자처럼 인식해도 글자로 인식 올릴수록 엄격해짐
recognition = 0.5

# ====================================================================================
# 3D 모델 좌표계의 정의
# ====================================================================================
# PnP 알고리즘이 기준점으로 삼을 실제 세계의 3D 좌표 (중심이 0,0,0인 평면)
# 번호판의 정중앙을 0,0,0
# X 좌표: 왼쪽은 -절반, 오른쪽은 +절반으로 배치합니다.
# Y 좌표: 위쪽은 -절반, 아래쪽은 +절반으로 배치합니다.
# (컴퓨터 그래픽스 좌표계는 아래로 갈수록 Y가 커지기 때문입니다.)
# Z 좌표: 번호판 표면이므로 모두 0입니다.
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
        # ====================================================================================
        # 카메라 영상 받아오기 및 전처리
        # ====================================================================================
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        depth_frame = aligned_frames.get_depth_frame()
        img_bgr = np.asanyarray(color_frame.get_data())
        
        # [수정] HSV 필터링 과정을 제거하고 원본 img_bgr을 그대로 AI 모델에 입력합니다.
        results = get_sliced_prediction(img_bgr, detection_model, slice_height=960, slice_width=960, verbose=0)

        # ====================================================================================
        # 찾은 글자의 정밀 좌표 추출 (숫자만 인식 - 한글 글자 완전 제외)
        # ====================================================================================
        detections = []
        for obj in results.object_prediction_list:
            
            try:
                if not obj.category.name.isdigit():   # '0'~'9'만 통과
                    continue
            except (AttributeError, TypeError):
                continue

            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)

            # ====================================================================================
            # 글자 주변 도려내기 (ROI 설정)
            # ====================================================================================
            pad = 5
            rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(1280, x2+pad), min(720, y2+pad)
            roi = img_bgr[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # ====================================================================================
                # 기울어진 사각형(OBB) 계산
                # ====================================================================================
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
        # PnP계산 (다중 번호판 대응 수정 버전)
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
                    # ====================================================================================
                    # EMA 지터 방지 필터
                    # ====================================================================================
                    if prev_tvec is None or prev_rvec is None:
                        smoothed_tvec = tvec.copy()
                        smoothed_rvec = rvec.copy()
                    else:
                        if np.linalg.norm(tvec - prev_tvec) > JUMP_THRESHOLD:
                            smoothed_tvec = tvec.copy()
                        else:
                            smoothed_tvec = alpha * tvec + (1 - alpha) * prev_tvec
                        
                        smoothed_rvec = alpha * rvec + (1 - alpha) * prev_rvec

                    prev_tvec = smoothed_tvec.copy()
                    prev_rvec = smoothed_rvec.copy()

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