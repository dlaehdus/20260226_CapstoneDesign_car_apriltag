"""
1번에서 1,4,6,9만 인식 안하게함
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
PLATE_WIDTH = 0.22
# 실제 번호판의 세로 m단위
PLATE_HEIGHT = 0.04

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
                # [수정된 부분] 1, 4, 6, 9 숫자 제외 필터링 추가
                name = obj.category.name
                if not name.isdigit():   # '0'~'9'만 통과
                    # AI(SAHI) 모델이 검출한 객체 중 이름(category.name)이 숫자인 것만 골라냅니다. 즉 번호판의 한글 허, 다, 가 제외
                    continue
                
                if name in ['1','4', '6', '9']: # 1, 4, 6, 9는 제외
                    continue
            except (AttributeError, TypeError):
                continue
            
            # AI가 "여기 숫자가 있어요!"라고 보고한 위치 데이터를 **[왼쪽 위 X, 왼쪽 위 Y, 오른쪽 아래 X, 오른쪽 아래 Y]**라는 4개의 정수로 변환하는 작업
            try: bbox = obj.bbox.xyxy
            except AttributeError: bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)

            # ====================================================================================
            # 글자 주변 도려내기 (ROI 설정)
            # ====================================================================================
            pad = 5
            # AI가 잡아준 상자는 가끔 글자의 끝부분을 자르는 경우가 있습니다. 그래서 상하좌우로 5픽셀(pad)만큼 여유를 더 줍니다.
            rx1, ry1, rx2, ry2 = max(0, x1-pad), max(0, y1-pad), min(1280, x2+pad), min(720, y2+pad)
            # ROI(Region of Interest, 관심 영역) 설정
            roi = img_bgr[ry1:ry2, rx1:rx2]
            
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # 컬러 이미지를 흑백으로 바꾼 뒤, Otsu 알고리즘을 사용해 배경은 검은색(0), 글자는 흰색(255)으로 명확히 나눕니다.
                # Otsu(오츠) 알고리즘은 이미지 처리에서 **'최적의 임계값(Threshold)'**을 자동으로 찾아내어 이미지를 흑백(이진화)으로 분리하는 가장 대표적인 통계적 방법입니다.
                # 컴퓨터가 사진을 볼 때, "어디까지가 배경이고 어디부터가 글자인지"를 수학적으로 결정하는 기준을 제시합니다.
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                # gray: 입력된 글자 주변 이미지.
                # OTSU: 이미지 전체의 밝기 분포를 분석해 최적의 $t$를 계산합니다.
                # BINARY_INV: 숫자가 보통 어두운 색이므로, 숫자를 흰색(255)으로 배경을 검은색(0)으로 반전시켜 나중에 외곽선(findContours)을 따기 좋게 만듭니다.
                # 2. cv2.THRESH_OTSU (오츠 알고리즘)
                # 이게 이 코드의 핵심입니다. 보통 "밝기 120보다 낮으면 검정, 높으면 흰색으로 해"라고 사람이 정해주는데, 조명이 바뀌면 이 기준이 다 깨집니다.
                # 수학적 원리: 이미지의 밝기 히스토그램을 분석합니다. 배경 집단과 물체 집단 사이의 분산(Variance)을 최대화하는 지점을 수학적으로 계산해 '최적의 기준점'을 찾아냅니다.
                # 효과: 조명이 밝든 어둡든, AI가 잘라준 영역 안에서 가장 적절한 흑백 분리 기준을 자동으로 잡습니다. 
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 상태: 배경은 0(검정), 물체는 255(흰색)로 분리된 상태입니다.
                # : 이 함수는 오직 흑백 이미지에서만 작동하며, 흰색(값이 있는 곳)을 물체로 인식하고 그 경계선을 추적합니다.
                # 의미: **"가장 바깥쪽 테두리만 가져와라"**는 뜻입니다.
                # 수학적/논리적 이유: 숫자 '8'이나 '0'을 생각해보세요. 숫자 안에는 구멍(홀)이 있습니다.
                # 만약 다른 옵션(RETR_TREE)을 쓰면 숫자 바깥 테두리와 안쪽 구멍 테두리를 모두 찾습니다.
                # 하지만 우리는 숫자의 전체적인 위치와 기울기만 알면 되므로, 가장 바깥쪽 외곽선 하나만 필요합니다. 그래서 EXTERNAL 모드를 사용해 연산 효율을 높이고 데이터 구조를 단순화합니다.
                # 3. cv2.CHAIN_APPROX_SIMPLE (근사화 방법)
                # 의미: **"꼭 필요한 점들만 남겨라"**는 뜻입니다.
                # 수학적 원리
                # 만약 숫자의 한쪽 면이 완벽한 직선이라면, 그 직선 위의 모든 픽셀 좌표(예: 100개)를 다 저장할 필요가 있을까요? 아니요, 양 끝점 2개만 있으면 직선을 정의할 수 있습니다
                # CHAIN_APPROX_NONE: 모든 경계점 좌표를 다 저장합니다 (메모리 많이 사용).
                # CHAIN_APPROX_SIMPLE: 수평, 수직, 대각선 선분을 압축하여 끝점만 남깁니다 (메모리 절약, 속도 향상).
                # contours: 찾아낸 외곽선들의 리스트입니다. 각 외곽선은 (x, y) 좌표들의 배열로 이루어져 있습니다.
                # _ (Hierarchy): 외곽선들 간의 계층 구조(어떤 선이 어떤 선 안에 있는지 등) 정보입니다. 여기서는 RETR_EXTERNAL을 써서 계층이 무의미하므로 _ 변수에 넣어 버립니다(무시합니다).

                # 여기까지가 각 숫자에 테두리를 딴 영역
            
                # ====================================================================================
                # 각 숫자의 기울어진 사각형(OBB) 계산
                # ====================================================================================
                if contours:
                    main_cnt = max(contours, key=cv2.contourArea)
                    # 설명: 앞 단계에서 findContours로 찾은 수많은 하얀 덩어리들 중 면적(contourArea)이 가장 큰 것 하나만 골라냅니다.
                    # 이유: 작은 점이나 먼지 같은 노이즈를 무시하고, 진짜 "숫자" 형태만 처리하기 위해서입니다. 면적이 20픽셀보다 작으면 무시하는 안전장치도 걸려 있네요.
                    if cv2.contourArea(main_cnt) > 20:
                        rect = cv2.minAreaRect(main_cnt)
                        # 수학적 의미: 이 함수는 외곽선 점들을 모두 감싸는 사각형 중 면적이 최소가 되는 사각형을 찾습니다.
                        # 특징: 일반적인 사각형(가로, 세로)과 달리 **회전 각도($\theta$)**를 결과로 줍니다. 덕분에 숫자가 왼쪽으로 15도 누워있든, 오른쪽으로 5도 누워있든 그 기울기를 정확히 잡아냅니다.
                        box_pts = cv2.boxPoints(rect)
                        box_pts[:, 0] += rx1; box_pts[:, 1] += ry1
                        # boxPoints: 위에서 구한 rect(중심, 크기, 각도) 데이터를 바탕으로 실제 4개 꼭짓점의 $(x, y)$ 좌표를 계산해냅니다.
                        # 좌표 보정: 아까 숫자를 따낼 때 특정 영역(ROI)만 잘라서 썼죠? 그래서 잘라낸 만큼의 시작 좌표(rx1, ry1)를 다시 더해줘야 전체 화면 기준의 진짜 좌표가 됩니다.
                        pts = sorted(box_pts, key=lambda x: x[1])
                        top = sorted(pts[:2], key=lambda x: x[0])
                        bottom = sorted(pts[2:], key=lambda x: x[0])
                        # 컴퓨터가 3D 거리를 계산하는 solvePnP 알고리즘에 이 점들을 넣으려면, 점들의 순서가 항상 일정해야 합니다 (예: 1번은 무조건 왼쪽 위, 2번은 오른쪽 위...).
                        # 수학적 정렬: 4개의 무작위 점을 좌상 -> 우상 -> 좌하 -> 우하 순서로 강제 재배치합니다. 이 순서가 틀리면 나중에 번호판이 뒤집히거나 꼬인 것으로 인식됩니다.
                        obb_corners = np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)
                        # 이제 obb_corners 데이터는 항상 아래와 같은 순서를 보장받습니다:
                        # obb_corners[0]: 좌상단 (Top-Left)
                        # obb_corners[1]: 우상단 (Top-Right)
                        # obb_corners[2]: 좌하단 (Bottom-Left)
                        # obb_corners[3]: 우하단 (Bottom-Right)


                        detections.append({
                            'center': ((x1+x2)/2, (y1+y2)/2),
                            'obb_corners': obb_corners,
                            'w': x2-x1, 'h': y2-y1
                        })
                        # 이렇게 정밀하게 계산된 **기울어진 4개 점(obb_corners)**을 리스트에 담습니다. 이제 이 점들은 나중에 여러 숫자를 하나로 묶어 번호판 전체의 3D 위치를 계산하는 데 사용됩니다.

        # ====================================================================================
        # 그룹화와 번호판 사각형 만들고 PnP계산 (다중 번호판 대응 수정 버전)
        # ====================================================================================
        visited = [False] * len(detections)
        # 탐지된 모든 숫자 개수만큼 False로 채워진 리스트를 생성함 이미 처리된 숫자인지 체크하기 위한 리스트
        for i in range(len(detections)):
            # 숫자 하나하나를 검출
            if visited[i]: continue
                # 만약 이 숫자가 이미 이전 루프에서 다른 그룹의 맴버로 합류했다면 건너뛰고 다음 숫자를 봄
            current_group = [detections[i]]
                # 새로운 번호판 그룹을 하나 만듬 현재 해당 숫자가 이 그룹의 1번 member가 됨
            visited[i] = True
                # 해당 숫자는 True로 변환후 다시는 그룹을 생성하거나 들어가지 못함
            ref_w = detections[i]['w']
            ref_h = detections[i]['h']
                # 기준이 되는 숫자의 가로너비와 세로 높이를 저장
            
            changed = True
            while changed:
                # 새로운 맴버가 추가되었는지 감지하는 루프 ex 처음에 1이라는 숫자를 찾았고 그 옆에 2를 찾아서 그룹에 넣었다면 이제 2옆의 3도 찾아야함
                changed = False
                for j in range(len(detections)):
                    # 아직 어떤 그룹에도 속하지 않은 모든 숫자를 검사 대상으로 올림
                    if not visited[j]:
                        for member in current_group:
                            # 현재 우리 그룹 전체를 한명씩 돌아가며 새로운 후보와 비교
                            dx = abs(member['center'][0] - detections[j]['center'][0])
                            dy = abs(member['center'][1] - detections[j]['center'][1])
                            if dx < (ref_w * 3.5) and dy < (ref_h * 0.8):
                                current_group.append(detections[j])
                                visited[j] = True
                                changed = True
                                break
                            # dx: 우리 팀원 중 누구라도 후보 숫자와의 가로 거리가 글자 너비의 3.5배 이내이고,
                            # dy: 세로(높이) 차이가 글자 높이의 0.8배 이내라면?
                            # 너도 우리 그룹이다라고 인식
            
            if len(current_group) >= 3:
            # 보통 번호판은 숫자가 7,8개로 이루어져 있으므로 해당 그룹의 숫자가 최소 3개 이상일때만 진짜 번호판으로 인정함
                current_group.sort(key=lambda x: x['center'][0])
                # 그룹에 속한 숫자들을 x좌표가 작은 순서대로 배치함

                for item in current_group:
                    cv2.drawContours(img_bgr, [np.int64(item['obb_corners'])], -1, (0, 255, 0), 1)
                        # 같은 번호판으로 묶인 숫자들 각각에 대해 기울어진 사각형 OBB를 생성함 번호판의 각 숫자가 초록색 테두리로 감싸지며 컴퓨터가 이 숫자들을 한 팀으로 인식했는지 확인

                img_pts_2d = np.array([
                    current_group[0]['obb_corners'][0],
                    current_group[-1]['obb_corners'][1],
                    current_group[0]['obb_corners'][2],
                    current_group[-1]['obb_corners'][3]
                ], dtype=np.float32)
                    # PnP알고리즘에 넣기 위해 여러 숫자중 가장 바깥쪽 끝점 4개만 골라냄
                    # current_group[0]['obb_corners'][0],  # 1. 맨 왼쪽 숫자의 '좌상단'
                    # current_group[-1]['obb_corners'][1], # 2. 맨 오른쪽 숫자의 '우상단'
                    # current_group[0]['obb_corners'][2],  # 3. 맨 왼쪽 숫자의 '좌하단'
                    # current_group[-1]['obb_corners'][3]  # 4. 맨 오른쪽 숫자의 '우하단'


                success, rvec, tvec = cv2.solvePnP(obj_points, img_pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                # 이 함수는 **"2차원 이미지의 점들"**과 **"실제 세계의 3차원 크기"**를 비교하여 카메라와 물체 사이의 상대적 위치를 계산합니다.

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