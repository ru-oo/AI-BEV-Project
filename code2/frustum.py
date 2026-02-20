import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# ------------------------------------------------
# 설정 (LSS 알고리즘의 하이퍼파라미터와 유사)
# ------------------------------------------------
H, W = 900, 1600          # 원본 이미지 크기
DOWN_RATIO = 16           # 이미지를 작게 줄여서 점을 찍음 (속도 향상)
D_MIN, D_MAX, D_BIN = 4, 45, 1.0  # 깊이 범위: 4m부터 45m까지 1m 간격

def create_frustum():
    """
    2D 이미지 픽셀 위치(u, v)와 가정된 깊이(d)를 조합하여 
    (u, v, d) 형태의 3D 격자를 만듭니다.
    """
    # 1. 이미지 크기를 줄여서 처리 (너무 많으면 느림)
    dH, dW = H // DOWN_RATIO, W // DOWN_RATIO
    
    # 2. 픽셀 좌표 (u, v) 생성
    xs = np.linspace(0, W - 1, dW, dtype=np.float32)
    ys = np.linspace(0, H - 1, dH, dtype=np.float32)
    
    # 3. 깊이 좌표 (d) 생성 (4m ~ 45m)
    ds = np.arange(D_MIN, D_MAX, D_BIN, dtype=np.float32)
    
    # 4. Meshgrid로 모든 조합 생성 (D x H x W)
    d, v, u = np.meshgrid(ds, ys, xs, indexing='ij')
    
    # 결과: (N, 3) 형태의 점들 [u, v, depth]
    frustum = np.stack([u, v, d], axis=-1)
    return frustum

def get_geometry(nusc, sample_data_token):
    """
    nuScenes에서 카메라의 위치(Extrinsic)와 렌즈 정보(Intrinsic)를 가져옵니다.
    """
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    
    # 1. Intrinsic (카메라 내부 파라미터: 3x3 행렬)
    intrinsics = np.array(cs_record['camera_intrinsic'])
    
    # 2. Rotation (회전)
    rot = Quaternion(cs_record['rotation']).rotation_matrix
    
    # 3. Translation (이동)
    trans = np.array(cs_record['translation'])
    
    return intrinsics, rot, trans

def lift_to_3d(frustum, intrinsics, rot, trans):
    """
    [핵심 알고리즘]
    픽셀(u,v)과 깊이(d)를 실제 3D 좌표(x,y,z)로 변환합니다.
    공식: P_ego = Rot * (K_inv * P_pixel * depth) + Trans
    """
    # (D, H, W, 3) -> (N, 3) 형태로 펼치기
    points = frustum.reshape(-1, 3) 
    u, v, d = points[:, 0], points[:, 1], points[:, 2]

    # 1. 픽셀 좌표를 정규화된 카메라 좌표로 변환 (Unprojection)
    # x_cam = (u - cx) * z / fx
    # y_cam = (v - cy) * z / fy
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d  # 깊이(d)가 곧 카메라 앞쪽 거리(z)
    
    # (N, 3) 형태의 카메라 기준 3D 좌표
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    
    # 2. 카메라 좌표 -> 자율주행차(Ego) 기준 좌표로 변환 (Extrinsic)
    points_ego = (rot @ points_cam.T).T + trans
    
    return points_ego

# ------------------------------------------------
# 메인 실행 코드
# ------------------------------------------------
dataroot = './data/sets/nuscenes'
try:
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
except:
    print("데이터 경로를 확인해주세요!")
    exit()

# 1. 데이터 가져오기 (전방 카메라)
my_sample = nusc.sample[0]
cam_token = my_sample['data']['CAM_FRONT']

# 2. Frustum 생성 (가상의 점들)
frustum_grid = create_frustum()

# 3. 카메라 정보 가져오기
intrinsics, rot, trans = get_geometry(nusc, cam_token)

# 4. 3D 변환 실행 (Lift!)
points_3d = lift_to_3d(frustum_grid, intrinsics, rot, trans)

# 5. 시각화 (Bird's Eye View - 위에서 본 모습)
print("3D 점구름을 그리는 중입니다...")
plt.figure(figsize=(10, 10))

# 3D 점 찍기 (x축, y축) -> 위에서 내려다본 모습
plt.scatter(points_3d[:, 0], points_3d[:, 1], s=0.5, c=points_3d[:, 2], cmap='viridis', alpha=0.5)

# 자율주행차 위치 (0,0) 표시
plt.plot(0, 0, 'rx', markersize=15, label='Ego Vehicle')

plt.title("Lift: Image to 3D Point Cloud (Frustum)")
plt.xlabel("X (meter) - Front/Back")
plt.ylabel("Y (meter) - Left/Right")
plt.axis('equal') # 비율 고정
plt.grid(True)
plt.legend()

print("완료! 팝업창을 확인하세요.")
plt.show()