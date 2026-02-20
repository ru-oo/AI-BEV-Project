import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

def get_lidar_data(nusc, sample_token):
    """
    nuScenes에서 라이다 데이터를 가져와서 [Ego Vehicle] 좌표계로 변환합니다.
    """
    # 1. 데이터 토큰 찾기
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    sd_record = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    
    # 2. 라이다 포인트 클라우드 파일 읽기
    lidar_path = nusc.get_sample_data_path(lidar_token)
    pc = LidarPointCloud.from_file(lidar_path)
    
    # 3. 좌표 변환 (Lidar 센서 위치 -> 차체(Ego) 중심)
    # 라이다는 지붕 위에 있으므로 위치(Translation)와 회전(Rotation)을 보정해야 함
    lidar_to_ego = transform_matrix(
        np.array(cs_record['translation']), 
        Quaternion(cs_record['rotation']), 
        inverse=False
    )
    
    # 4. 변환 적용
    # pc.points는 (4, N) 형태: [x, y, z, intensity]
    # 앞의 3개(x,y,z)만 변환 행렬을 곱해줍니다.
    pc.transform(lidar_to_ego)
    
    return pc.points

def points_to_bev(points, xbound, ybound, zbound):
    """
    3D 포인트들을 2D 격자 지도(Occupancy Grid)로 변환합니다.
    """
    # 1. 범위 설정 (우리가 splat.py에서 정한 것과 똑같이!)
    # x, y: -50m ~ 50m
    # z: -10m ~ 10m (너무 높거나 낮은 노이즈 제거)
    mask = (points[0] >= xbound[0]) & (points[0] < xbound[1]) & \
           (points[1] >= ybound[0]) & (points[1] < ybound[1]) & \
           (points[2] >= zbound[0]) & (points[2] < zbound[1])
    
    points = points[:, mask] # 범위 밖의 점 제거

    # 2. 격자 좌표로 변환 (Quantization)
    # 좌표값(meter) -> 인덱스(index)
    # 예: -50m는 0번 인덱스, 50m는 200번 인덱스
    dx = xbound[2]
    dy = ybound[2]
    
    nx = int((xbound[1] - xbound[0]) / dx)
    ny = int((ybound[1] - ybound[0]) / dy)
    
    # X, Y 좌표만 가져오기 (위에서 본 모습이므로 Z는 무시)
    # -50을 0으로 만들기 위해 xbound[0]을 뺌
    x_idx = ((points[0] - xbound[0]) / dx).astype(np.int32)
    y_idx = ((points[1] - ybound[0]) / dy).astype(np.int32)
    
    # 3. 빈 지도 만들기 (0으로 채움)
    bev_map = np.zeros((nx, ny))
    
    # 4. 점이 있는 곳에 1 표시 (Occupancy)
    # x_idx, y_idx 위치에 1을 채워 넣음
    bev_map[x_idx, y_idx] = 1
    
    return bev_map

# --------------------------------------------------
# 메인 실행
# --------------------------------------------------
def main():
    dataroot = './data/sets/nuscenes'
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
    
    # 첫 번째 샘플 가져오기
    my_sample = nusc.sample[0]
    
    print("1. 라이다 데이터를 가져오는 중...")
    points = get_lidar_data(nusc, my_sample['token'])
    print(f"   -> 총 {points.shape[1]}개의 점을 찾았습니다.")
    
    print("2. BEV 지도(정답지)로 변환 중...")
    # splat.py 설정과 동일하게 맞춤
    xbound = [-50, 50, 0.5]
    ybound = [-50, 50, 0.5]
    zbound = [-10, 10, 20.0]
    
    gt_bev = points_to_bev(points, xbound, ybound, zbound)
    print(f"   -> 지도 크기: {gt_bev.shape}")

    # === 시각화 ===
    print("3. 결과 확인")
    plt.figure(figsize=(10, 10))
    
    # Ego Car (내 차 위치)
    plt.plot(100, 100, 'r^', markersize=15, label='Ego Car')
    
    # 라이다 지도 그리기 (흑백)
    # 0(검정): 빈 공간, 1(흰색): 물체 있음
    # origin='lower'를 써야 좌표계가 뒤집히지 않음
    # .T (Transpose)를 해주는 이유는 행렬(행,열)과 좌표(x,y) 순서가 반대라서
    plt.imshow(gt_bev.T, origin='lower', cmap='gray')
    
    plt.title("Ground Truth: LiDAR Occupancy Grid")
    plt.xlabel("Y (Left-Right)")
    plt.ylabel("X (Front-Back)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()