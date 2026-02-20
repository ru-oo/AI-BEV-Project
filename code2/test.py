import os
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

# 1. 데이터셋 경로 설정
dataroot = './data/sets/nuscenes'

# 2. nuScenes 데이터 로딩
print("데이터를 로딩 중입니다... (잠시만 기다려주세요)")
try:
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
except Exception as e:
    print("\n[오류 발생] 데이터 경로를 찾지 못했습니다!")
    exit()

# 3. 첫 번째 장면(Scene) 가져오기
my_scene = nusc.scene[0]
print(f"\n현재 장면 설명: {my_scene['description']}")

# 4. 첫 번째 샘플(Sample) 가져오기
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)

# 5. 시각화 (이미지 + 라이다 포인트)
print("이미지를 생성하고 있습니다...")

# [수정된 부분]
# 첫 번째 인자: my_sample['token'] (샘플 ID)
# camera_channel: 'CAM_FRONT' (보고 싶은 카메라 방향)
nusc.render_pointcloud_in_image(
    my_sample['token'],           
    pointsensor_channel='LIDAR_TOP',
    camera_channel='CAM_FRONT',
    dot_size=3
)

# 6. 화면에 띄우기
plt.show()