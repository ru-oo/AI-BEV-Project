import torch
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from model import CamEncoder
from splat import VoxelPooling

# --------------------------------------------------
# 1. 기하학적 정보 생성 함수 (3D 좌표 만들기)
# --------------------------------------------------
def create_frustum_grid(H, W, D):
    """
    이미지 픽셀 위치에 깊이(Depth)를 곱해 3D 좌표(Frustum)를 만듭니다.
    (복잡한 카메라 행렬 연산 대신, 시각화를 위해 단순화된 부채꼴 모양을 만듭니다)
    """
    # 1. 픽셀 좌표 만들기
    xs = torch.linspace(0, W - 1, W)
    ys = torch.linspace(0, H - 1, H)
    depths = torch.linspace(4, 45, D) # 4m ~ 45m

    # 2. Meshgrid
    # D x H x W 형태의 격자 생성
    d_grid, y_grid, x_grid = torch.meshgrid(depths, ys, xs, indexing='ij')

    # 3. 픽셀(u,v) -> 3D(x,y,z) 변환 (간이 공식)
    # 카메라가 (0,0,0)에 있고 앞(x축)을 본다고 가정
    # x = depth
    # y = (u - center) * depth * scale
    x_3d = d_grid
    y_3d = (x_grid - W/2) * d_grid * 0.015 # 0.003은 퍼짐 정도(FOV) 조절
    z_3d = torch.zeros_like(x_3d) # 높이는 무시 (평평하게)

    # (B, D, H, W, 3) 형태로 만듦
    geom_feats = torch.stack([x_3d, y_3d, z_3d], dim=-1)
    return geom_feats.unsqueeze(0) # Batch 차원 추가

# --------------------------------------------------
# 2. 메인 실행 코드
# --------------------------------------------------
def main():
    # === 설정 ===
    dataroot = './data/sets/nuscenes'
    H_in, W_in = 256, 704   # 입력 이미지 크기
    H_out, W_out = 8, 22    # 모델 통과 후 크기 (1/32)
    D = 41                  # 깊이 구간
    C = 64                  # 채널 수
    
    # === 준비 ===
    # 1. 데이터 로드
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
    my_sample = nusc.sample[0]
    cam_token = my_sample['data']['CAM_FRONT']
    
    # 실제 이미지 가져오기 (시각화용)
    cam_data = nusc.get_sample_data(cam_token)
    real_image_path = cam_data[0]
    real_img = plt.imread(real_image_path)

    # 2. 모델 준비 (랜덤 가중치)
    model = CamEncoder(D=D, C=C)
    model.eval()
    
    # 3. Splat 모듈 준비 (-50m ~ 50m)
    pooler = VoxelPooling(xbound=[-50, 50, 0.5], 
                          ybound=[-50, 50, 0.5], 
                          zbound=[-10, 10, 20], 
                          dbound=[4, 45, 1])

    # === 실행 (Pipeline) ===
    print("1. 이미지를 AI 모델에 넣습니다...")
    # 가짜 텐서 입력 (실제 학습 땐 전처리된 이미지를 넣어야 함)
    dummy_input = torch.randn(1, 3, H_in, W_in)
    with torch.no_grad():
        depth_probs, context = model(dummy_input)

    print("2. 2D 이미지를 3D 공간으로 들어올립니다 (Lift)...")
    # 부채꼴 좌표 생성
    geom_feats = create_frustum_grid(H_out, W_out, D)
    
    # Lift (Outer Product)
    # Context(특징) * Depth(확률)
    context = context.unsqueeze(1)
    depth_probs = depth_probs.unsqueeze(2)
    frustum_features = context * depth_probs  # (1, 41, 64, 8, 22)

    print("3. 3D 점들을 바닥 지도에 떨어뜨립니다 (Splat)...")
    bev_map = pooler(geom_feats, frustum_features)
    # 결과: (1, 64, 200, 200)

    # === 시각화 ===
    print("4. 결과 그리기...")
    plt.figure(figsize=(15, 7))

    # [왼쪽] 원본 카메라 이미지
    plt.subplot(1, 2, 1)
    plt.title("Input: Camera Image (2D)")
    plt.imshow(real_img)
    plt.axis('off')

    # [오른쪽] 생성된 BEV 지도
    # 64개 채널을 다 볼 수 없으니 합쳐서 봅니다 (Sum)
    bev_image = bev_map[0].sum(dim=0).cpu().numpy()
    
    plt.subplot(1, 2, 2)
    plt.title("Output: BEV Map (3D -> 2D Top View)")
    # 중앙을 기준으로 부채꼴이 보여야 함
    plt.imshow(bev_map[0].sum(dim=0).T, origin='lower', cmap='inferno') 
    
    # 자율주행차 위치 표시
    plt.plot(100, 100, 'w^', markersize=15, label='Ego Car') # 200x200의 중심
    plt.legend()
    plt.xlabel("Y (Left-Right)")
    plt.ylabel("X (Front-Back)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()