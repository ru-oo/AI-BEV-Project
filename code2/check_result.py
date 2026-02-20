# code/check_result.py 전체 수정 코드

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from train import LSSModel 

def check():
    # === 설정 ===
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"체크 장치: {device}")
    
    # 1. 모델 준비 (Occupancy 설정)
    model = LSSModel(xbound=[-50, 50, 0.5], 
                     ybound=[-50, 50, 0.5], 
                     zbound=[-2.0, 6.0, 2.0], # 높이 4칸 (-2~0, 0~2, 2~4, 4~6)
                     dbound=[4, 45, 1]).to(device)

    model_path = "best_lss_multicam.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ {model_path} 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    model.eval()

    # 2. 데이터 가져오기
    dataset = NuScenesDataset(dataroot='./data/sets/nuscenes', is_train=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # 데이터 하나 뽑기
    imgs, intrinsics, sensor2ego, gt_bev = next(iter(loader))

    imgs = imgs.to(device)
    intrinsics = intrinsics.float().to(device)
    rots = sensor2ego[:, :, :3, :3].float().to(device)
    trans = sensor2ego[:, :, :3, 3].float().to(device)

    # 3. 예측
    with torch.no_grad():
        preds = model(imgs, rots, trans, intrinsics)
        preds_prob = torch.sigmoid(preds) # (1, 4, 200, 200)

    # Numpy 변환
    pred_cube = preds_prob[0].cpu().numpy() # (4, 200, 200)
    gt_cube = gt_bev[0].cpu().numpy()       # (4, 200, 200)

    # === 시각화 (층별 분석) ===
    # 4개 층이므로 4행 3열로 그립니다.
    # [입력 이미지]는 맨 위에 하나만 표시
    
    plt.figure(figsize=(15, 12))
    
    # 층별 이름 (높이 구간)
    layer_names = ["-2m ~ 0m (Ground)", "0m ~ 2m (Car Body)", "2m ~ 4m (High)", "4m ~ 6m (Very High)"]

    for i in range(4):
        # 1. 모델 예측 (Prediction)
        plt.subplot(4, 3, i*3 + 1)
        plt.imshow(pred_cube[i].T, origin='lower', cmap='jet', vmin=0, vmax=1)
        plt.title(f"Pred: {layer_names[i]}")
        plt.ylabel("Front-Back")
        if i == 3: plt.xlabel("Left-Right")

        # 2. 정답 (GT)
        plt.subplot(4, 3, i*3 + 2)
        plt.imshow(gt_cube[i].T, origin='lower', cmap='gray', vmin=0, vmax=1)
        plt.title(f"GT: {layer_names[i]}")
        if i == 3: plt.xlabel("Left-Right")

        # 3. 겹쳐보기 (오차 분석)
        # 초록: 정답, 빨강: 환각, 파랑: 놓침
        vis_map = np.zeros((200, 200, 3))
        p = (pred_cube[i] > 0.4).astype(int)
        g = (gt_cube[i] > 0.5).astype(int)
        
        vis_map[(p==1) & (g==1)] = [0, 1, 0] # 일치
        vis_map[(p==1) & (g==0)] = [1, 0, 0] # 예측만 함
        vis_map[(p==0) & (g==1)] = [0, 0, 1] # 놓침
        
        plt.subplot(4, 3, i*3 + 3)
        plt.imshow(vis_map.transpose(1, 0, 2), origin='lower')
        plt.title(f"Diff: {layer_names[i]}")
        
    plt.suptitle("3D Occupancy Layer Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check()