import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from train import LSSModel

def visualize_overlap():
    # 1. 설정 및 장치 준비
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"평가 장치: {device}")

    # Threshold: 확률이 이 값보다 크면 물체가 있다고 판단 (조절 가능)
    threshold = 0.4 

    # 2. 모델 준비
    # [중요] 학습할 때 사용한 설정과 100% 동일해야 합니다.
    # zbound=[-2.0, 6.0, 2.0] -> 높이 채널 4개 생성
    model = LSSModel(xbound=[-50, 50, 0.5], 
                     ybound=[-50, 50, 0.5], 
                     zbound=[-2.0, 6.0, 2.0], 
                     dbound=[4, 45, 1]).to(device)

    model_path = "best_lss_multicam.pth"
    
    try:
        # map_location을 사용하여 장치에 맞게 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 모델 로드 성공: {model_path}")
    except FileNotFoundError:
        print(f"❌ '{model_path}' 파일이 없습니다. 학습이 완료되었는지 확인하세요.")
        return
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        return

    model.eval()

    # 3. 데이터 가져오기
    # num_workers=0으로 해야 맥에서 안전합니다.
    dataset = NuScenesDataset(dataroot='./data/sets/nuscenes', is_train=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    try:
        imgs, intrinsics, sensor2ego, gt_bev = next(iter(loader))
    except StopIteration:
        print("❌ 데이터셋에서 데이터를 가져올 수 없습니다.")
        return
    
    # 장치로 이동
    imgs = imgs.to(device)
    intrinsics = intrinsics.float().to(device)
    gt_bev = gt_bev.to(device)
    
    rots = sensor2ego[:, :, :3, :3].float().to(device)
    trans = sensor2ego[:, :, :3, 3].float().to(device)
    
    # 4. 예측 수행
    print("🤖 AI가 분석 중입니다...")
    with torch.no_grad():
        preds = model(imgs, rots, trans, intrinsics)
        preds_prob = torch.sigmoid(preds) # 0~1 사이 확률값으로 변환
        
    # === [핵심 수정] 3D -> 2D 압축 (Projection) ===
    # preds_prob 형태: (Batch, Z, X, Y) -> 예: (1, 4, 200, 200)
    # 4개의 높이 층 중 '가장 높은 확률'을 가져와서 2D 지도로 만듭니다.
    # 즉, 바닥이든 공중이든 물체가 있으면 표시합니다.
    
    # (1, 4, 200, 200) -> (200, 200)
    pred_map = torch.max(preds_prob[0], dim=0)[0].cpu().numpy()
    
    # 정답(GT)도 4층짜리 3D 데이터이므로 똑같이 압축합니다.
    if gt_bev.shape[1] > 1:
        gt_map = torch.max(gt_bev[0], dim=0)[0].cpu().numpy()
    else:
        # 혹시 GT가 1층짜리라면 그대로 씁니다.
        gt_map = gt_bev[0, 0].cpu().numpy()
    
    # 5. 이진화 및 IoU 계산
    pred_binary = (pred_map > threshold).astype(int)
    gt_binary = (gt_map > 0.5).astype(int)

    intersection = (pred_binary & gt_binary).sum()
    union = (pred_binary | gt_binary).sum()
    iou_score = intersection / union if union > 0 else 0.0
    print(f"📊 현재 샘플의 BEV 일치도(IoU): {iou_score*100:.2f}%")

    # 6. 시각화 맵 생성 (RGB)
    H, W = pred_binary.shape
    vis_map = np.zeros((H, W, 3))
    
    # 색상 지정
    vis_map[(pred_binary == 1) & (gt_binary == 1)] = [0, 1, 0] # 🟢 정답 (일치): 초록
    vis_map[(pred_binary == 1) & (gt_binary == 0)] = [1, 0, 0] # 🔴 환각 (오답): 빨강
    vis_map[(pred_binary == 0) & (gt_binary == 1)] = [0, 0, 1] # 🔵 놓침 (미탐): 파랑

    # 그림 그리기
    plt.figure(figsize=(14, 7))
    
    # [왼쪽] 전방 카메라
    plt.subplot(1, 2, 1)
    # dataset.cams[1] == 'CAM_FRONT'
    front_cam_idx = 1 
    vis_img = imgs[0, front_cam_idx].permute(1, 2, 0).cpu().numpy()
    
    # 정규화 해제 (밝게 보정)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    vis_img = std * vis_img + mean
    vis_img = np.clip(vis_img, 0, 1)
    
    plt.imshow(vis_img)
    plt.title("Front Camera View")
    plt.axis('off')

    # [오른쪽] BEV 지도 (Occupancy Projection)
    plt.subplot(1, 2, 2)
    # Transpose(1, 0, 2)를 해야 X, Y축이 올바르게 보입니다.
    # origin='lower'로 해야 좌표계가 뒤집히지 않습니다.
    plt.imshow(vis_map.transpose(1, 0, 2), origin='lower')
    
    # 자율주행차 위치 (중앙)
    center_x, center_y = W // 2, H // 2
    plt.plot(center_x, center_y, 'w^', markersize=12, markeredgecolor='k', label='Ego Car')
    
    plt.title(f"3D Occupancy Projected to BEV\n(IoU: {iou_score*100:.1f}%)")
    
    # 범례 추가
    legend_elements = [
        Patch(facecolor='green', label='Match (Correct)'),
        Patch(facecolor='red', label='Pred Only (False Positive)'),
        Patch(facecolor='blue', label='GT Only (Missed)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_overlap()