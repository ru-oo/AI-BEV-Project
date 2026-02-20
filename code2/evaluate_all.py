# code/evaluate_all.py 전체 수정 코드

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from train import LSSModel

def evaluate_dataset():
    # === 설정 ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 NVIDIA GPU (CUDA) 가속 활성화됨! 사용 장치: {torch.cuda.get_device_name(0)}")
        
    # 2순위: MPS (Apple Silicon Mac) 확인
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Apple M1/M2/M3 MPS 가속 활성화됨!")
        
    # 3순위: CPU (모두 없을 경우)
    else:
        device = torch.device("cpu")
        print("⚠️ GPU를 찾을 수 없습니다. CPU로 실행합니다.")
    batch_size = 4  
    threshold = 0.4 # 확률이 40% 이상이면 물체로 판단
    
    print(f"평가 장치: {device}")

    # 1. 모델 준비 (학습 코드와 설정 통일)
    # [수정] zbound를 4개 층 설정으로 변경
    model = LSSModel(xbound=[-50, 50, 0.5], 
                     ybound=[-50, 50, 0.5], 
                     zbound=[-2.0, 6.0, 2.0], 
                     dbound=[4, 45, 1]).to(device)

    model_path = "best_lss_multicam.pth" 
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 모델 로드 성공: {model_path}")
    except FileNotFoundError:
        print(f"❌ {model_path} 파일이 없습니다. 먼저 학습을 진행하세요.")
        return
    except Exception as e:
        print(f"❌ 모델 로드 에러: {e}")
        return

    model.eval()

    # 2. 데이터셋 로드
    dataset = NuScenesDataset(dataroot='../data/sets/nuscenesmini', is_train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"총 데이터 개수: {len(dataset)}개")
    print("🚀 3D Occupancy IoU 평가 시작...")
    
    total_intersection = 0
    total_union = 0
    
    with torch.no_grad():
        for i, (imgs, intrinsics, sensor2ego, gt_bev) in enumerate(tqdm(loader)):
            imgs = imgs.to(device)
            intrinsics = intrinsics.float().to(device)
            gt_bev = gt_bev.to(device) # (B, 4, 200, 200)
            
            rots = sensor2ego[:, :, :3, :3].float().to(device)
            trans = sensor2ego[:, :, :3, 3].float().to(device)
            
            # 예측 (B, 4, 200, 200)
            preds = model(imgs, rots, trans, intrinsics)
            preds_prob = torch.sigmoid(preds)
            
            preds_np = preds_prob.cpu().numpy()
            gt_np = gt_bev.cpu().numpy()
            
            # 3D Voxel 단위 이진화
            pred_binary = (preds_np > threshold).astype(int)
            gt_binary = (gt_np > 0.5).astype(int)
            
            # 교집합 & 합집합 (모든 층 포함)
            intersection = (pred_binary & gt_binary).sum()
            union = (pred_binary | gt_binary).sum()
            
            total_intersection += intersection
            total_union += union

    # 최종 mIoU 계산
    if total_union == 0:
        final_iou = 0.0
    else:
        final_iou = total_intersection / total_union

    print("\n" + "="*40)
    print(f"📊 최종 3D Occupancy 평가 결과")
    print(f" - 전체 3D mIoU: {final_iou * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_dataset()