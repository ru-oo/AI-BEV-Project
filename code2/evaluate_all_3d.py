import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from train import LSSModel

CLASS_NAMES = ['Empty', 'Car', 'Truck/Bus', 'Pedestrian/Bike']

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"NVIDIA GPU 사용: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple MPS 사용")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    return device

def evaluate_dataset():
    device = get_device()
    batch_size = 4
    num_classes = 4

    model = LSSModel(
        xbound=[-50, 50, 0.5],
        ybound=[-50, 50, 0.5],
        zbound=[-2.0, 6.0, 2.0],
        dbound=[4, 45, 1],
        num_classes=num_classes
    ).to(device)

    model_path = "best_semantic_mini_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 파일 로드 실패: {e}")
        return

    model.eval()

    dataset = NuScenesDataset(dataroot='../data/sets/nuscenesmini', is_train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n3D Semantic mIoU 평가 시작 (데이터셋: {len(dataset)}개 샘플)...")

    # 클래스별 TP, FP, FN 누적
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for imgs, intrinsics, sensor2ego, gt_bev in tqdm(loader):
            imgs = imgs.to(device)
            intrinsics = intrinsics.float().to(device)
            rots = sensor2ego[:, :, :3, :3].float().to(device)
            trans = sensor2ego[:, :, :3, 3].float().to(device)
            gt_bev = gt_bev.to(device)  # (B, nz, X, Y) Long

            preds = model(imgs, rots, trans, intrinsics)  # (B, num_classes, nz, X, Y)

            # argmax로 예측 클래스 결정
            pred_classes = torch.argmax(preds, dim=1)  # (B, nz, X, Y)

            pred_np = pred_classes.cpu().numpy()
            gt_np = gt_bev.cpu().numpy()

            for c in range(num_classes):
                pred_c = (pred_np == c)
                gt_c = (gt_np == c)
                tp[c] += int((pred_c & gt_c).sum())
                fp[c] += int((pred_c & ~gt_c).sum())
                fn[c] += int((~pred_c & gt_c).sum())

    print("\n" + "=" * 50)
    print(f"{'클래스':<18} {'IoU':>8}  {'TP':>10}  {'FP':>10}  {'FN':>10}")
    print("-" * 50)

    iou_list = []
    for c in range(num_classes):
        denom = tp[c] + fp[c] + fn[c]
        iou = tp[c] / denom if denom > 0 else 0.0
        iou_list.append(iou)
        print(f"  {CLASS_NAMES[c]:<16} {iou * 100:>7.2f}%  {tp[c]:>10}  {fp[c]:>10}  {fn[c]:>10}")

    mean_iou = np.mean(iou_list)
    fg_iou = np.mean(iou_list[1:])  # Empty 제외

    print("=" * 50)
    print(f"  전체 mIoU (4 classes): {mean_iou * 100:.2f}%")
    print(f"  전경 mIoU (3 classes, Empty 제외): {fg_iou * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_dataset()
