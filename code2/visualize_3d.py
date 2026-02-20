import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from train import LSSModel

# 클래스 색상 (RGB 0~1)
CLASS_COLORS = np.array([
    [0.10, 0.10, 0.10],   # 0: Empty      → 짙은 회색 (배경)
    [0.20, 0.40, 1.00],   # 1: Car        → 파랑
    [1.00, 0.55, 0.00],   # 2: Truck/Bus  → 주황
    [0.15, 0.80, 0.20],   # 3: Pedestrian → 초록
])
CLASS_NAMES = ['Empty', 'Car', 'Truck/Bus', 'Pedestrian/Bike']


def make_semantic_bev(class_map_2d):
    """
    class_map_2d: (X, Y) ndarray, 값=클래스 ID (0~3)
    반환: (X, Y, 3) RGB 이미지
    """
    H, W = class_map_2d.shape
    rgb = CLASS_COLORS[class_map_2d.clip(0, len(CLASS_COLORS) - 1)]  # (H, W, 3)
    return rgb


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


def visualize_3d():
    device = get_device()

    model = LSSModel(
        xbound=[-50, 50, 0.5],
        ybound=[-50, 50, 0.5],
        zbound=[-2.0, 6.0, 2.0],
        dbound=[4, 45, 1],
        num_classes=4
    ).to(device)

    model_path = "best_semantic_mini_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"모델 로드 완료: {model_path}")
    except Exception as e:
        print(f"모델 파일 로드 실패: {e}")
        return

    model.eval()

    dataset = NuScenesDataset(dataroot='../data/sets/nuscenesmini', is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    try:
        imgs, intrinsics, sensor2ego, gt_bev = next(iter(loader))
    except StopIteration:
        print("데이터가 없습니다.")
        return

    imgs = imgs.to(device)
    intrinsics = intrinsics.float().to(device)
    rots = sensor2ego[:, :, :3, :3].float().to(device)
    trans = sensor2ego[:, :, :3, 3].float().to(device)

    print("3D 공간 예측 중...")
    with torch.no_grad():
        preds = model(imgs, rots, trans, intrinsics)  # (1, 4, nz, 200, 200)
        pred_classes = torch.argmax(preds, dim=1)     # (1, nz, 200, 200)

    # (nz, 200, 200) → Z축 argmax로 2D BEV 클래스 맵 생성
    # 각 (X, Y) 위치에서 "가장 높은 Z층에 있는 물체" 클래스 표시
    pred_3d = pred_classes[0].cpu().numpy()   # (nz, 200, 200)
    gt_3d = gt_bev[0].cpu().numpy()           # (nz, 200, 200)

    layer_names = ["-2m~0m", "0m~2m", "2m~4m", "4m~6m"]

    # === 레이아웃: 상단 = Z층별 비교, 하단 = 2D BEV 요약 ===
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("3D Voxel Semantic Prediction", fontsize=14, fontweight='bold')

    nz = pred_3d.shape[0]

    # 상단: Z층별 예측 vs GT (클래스 색상)
    for z in range(nz):
        pred_layer = pred_3d[z]        # (200, 200)
        gt_layer = gt_3d[z]

        # 예측
        ax = fig.add_subplot(3, nz * 2, z * 2 + 1)
        pred_rgb = make_semantic_bev(pred_layer)
        ax.imshow(pred_rgb.transpose(1, 0, 2), origin='lower')
        ax.set_title(f"Pred Z{z}: {layer_names[z]}", fontsize=9)
        ax.axis('off')

        # GT
        ax = fig.add_subplot(3, nz * 2, z * 2 + 2)
        gt_rgb = make_semantic_bev(gt_layer)
        ax.imshow(gt_rgb.transpose(1, 0, 2), origin='lower')
        ax.set_title(f"GT Z{z}: {layer_names[z]}", fontsize=9)
        ax.axis('off')

    # 중간: 2D BEV 요약 (Z 전체 Max Pooling → 지배 클래스)
    # Empty(0)을 제외하고 전경 클래스가 있으면 그걸 표시
    pred_2d = np.zeros((200, 200), dtype=np.int64)  # 기본 Empty
    gt_2d = np.zeros((200, 200), dtype=np.int64)
    for z in range(nz - 1, -1, -1):  # 아래층부터 위층 순으로 덮어쓰기
        mask = pred_3d[z] > 0
        pred_2d[mask] = pred_3d[z][mask]
        mask_gt = gt_3d[z] > 0
        gt_2d[mask_gt] = gt_3d[z][mask_gt]

    ax_pred_2d = fig.add_subplot(3, 2, 5)
    ax_pred_2d.imshow(make_semantic_bev(pred_2d).transpose(1, 0, 2), origin='lower')
    ax_pred_2d.set_title("Pred BEV (Z-max)", fontsize=11)
    ax_pred_2d.set_xlabel("Y (Left-Right)")
    ax_pred_2d.set_ylabel("X (Front-Back)")

    ax_gt_2d = fig.add_subplot(3, 2, 6)
    ax_gt_2d.imshow(make_semantic_bev(gt_2d).transpose(1, 0, 2), origin='lower')
    ax_gt_2d.set_title("GT BEV (Z-max)", fontsize=11)
    ax_gt_2d.set_xlabel("Y (Left-Right)")
    ax_gt_2d.set_ylabel("X (Front-Back)")

    # 에고카 위치 표시
    for ax in [ax_pred_2d, ax_gt_2d]:
        ax.plot(100, 100, 'w^', markersize=8, label='Ego')
        ax.legend(loc='upper right', fontsize=8)

    # 범례
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    visualize_3d()
