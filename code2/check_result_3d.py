import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from train import LSSModel

# 클래스 색상 (RGB 0~1)
CLASS_COLORS = np.array([
    [0.10, 0.10, 0.10],   # 0: Empty
    [0.20, 0.40, 1.00],   # 1: Car
    [1.00, 0.55, 0.00],   # 2: Truck/Bus
    [0.15, 0.80, 0.20],   # 3: Pedestrian
])
CLASS_NAMES = ['Empty', 'Car', 'Truck/Bus', 'Pedestrian/Bike']


def make_semantic_bev(class_map):
    """class_map: (H, W) int → (H, W, 3) RGB"""
    return CLASS_COLORS[class_map.clip(0, len(CLASS_COLORS) - 1)]


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


def check():
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
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

    model.eval()

    dataset = NuScenesDataset(dataroot='../data/sets/nuscenesmini', is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    imgs, intrinsics, sensor2ego, gt_bev = next(iter(loader))

    imgs = imgs.to(device)
    intrinsics = intrinsics.float().to(device)
    rots = sensor2ego[:, :, :3, :3].float().to(device)
    trans = sensor2ego[:, :, :3, 3].float().to(device)

    with torch.no_grad():
        preds = model(imgs, rots, trans, intrinsics)  # (1, 4, nz, 200, 200)
        pred_classes = torch.argmax(preds, dim=1)     # (1, nz, 200, 200)

    pred_cube = pred_classes[0].cpu().numpy()   # (nz, 200, 200)
    gt_cube = gt_bev[0].cpu().numpy()           # (nz, 200, 200)

    layer_names = ["-2m~0m", "0m~2m", "2m~4m", "4m~6m"]
    nz = pred_cube.shape[0]

    fig, axes = plt.subplots(nz, 3, figsize=(12, nz * 3.5))
    fig.suptitle("3D Semantic Voxel: Pred vs GT vs Diff", fontsize=13, fontweight='bold')

    for z in range(nz):
        pred_layer = pred_cube[z]   # (200, 200)
        gt_layer = gt_cube[z]

        # 예측 시맨틱 맵
        axes[z, 0].imshow(make_semantic_bev(pred_layer).transpose(1, 0, 2), origin='lower')
        axes[z, 0].set_title(f"Pred Z{z}: {layer_names[z]}")
        axes[z, 0].axis('off')

        # GT 시맨틱 맵
        axes[z, 1].imshow(make_semantic_bev(gt_layer).transpose(1, 0, 2), origin='lower')
        axes[z, 1].set_title(f"GT Z{z}: {layer_names[z]}")
        axes[z, 1].axis('off')

        # 차이 맵: 초록=일치(전경), 빨강=오탐(FP), 파랑=미탐(FN), 검정=배경 일치
        diff_rgb = np.zeros((200, 200, 3))
        p_fg = pred_layer > 0
        g_fg = gt_layer > 0
        diff_rgb[(p_fg) & (g_fg)]   = [0.0, 0.9, 0.0]   # TP (초록)
        diff_rgb[(p_fg) & (~g_fg)]  = [0.9, 0.0, 0.0]   # FP (빨강)
        diff_rgb[(~p_fg) & (g_fg)]  = [0.0, 0.0, 0.9]   # FN (파랑)

        axes[z, 2].imshow(diff_rgb.transpose(1, 0, 2), origin='lower')
        axes[z, 2].set_title(f"Diff Z{z}: G=TP R=FP B=FN")
        axes[z, 2].axis('off')

    # 범례
    legend_patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
                      for i in range(len(CLASS_NAMES))]
    legend_patches += [
        mpatches.Patch(color=[0, 0.9, 0], label='TP (Diff)'),
        mpatches.Patch(color=[0.9, 0, 0], label='FP (Diff)'),
        mpatches.Patch(color=[0, 0, 0.9], label='FN (Diff)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/check_result_3d.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("결과 저장: results/check_result_3d.png")


if __name__ == "__main__":
    check()
