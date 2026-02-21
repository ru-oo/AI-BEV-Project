"""
dataset_nuscenes_v3.py — FPVNet용 nuScenes 데이터셋
====================================================
변경점 (code2 대비):
  - 6카메라 → 전방 카메라(CAM_FRONT) 단일 입력 (실제 차량 적용 용이)
  - 이미지 크기: 1056×384 → 400×224 (메모리 절약, ~3.5배)
  - 클래스: 4 → 5 (Free/Road/Vehicle/Pedestrian/StaticObstacle)
  - 도로 바닥면 GT 추가 (z 슬라이스 활용)
  - FPVNet 입력: (img, K)  — 단일 카메라 내부 파라미터만 필요

클래스 정의:
  0: Free (빈 공간)
  1: Road (도로/지면)
  2: Vehicle (승용차, 트럭, 버스 등)
  3: Pedestrian (보행자, 자전거)
  4: StaticObstacle (기타 고정 장애물)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    _NUSCENES_AVAILABLE = True
except ImportError:
    _NUSCENES_AVAILABLE = False
    print("[경고] nuscenes-devkit 없음: pip install nuscenes-devkit")


# ─────────────────────────────────────────────────────
# 클래스 매핑
# ─────────────────────────────────────────────────────
CLASS_MAP = {
    # Vehicle (2)
    'vehicle.car':           2,
    'vehicle.emergency':     2,
    'vehicle.truck':         2,
    'vehicle.bus':           2,
    'vehicle.construction':  2,
    'vehicle.trailer':       2,
    # Pedestrian (3)
    'human.pedestrian':      3,
    'vehicle.bicycle':       3,
    'vehicle.motorcycle':    3,
    # Static Obstacle (4)
    'static_object':         4,
    'movable_object':        4,
}

NUM_CLASSES   = 5
CLASS_NAMES   = ['Free', 'Road', 'Vehicle', 'Pedestrian', 'StaticObst']

# 이미지 크기 (FPVNet 입력) — 메모리 절약
IMG_W = 400
IMG_H = 224


class NuScenesV3Dataset(Dataset):
    """
    FPVNet용 nuScenes 단일 전방 카메라 데이터셋

    Parameters
    ----------
    dataroot : nuScenes 데이터 경로 (예: '../data/sets/nuscenesmini')
    version  : 'v1.0-mini' | 'v1.0-trainval'
    is_train : 학습/검증 분리
    xbound   : [xmin, xmax, xstep]
    ybound   : [ymin, ymax, ystep]
    zbound   : [zmin, zmax, zstep]
    """

    def __init__(self,
                 dataroot: str,
                 version:  str   = 'v1.0-mini',
                 is_train: bool  = True,
                 xbound=(-25.0, 25.0, 0.5),
                 ybound=(-25.0, 25.0, 0.5),
                 zbound=(-2.0,  6.0,  1.0)):

        assert _NUSCENES_AVAILABLE, "nuscenes-devkit 설치 필요"

        self.nusc     = NuScenes(version=version,
                                 dataroot=dataroot, verbose=False)
        self.is_train = is_train
        self.xbound   = xbound
        self.ybound   = ybound
        self.zbound   = zbound

        # 복셀 그리드 크기
        self.nX = int((xbound[1] - xbound[0]) / xbound[2])
        self.nY = int((ybound[1] - ybound[0]) / ybound[2])
        self.nZ = int((zbound[1] - zbound[0]) / zbound[2])

        # 전체 샘플 로드 후 train/val 분리 (mini: 총 323 샘플)
        all_samples = list(self.nusc.sample)
        split = int(len(all_samples) * 0.8)
        if is_train:
            self.samples = all_samples[:split]
        else:
            self.samples = all_samples[split:]

        # 이미지 전처리
        _norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        if is_train:
            self.tf = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4,
                    saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(), _norm,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(), _norm,
            ])

        print(f"[NuScenesV3] {'학습' if is_train else '검증'}: "
              f"{len(self.samples)}샘플, "
              f"복셀 {self.nZ}×{self.nX}×{self.nY}, "
              f"이미지 {IMG_H}×{IMG_W}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        # ── 전방 카메라 로드 ──────────────────
        cam_token  = rec['data']['CAM_FRONT']
        cam_sd     = self.nusc.get('sample_data', cam_token)
        cs_rec     = self.nusc.get('calibrated_sensor',
                                   cam_sd['calibrated_sensor_token'])

        img_path = self.nusc.get_sample_data_path(cam_token)
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size                       # 1600, 900
        img = img.resize((IMG_W, IMG_H))
        img_tensor = self.tf(img)                       # (3, H, W)

        # ── 카메라 내부 파라미터 (이미지 크기에 맞게 조정) ──
        K = np.array(cs_rec['camera_intrinsic'], dtype=np.float32)
        K[0] *= IMG_W / orig_w
        K[1] *= IMG_H / orig_h
        K_tensor = torch.from_numpy(K)                  # (3, 3)

        # ── Ego→Camera 변환 (GT 좌표 변환용) ──
        cam_rot   = Quaternion(cs_rec['rotation']).rotation_matrix
        cam_trans = np.array(cs_rec['translation'])

        # ── LIDAR GT 박스 로드 (Ego 좌표계 변환) ──
        lidar_token = rec['data']['LIDAR_TOP']
        lidar_sd    = self.nusc.get('sample_data', lidar_token)
        lidar_cs    = self.nusc.get('calibrated_sensor',
                                    lidar_sd['calibrated_sensor_token'])
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)

        # LIDAR 센서 → Ego 변환
        lidar_rot   = Quaternion(lidar_cs['rotation'])
        lidar_trans = np.array(lidar_cs['translation'])
        for box in boxes:
            box.rotate(lidar_rot)
            box.translate(lidar_trans)

        # ── 3D 복셀 GT 생성 ──────────────────
        gt_voxel = self._boxes_to_voxel(boxes)          # (nZ, nX, nY)

        return img_tensor, K_tensor, gt_voxel

    # ── 박스 → 복셀 변환 ─────────────────────
    def _boxes_to_voxel(self, boxes):
        voxel = np.zeros((self.nZ, self.nX, self.nY), dtype=np.int64)

        # 도로 바닥 슬라이스 (z=0 근처) → Road (1)
        z_road_lo = (0.0 - self.zbound[0]) / self.zbound[2]
        z_road_hi = (0.5 - self.zbound[0]) / self.zbound[2]
        zi_lo = max(0, int(z_road_lo))
        zi_hi = min(self.nZ, int(z_road_hi) + 1)
        voxel[zi_lo:zi_hi, :, :] = 1  # Road

        for box in boxes:
            cls_id = 0
            for prefix, cid in CLASS_MAP.items():
                if box.name.startswith(prefix):
                    cls_id = cid
                    break
            if cls_id == 0:
                continue

            cx, cy, cz = box.center
            w, l, h    = box.wlh

            x0 = (cx - l/2 - self.xbound[0]) / self.xbound[2]
            x1 = (cx + l/2 - self.xbound[0]) / self.xbound[2]
            y0 = (cy - w/2 - self.ybound[0]) / self.ybound[2]
            y1 = (cy + w/2 - self.ybound[0]) / self.ybound[2]
            z0 = (cz - h/2 - self.zbound[0]) / self.zbound[2]
            z1 = (cz + h/2 - self.zbound[0]) / self.zbound[2]

            xi0 = int(max(0, x0));  xi1 = int(min(self.nX, x1 + 1))
            yi0 = int(max(0, y0));  yi1 = int(min(self.nY, y1 + 1))
            zi0 = int(max(0, z0));  zi1 = int(min(self.nZ, z1 + 1))

            if xi0 < xi1 and yi0 < yi1 and zi0 < zi1:
                voxel[zi0:zi1, xi0:xi1, yi0:yi1] = cls_id

        return torch.from_numpy(voxel)  # (nZ, nX, nY)


# ─────────────────────────────────────────────────────
# 데이터셋 테스트
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = NuScenesV3Dataset(
        dataroot='../data/sets/nuscenesmini',
        version='v1.0-mini',
        is_train=True,
    )
    img, K, gt = ds[0]
    print(f"img: {img.shape}  K: {K.shape}  gt: {gt.shape}")
    print(f"클래스 분포: {[(gt==c).sum().item() for c in range(NUM_CLASSES)]}")
