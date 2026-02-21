"""
dataset_nuscenes_v3.py — FastOcc용 nuScenes 데이터셋
=====================================================
단일 전방 카메라(CAM_FRONT) 기반, 5클래스 semantic occupancy GT 생성

클래스:
  0: Free          (빈 공간)
  1: Road          (도로/지면)
  2: Vehicle       (승용차·트럭·버스)
  3: Pedestrian    (보행자·자전거)
  4: StaticObstacle (고정 장애물)

반환 (return_s2e=True):
  img      : (3, H, W)        — 정규화된 이미지
  K        : (3, 3)           — 카메라 내부 파라미터 (이미지 크기 보정)
  sensor2ego: (4, 4)          — 카메라→ego 변환행렬
  gt_voxel : (nZ, nX, nY)    — 3D Semantic GT (long)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    _NS_OK = True
except ImportError:
    _NS_OK = False
    print('[경고] nuscenes-devkit 없음: pip install nuscenes-devkit')

# ── 클래스 매핑 ───────────────────────────────────────
CLASS_MAP = {
    'vehicle.car':          2,
    'vehicle.emergency':    2,
    'vehicle.truck':        2,
    'vehicle.bus':          2,
    'vehicle.construction': 2,
    'vehicle.trailer':      2,
    'human.pedestrian':     3,
    'vehicle.bicycle':      3,
    'vehicle.motorcycle':   3,
    'movable_object':       4,
    'static_object':        4,
}
NUM_CLASSES  = 5
CLASS_NAMES  = ['Free', 'Road', 'Vehicle', 'Pedestrian', 'StaticObst']

# 기본 이미지 크기
IMG_H = 224
IMG_W = 400


class NuScenesV3Dataset(Dataset):
    """
    FastOcc 학습용 nuScenes 데이터셋

    Parameters
    ----------
    dataroot  : nuScenes 루트 경로
    version   : 'v1.0-mini' | 'v1.0-trainval'
    is_train  : 학습/검증 구분
    return_s2e: True면 sensor2ego 행렬 반환 (FastOcc 투영용)
    img_h/w   : 입력 이미지 크기
    """

    def __init__(self,
                 dataroot: str,
                 version:  str  = 'v1.0-mini',
                 is_train: bool = True,
                 xbound=(-25., 25., .5),
                 ybound=(-25., 25., .5),
                 zbound=(-2.,  6.,  .5),
                 img_h: int  = IMG_H,
                 img_w: int  = IMG_W,
                 return_s2e: bool = True):

        assert _NS_OK, 'nuscenes-devkit 설치 필요: pip install nuscenes-devkit'

        self.nusc       = NuScenes(version=version,
                                   dataroot=dataroot, verbose=False)
        self.is_train   = is_train
        self.return_s2e = return_s2e
        self.img_h      = img_h
        self.img_w      = img_w
        self.xbound     = xbound
        self.ybound     = ybound
        self.zbound     = zbound

        # 복셀 크기
        self.nX = int((xbound[1]-xbound[0]) / xbound[2])
        self.nY = int((ybound[1]-ybound[0]) / ybound[2])
        self.nZ = int((zbound[1]-zbound[0]) / zbound[2])

        # 학습 80% / 검증 20% 분리
        all_samps = list(self.nusc.sample)
        split = int(len(all_samps) * 0.8)
        self.samples = all_samps[:split] if is_train else all_samps[split:]

        # 이미지 전처리
        _n = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        if is_train:
            self.tf = transforms.Compose([
                transforms.ColorJitter(.4,.4,.4,.1),
                transforms.RandomGrayscale(.1),
                transforms.ToTensor(), _n])
        else:
            self.tf = transforms.Compose([transforms.ToTensor(), _n])

        print(f'[NuScenesV3] {"학습" if is_train else "검증"}: '
              f'{len(self.samples)} 샘플 | '
              f'복셀 {self.nZ}×{self.nX}×{self.nY} | '
              f'이미지 {img_h}×{img_w}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        # ── 전방 카메라 로드 ──────────────────────────
        cam_tok = rec['data']['CAM_FRONT']
        cam_sd  = self.nusc.get('sample_data', cam_tok)
        cs      = self.nusc.get('calibrated_sensor',
                                cam_sd['calibrated_sensor_token'])

        img = Image.open(self.nusc.get_sample_data_path(cam_tok)).convert('RGB')
        ow, oh = img.size                        # 원본: 1600×900
        img    = img.resize((self.img_w, self.img_h))
        img_t  = self.tf(img)                    # (3, H, W)

        # ── 내부 파라미터 K (이미지 크기 보정) ────────
        K = np.array(cs['camera_intrinsic'], dtype=np.float32)
        K[0] *= self.img_w / ow
        K[1] *= self.img_h / oh
        K_t = torch.from_numpy(K)               # (3, 3)

        # ── sensor2ego 변환행렬 ────────────────────────
        rot   = Quaternion(cs['rotation']).rotation_matrix
        trans = np.array(cs['translation'], dtype=np.float32)
        s2e   = np.eye(4, dtype=np.float32)
        s2e[:3, :3] = rot
        s2e[:3,  3] = trans
        s2e_t = torch.from_numpy(s2e)           # (4, 4)

        # ── LIDAR 박스 → Ego 좌표 → GT 복셀 ──────────
        lidar_tok = rec['data']['LIDAR_TOP']
        lidar_sd  = self.nusc.get('sample_data', lidar_tok)
        lidar_cs  = self.nusc.get('calibrated_sensor',
                                   lidar_sd['calibrated_sensor_token'])
        _, boxes, _ = self.nusc.get_sample_data(lidar_tok)

        lidar_rot   = Quaternion(lidar_cs['rotation'])
        lidar_trans = np.array(lidar_cs['translation'])
        for b in boxes:
            b.rotate(lidar_rot)
            b.translate(lidar_trans)

        gt = self._boxes_to_voxel(boxes)         # (nZ, nX, nY) long

        if self.return_s2e:
            return img_t, K_t, s2e_t, gt
        return img_t, K_t, gt

    # ── 박스 → 복셀 GT ────────────────────────────────
    def _boxes_to_voxel(self, boxes):
        voxel = np.zeros((self.nZ, self.nX, self.nY), dtype=np.int64)

        # 도로 바닥 (z ≈ 0) → Road(1)
        zi_road = int(max(0, (0.0 - self.zbound[0]) / self.zbound[2]))
        voxel[max(0, zi_road):min(self.nZ, zi_road+1), :, :] = 1

        for box in boxes:
            cls = 0
            for pfx, cid in CLASS_MAP.items():
                if box.name.startswith(pfx):
                    cls = cid; break
            if cls == 0:
                continue

            cx, cy, cz = box.center
            w, l, h    = box.wlh

            def idx(v, lo, step, n):
                return int(max(0, min(n, (v-lo)/step)))

            xi0 = idx(cx - l/2, self.xbound[0], self.xbound[2], self.nX)
            xi1 = idx(cx + l/2, self.xbound[0], self.xbound[2], self.nX)
            yi0 = idx(cy - w/2, self.ybound[0], self.ybound[2], self.nY)
            yi1 = idx(cy + w/2, self.ybound[0], self.ybound[2], self.nY)
            zi0 = idx(cz - h/2, self.zbound[0], self.zbound[2], self.nZ)
            zi1 = idx(cz + h/2, self.zbound[0], self.zbound[2], self.nZ)

            if xi0 < xi1 and yi0 < yi1 and zi0 < zi1:
                voxel[zi0:zi1, xi0:xi1, yi0:yi1] = cls

        return torch.from_numpy(voxel)           # (nZ, nX, nY)


# ── 빠른 테스트 ───────────────────────────────────────
if __name__ == '__main__':
    ds = NuScenesV3Dataset('../data/sets/nuscenesmini',
                            version='v1.0-mini', is_train=True)
    img, K, s2e, gt = ds[0]
    print(f'img: {img.shape}  K: {K.shape}  s2e: {s2e.shape}  gt: {gt.shape}')
    dist = [(gt == c).sum().item() for c in range(NUM_CLASSES)]
    print(f'클래스 분포: {dict(zip(CLASS_NAMES, dist))}')
