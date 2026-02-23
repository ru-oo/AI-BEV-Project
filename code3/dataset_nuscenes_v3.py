"""
dataset_nuscenes_v3.py — FastOcc 6-Camera Surround 데이터셋
============================================================
nuScenes 6방향 카메라 전체 사용 (code2와 동일한 카메라 구성)

카메라 순서:
  0: CAM_FRONT_LEFT   1: CAM_FRONT   2: CAM_FRONT_RIGHT
  3: CAM_BACK_LEFT    4: CAM_BACK    5: CAM_BACK_RIGHT

반환:
  imgs      : (6, 3, H, W)   정규화 이미지
  Ks        : (6, 3, 3)       각 카메라 내부 파라미터
  sensor2ego: (6, 4, 4)       cam→ego 변환행렬
  gt_voxel  : (nZ, nX, nY)   3D Semantic GT (long)

클래스 (5):
  0=Free  1=Road  2=Vehicle  3=Pedestrian  4=StaticObst
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

# ── 카메라 순서 ────────────────────────────────────────
CAMERAS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]
NUM_CAMS = len(CAMERAS)

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
NUM_CLASSES = 5
CLASS_NAMES = ['Free', 'Road', 'Vehicle', 'Pedestrian', 'StaticObst']

# 기본 이미지 크기 (메모리 균형)
IMG_H = 256
IMG_W = 704


class NuScenesV3Dataset(Dataset):
    """
    FastOcc 6-Camera Surround 학습용 데이터셋

    Parameters
    ----------
    dataroot   : nuScenes 루트 경로
    version    : 'v1.0-mini' | 'v1.0-trainval'
    is_train   : 학습/검증 구분
    xbound     : (min, max, step)  BEV x 범위
    ybound     : (min, max, step)  BEV y 범위
    zbound     : (min, max, step)  높이 범위
    img_h, img_w : 입력 이미지 크기
    """

    def __init__(self,
                 dataroot: str,
                 version:  str  = 'v1.0-mini',
                 is_train: bool = True,
                 xbound=(-50., 50., .5),
                 ybound=(-50., 50., .5),
                 zbound=(-2.,  6.,  .5),
                 img_h: int = IMG_H,
                 img_w: int = IMG_W):

        assert _NS_OK, 'nuscenes-devkit 설치: pip install nuscenes-devkit'
        self.nusc     = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.is_train = is_train
        self.img_h    = img_h
        self.img_w    = img_w
        self.xbound   = xbound
        self.ybound   = ybound
        self.zbound   = zbound

        self.nX = int((xbound[1] - xbound[0]) / xbound[2])
        self.nY = int((ybound[1] - ybound[0]) / ybound[2])
        self.nZ = int((zbound[1] - zbound[0]) / zbound[2])

        # 80:20 학습/검증 분리
        all_s = list(self.nusc.sample)
        cut   = int(len(all_s) * 0.8)
        self.samples = all_s[:cut] if is_train else all_s[cut:]

        # 이미지 전처리
        _n = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        if is_train:
            self.tf = transforms.Compose([
                transforms.ColorJitter(.4, .4, .4, .1),
                transforms.RandomGrayscale(.1),
                transforms.ToTensor(), _n])
        else:
            self.tf = transforms.Compose([transforms.ToTensor(), _n])

        print(f'[NuScenesV3] {"학습" if is_train else "검증"}: '
              f'{len(self.samples)}샘플 | 6카메라 서라운드뷰 | '
              f'복셀 {self.nZ}×{self.nX}×{self.nY} | {img_h}×{img_w}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        imgs_list = []
        K_list    = []
        s2e_list  = []

        # ── 6카메라 데이터 로드 ────────────────────────
        for cam in CAMERAS:
            tok   = rec['data'][cam]
            sd    = self.nusc.get('sample_data', tok)
            cs    = self.nusc.get('calibrated_sensor',
                                   sd['calibrated_sensor_token'])

            # 이미지 로드 & 리사이즈
            img   = Image.open(self.nusc.get_sample_data_path(tok)).convert('RGB')
            ow, oh = img.size          # 원본 1600×900
            img   = img.resize((self.img_w, self.img_h))
            imgs_list.append(self.tf(img))  # (3, H, W)

            # 내부 파라미터 K (이미지 크기 보정)
            K = np.array(cs['camera_intrinsic'], dtype=np.float32)
            K[0] *= self.img_w / ow
            K[1] *= self.img_h / oh
            K_list.append(torch.from_numpy(K))   # (3, 3)

            # sensor2ego (cam→ego)
            rot   = Quaternion(cs['rotation']).rotation_matrix
            trans = np.array(cs['translation'], dtype=np.float32)
            s2e   = np.eye(4, dtype=np.float32)
            s2e[:3, :3] = rot
            s2e[:3,  3] = trans
            s2e_list.append(torch.from_numpy(s2e))   # (4, 4)

        imgs       = torch.stack(imgs_list)   # (6, 3, H, W)
        Ks         = torch.stack(K_list)      # (6, 3, 3)
        sensor2ego = torch.stack(s2e_list)    # (6, 4, 4)

        # ── LIDAR 박스 → Ego 좌표계 → GT 복셀 ──────────
        lidar_tok = rec['data']['LIDAR_TOP']
        lidar_sd  = self.nusc.get('sample_data', lidar_tok)
        lidar_cs  = self.nusc.get('calibrated_sensor',
                                   lidar_sd['calibrated_sensor_token'])
        _, boxes, _ = self.nusc.get_sample_data(lidar_tok)

        # LIDAR 센서 → Ego 변환
        l_rot   = Quaternion(lidar_cs['rotation'])
        l_trans = np.array(lidar_cs['translation'])
        for b in boxes:
            b.rotate(l_rot)
            b.translate(l_trans)

        gt = self._boxes_to_voxel(boxes)   # (nZ, nX, nY) long

        return imgs, Ks, sensor2ego, gt

    # ── 박스 → 3D 복셀 GT ─────────────────────────────
    def _boxes_to_voxel(self, boxes):
        voxel = np.zeros((self.nZ, self.nX, self.nY), dtype=np.int64)

        # 도로 바닥 (z≈0) → Road(1)
        zi_lo = max(0, int((0.0 - self.zbound[0]) / self.zbound[2]))
        zi_hi = min(self.nZ, zi_lo + 1)
        voxel[zi_lo:zi_hi, :, :] = 1

        for box in boxes:
            cls = 0
            for pfx, cid in CLASS_MAP.items():
                if box.name.startswith(pfx):
                    cls = cid; break
            if cls == 0:
                continue

            cx, cy, cz = box.center
            w, l, h    = box.wlh

            def vi(v, lo, step, n):
                return int(np.clip((v - lo) / step, 0, n))

            xi0 = vi(cx - l/2, self.xbound[0], self.xbound[2], self.nX)
            xi1 = vi(cx + l/2, self.xbound[0], self.xbound[2], self.nX)
            yi0 = vi(cy - w/2, self.ybound[0], self.ybound[2], self.nY)
            yi1 = vi(cy + w/2, self.ybound[0], self.ybound[2], self.nY)
            zi0 = vi(cz - h/2, self.zbound[0], self.zbound[2], self.nZ)
            zi1 = vi(cz + h/2, self.zbound[0], self.zbound[2], self.nZ)

            if xi0 < xi1 and yi0 < yi1 and zi0 < zi1:
                voxel[zi0:zi1, xi0:xi1, yi0:yi1] = cls

        return torch.from_numpy(voxel)


# ── 빠른 테스트 ───────────────────────────────────────
if __name__ == '__main__':
    ds = NuScenesV3Dataset('../data/sets/nuscenesmini',
                            version='v1.0-mini', is_train=True)
    imgs, Ks, s2e, gt = ds[0]
    print(f'imgs: {imgs.shape}')        # (6, 3, 256, 704)
    print(f'Ks:   {Ks.shape}')          # (6, 3, 3)
    print(f's2e:  {s2e.shape}')         # (6, 4, 4)
    print(f'gt:   {gt.shape}')          # (nZ, nX, nY)
    dist = {n: (gt==i).sum().item() for i,n in enumerate(CLASS_NAMES)}
    print(f'GT 분포: {dist}')
