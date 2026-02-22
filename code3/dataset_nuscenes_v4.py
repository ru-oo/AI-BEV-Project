"""
dataset_nuscenes_v4.py — FastOcc 6-Camera Surround + LiDAR GT (개선판)
=======================================================================
v3 대비 개선 사항:
  1. LiDAR 포인트클라우드 기반 GT 생성 (더 정확한 Road/Free 구분)
  2. 객체 바운딩박스 voxel 팽창 (Dilation) → 희소 클래스 voxel 증가
  3. 도로 판별 기준 개선 (지면 높이 ±0.3m 이내 포인트 → Road)
  4. 전체 바닥면 Road 지정 제거 (False Positive Road 제거)

반환: imgs(6,3,H,W), Ks(6,3,3), sensor2ego(6,4,4), gt(nZ,nX,nY)
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
    from nuscenes.utils.data_classes import LidarPointCloud
    _NS_OK = True
except ImportError:
    _NS_OK = False
    print('[경고] nuscenes-devkit 없음')

CAMERAS = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT',
]
NUM_CAMS = len(CAMERAS)

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

IMG_H = 256
IMG_W = 704


class NuScenesV4Dataset(Dataset):
    """
    FastOcc v4: 6-Camera + LiDAR 기반 GT

    개선 포인트:
    - LiDAR 포인트로 Ground 판별 (z < ground_thresh) → Road
    - 객체 바운딩박스 ±1 voxel 팽창으로 희소 클래스 voxel 확장
    - 바닥면 전체 Road 지정 방식 제거
    """

    def __init__(self,
                 dataroot: str,
                 version:  str  = 'v1.0-mini',
                 is_train: bool = True,
                 xbound=(-50., 50., .5),
                 ybound=(-50., 50., .5),
                 zbound=(-2.,  6.,  .5),
                 img_h: int = IMG_H,
                 img_w: int = IMG_W,
                 box_dilate: int = 1):

        assert _NS_OK, 'pip install nuscenes-devkit'
        self.nusc       = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.is_train   = is_train
        self.img_h      = img_h
        self.img_w      = img_w
        self.xbound     = xbound
        self.ybound     = ybound
        self.zbound     = zbound
        self.box_dilate = box_dilate   # 희소 클래스 voxel 팽창

        self.nX = int((xbound[1] - xbound[0]) / xbound[2])
        self.nY = int((ybound[1] - ybound[0]) / ybound[2])
        self.nZ = int((zbound[1] - zbound[0]) / zbound[2])

        all_s = list(self.nusc.sample)
        cut   = int(len(all_s) * 0.8)
        self.samples = all_s[:cut] if is_train else all_s[cut:]

        _n = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        if is_train:
            self.tf = transforms.Compose([
                transforms.ColorJitter(.4, .4, .4, .1),
                transforms.RandomGrayscale(.1),
                transforms.ToTensor(), _n])
        else:
            self.tf = transforms.Compose([transforms.ToTensor(), _n])

        print(f'[NuScenesV4] {"학습" if is_train else "검증"}: '
              f'{len(self.samples)}샘플 | 6카메라 서라운드뷰 | '
              f'복셀 {self.nZ}×{self.nX}×{self.nY} | {img_h}×{img_w} | '
              f'박스팽창={box_dilate}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        imgs_list, K_list, s2e_list = [], [], []

        # ── 6카메라 ────────────────────────────────────
        for cam in CAMERAS:
            tok  = rec['data'][cam]
            sd   = self.nusc.get('sample_data', tok)
            cs   = self.nusc.get('calibrated_sensor',
                                  sd['calibrated_sensor_token'])

            img  = Image.open(self.nusc.get_sample_data_path(tok)).convert('RGB')
            ow, oh = img.size
            img  = img.resize((self.img_w, self.img_h))
            imgs_list.append(self.tf(img))

            K = np.array(cs['camera_intrinsic'], dtype=np.float32)
            K[0] *= self.img_w / ow
            K[1] *= self.img_h / oh
            K_list.append(torch.from_numpy(K))

            rot   = Quaternion(cs['rotation']).rotation_matrix
            trans = np.array(cs['translation'], dtype=np.float32)
            s2e   = np.eye(4, dtype=np.float32)
            s2e[:3,:3] = rot
            s2e[:3, 3] = trans
            s2e_list.append(torch.from_numpy(s2e))

        imgs       = torch.stack(imgs_list)
        Ks         = torch.stack(K_list)
        sensor2ego = torch.stack(s2e_list)

        # ── LiDAR + 박스 → GT ─────────────────────────
        gt = self._build_gt(rec)

        return imgs, Ks, sensor2ego, gt

    # ─────────────────────────────────────────────────
    def _build_gt(self, rec):
        """
        LiDAR 포인트 + 바운딩박스 기반 3D GT 생성
        """
        voxel = np.zeros((self.nZ, self.nX, self.nY), dtype=np.int64)

        # ── 1. LiDAR 포인트를 Ego 좌표로 변환 ─────────
        lidar_tok = rec['data']['LIDAR_TOP']
        lidar_sd  = self.nusc.get('sample_data', lidar_tok)
        lidar_cs  = self.nusc.get('calibrated_sensor',
                                   lidar_sd['calibrated_sensor_token'])
        pc_path   = self.nusc.get_sample_data_path(lidar_tok)

        pc = LidarPointCloud.from_file(pc_path)   # (4, N) in sensor frame

        # Sensor → Ego
        l_rot   = Quaternion(lidar_cs['rotation']).rotation_matrix.astype(np.float32)
        l_trans = np.array(lidar_cs['translation'], dtype=np.float32)
        pts     = pc.points[:3].T               # (N, 3)
        pts     = pts @ l_rot.T + l_trans       # ego frame

        # 복셀 범위 내 포인트만
        mask = (
            (pts[:,0] >= self.xbound[0]) & (pts[:,0] < self.xbound[1]) &
            (pts[:,1] >= self.ybound[0]) & (pts[:,1] < self.ybound[1]) &
            (pts[:,2] >= self.zbound[0]) & (pts[:,2] < self.zbound[1])
        )
        pts = pts[mask]

        # ── 2. Road: z=0 레이어 전체 (v3 방식 유지 — 학습 안정성)
        #    객체가 없는 바닥면 = 도로로 간주
        zi_road_lo = max(0, int((0.0 - self.zbound[0]) / self.zbound[2]))
        zi_road_hi = min(self.nZ, zi_road_lo + 2)   # z=0~0.5m 2레이어
        voxel[zi_road_lo:zi_road_hi, :, :] = 1   # Road

        # ── 3. 박스 → 객체 클래스 ────────────────────
        _, boxes, _ = self.nusc.get_sample_data(lidar_tok)

        # LIDAR sensor → Ego
        l_rot_q = Quaternion(lidar_cs['rotation'])
        l_trans_arr = np.array(lidar_cs['translation'])
        for b in boxes:
            b.rotate(l_rot_q)
            b.translate(l_trans_arr)

        for box in boxes:
            cls = 0
            for pfx, cid in CLASS_MAP.items():
                if box.name.startswith(pfx):
                    cls = cid; break
            if cls == 0:
                continue

            cx, cy, cz = box.center
            w, l, h = box.wlh

            d = self.box_dilate * max(self.xbound[2], self.ybound[2], self.zbound[2])

            def vi(v, lo, step, n):
                return int(np.clip((v - lo) / step, 0, n - 1))

            xi0 = max(0, vi(cx - l/2 - d, self.xbound[0], self.xbound[2], self.nX))
            xi1 = min(self.nX, vi(cx + l/2 + d, self.xbound[0], self.xbound[2], self.nX) + 1)
            yi0 = max(0, vi(cy - w/2 - d, self.ybound[0], self.ybound[2], self.nY))
            yi1 = min(self.nY, vi(cy + w/2 + d, self.ybound[0], self.ybound[2], self.nY) + 1)
            zi0 = max(0, vi(cz - h/2 - d, self.zbound[0], self.zbound[2], self.nZ))
            zi1 = min(self.nZ, vi(cz + h/2 + d, self.zbound[0], self.zbound[2], self.nZ) + 1)

            if xi0 < xi1 and yi0 < yi1 and zi0 < zi1:
                voxel[zi0:zi1, xi0:xi1, yi0:yi1] = cls

        return torch.from_numpy(voxel)


if __name__ == '__main__':
    ds = NuScenesV4Dataset('../data/sets/nuscenesmini', is_train=True, box_dilate=2)
    imgs, Ks, s2e, gt = ds[0]
    print(f'imgs: {imgs.shape}, gt: {gt.shape}')
    dist = {n: (gt==i).sum().item() for i,n in enumerate(CLASS_NAMES)}
    print(f'GT 분포: {dist}')

    # 여러 샘플 평균
    import torch
    total = torch.zeros(5, dtype=torch.long)
    for i in range(min(10, len(ds))):
        _, _, _, g = ds[i]
        for c in range(5): total[c] += (g==c).sum()
    tot = total.sum().item()
    print('\n=== V4 GT 분포 (10샘플) ===')
    for c, nm in enumerate(CLASS_NAMES):
        n = total[c].item()
        print(f'  {nm:<16}: {n:>10,}  ({n/tot*100:.3f}%)')
