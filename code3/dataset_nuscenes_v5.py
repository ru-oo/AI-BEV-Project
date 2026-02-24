"""
dataset_nuscenes_v5.py — FastOcc 6-Camera + GT .npy 캐싱 (최적화판)
====================================================================
v4 → v5 핵심 개선:
  1. GT Voxel .npy 사전 캐싱  : 매 에포크 LiDAR I/O 제거 → 수십 배 빠름
  2. numpy 완전 벡터화        : Python for 루프 없이 포인트 처리
  3. box_dilate=0 (기본값)    : Lovász Loss와 호환, 경계 정밀도 향상
  4. pre_cache_all()           : 학습 전 일괄 캐싱으로 GPU Starvation 제거

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


class NuScenesV5Dataset(Dataset):
    """
    FastOcc v5: 6-Camera + GT .npy 캐싱 + numpy 벡터화

    [병목 해소]
    - LiDAR I/O: 학습 전 pre_cache_all()로 모든 GT를 .npy로 저장
      → 학습 중에는 np.load() 한 번만 호출 (매우 빠름)
    - numpy 벡터화: 포인트별 Python 반복문 완전 제거
    - box_dilate=0: Lovász Loss와 호환 (FP 패널티 방지)
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
                 box_dilate: int = 0,          # ★ v5 기본값 0 (Lovász 호환)
                 cache_dir: str = None):        # GT .npy 캐시 디렉토리

        assert _NS_OK, 'pip install nuscenes-devkit'
        self.nusc       = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.is_train   = is_train
        self.img_h      = img_h
        self.img_w      = img_w
        self.xbound     = xbound
        self.ybound     = ybound
        self.zbound     = zbound
        self.box_dilate = box_dilate

        self.nX = int((xbound[1] - xbound[0]) / xbound[2])
        self.nY = int((ybound[1] - ybound[0]) / ybound[2])
        self.nZ = int((zbound[1] - zbound[0]) / zbound[2])

        # GT 캐시 디렉토리 (데이터셋 루트 안에 자동 생성)
        if cache_dir is None:
            cache_dir = os.path.join(dataroot, 'gt_cache_v5')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

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

        print(f'[NuScenesV5] {"학습" if is_train else "검증"}: '
              f'{len(self.samples)}샘플 | 6카메라 | '
              f'복셀 {self.nZ}×{self.nX}×{self.nY} | {img_h}×{img_w} | '
              f'box_dilate={box_dilate} | 캐시={os.path.basename(cache_dir)}')

    # ─────────────────────────────────────────────────
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

        # ── GT: 캐시 우선 로드 ────────────────────────
        gt = self._get_gt_cached(rec)

        return imgs, Ks, sensor2ego, gt

    # ─────────────────────────────────────────────────
    def _get_gt_cached(self, rec) -> torch.Tensor:
        """캐시(.npy)가 있으면 즉시 반환, 없으면 계산 후 저장"""
        tok  = rec['token']
        path = os.path.join(self.cache_dir, f'{tok}.npy')
        if os.path.exists(path):
            return torch.from_numpy(np.load(path).astype(np.int64))
        gt = self._build_gt(rec)
        # int16로 저장 (용량 절반: int64 → int16, 값 범위 0~4는 충분)
        np.save(path, gt.numpy().astype(np.int16))
        return gt

    # ─────────────────────────────────────────────────
    def _build_gt(self, rec) -> torch.Tensor:
        """
        numpy 완전 벡터화 GT 생성 (Python for-loop over points 없음)

        파이프라인:
          1. Road : z=0 레이어 2슬라이스 슬라이스 할당 (O(1))
          2. 객체  : 바운딩박스 → voxel 범위 계산 → 슬라이스 할당
                    (박스 수 = 보통 < 50, 루프 비용 무시 가능)
          3. LiDAR: (선택) 지면 포인트 Road 정제 — numpy 벡터 연산
        """
        voxel = np.zeros((self.nZ, self.nX, self.nY), dtype=np.int64)

        # ── 1. Road: z=0 레이어 (학습 안정성 — 밀도 높은 사전 지식) ──
        zi_road = max(0, int((0.0 - self.zbound[0]) / self.zbound[2]))
        zi_road_end = min(self.nZ, zi_road + 2)  # 2 슬라이스 (0 ~ 0.5m)
        voxel[zi_road:zi_road_end, :, :] = 1     # Road

        # ── 2. LiDAR 포인트 → 지면 정제 (numpy 벡터화) ───────────────
        lidar_tok = rec['data']['LIDAR_TOP']
        lidar_sd  = self.nusc.get('sample_data', lidar_tok)
        lidar_cs  = self.nusc.get('calibrated_sensor',
                                   lidar_sd['calibrated_sensor_token'])
        pc_path   = self.nusc.get_sample_data_path(lidar_tok)

        pc      = LidarPointCloud.from_file(pc_path)      # (4, N)
        l_rot   = Quaternion(lidar_cs['rotation']).rotation_matrix.astype(np.float32)
        l_trans = np.array(lidar_cs['translation'], dtype=np.float32)
        pts     = (pc.points[:3].T @ l_rot.T) + l_trans   # (N, 3) ego frame

        # 범위 내 포인트 마스크 (벡터화)
        in_range = (
            (pts[:, 0] >= self.xbound[0]) & (pts[:, 0] < self.xbound[1]) &
            (pts[:, 1] >= self.ybound[0]) & (pts[:, 1] < self.ybound[1]) &
            (pts[:, 2] >= self.zbound[0]) & (pts[:, 2] < self.zbound[1])
        )
        pts_in = pts[in_range]   # (M, 3)

        if len(pts_in) > 0:
            # 지면 포인트 (z ∈ [-0.3, 0.8]m) → Road 정제 (벡터화)
            ground_mask = (pts_in[:, 2] >= -0.3) & (pts_in[:, 2] <= 0.8)
            g_pts = pts_in[ground_mask]  # (G, 3)

            if len(g_pts) > 0:
                # voxel 인덱스 계산 (벡터화 — 반복문 없음)
                xi_g = np.floor((g_pts[:, 0] - self.xbound[0]) / self.xbound[2]).astype(np.int32)
                yi_g = np.floor((g_pts[:, 1] - self.ybound[0]) / self.ybound[2]).astype(np.int32)
                zi_g = np.floor((g_pts[:, 2] - self.zbound[0]) / self.zbound[2]).astype(np.int32)

                # 인덱스 범위 필터 (벡터화)
                valid = (
                    (xi_g >= 0) & (xi_g < self.nX) &
                    (yi_g >= 0) & (yi_g < self.nY) &
                    (zi_g >= 0) & (zi_g < self.nZ)
                )
                # Road voxel 한 번에 할당 (벡터화)
                voxel[zi_g[valid], xi_g[valid], yi_g[valid]] = 1

        # ── 3. 바운딩박스 → 객체 클래스 (슬라이스 할당) ──────────────
        l_rot_q = Quaternion(lidar_cs['rotation'])
        l_trans_arr = np.array(lidar_cs['translation'])
        _, boxes, _ = self.nusc.get_sample_data(lidar_tok)

        # 박스를 ego frame으로 변환
        for b in boxes:
            b.rotate(l_rot_q)
            b.translate(l_trans_arr)

        for box in boxes:
            # 클래스 매핑
            cls = 0
            for pfx, cid in CLASS_MAP.items():
                if box.name.startswith(pfx):
                    cls = cid
                    break
            if cls == 0:
                continue

            cx, cy, cz = box.center
            w, l, h = box.wlh
            d = self.box_dilate * float(max(self.xbound[2],
                                            self.ybound[2],
                                            self.zbound[2]))

            # voxel 인덱스 (박스당 한 번 계산 — 반복문 불필요)
            def to_idx(v, lo, step): return int((v - lo) / step)

            xi0 = max(0, to_idx(cx - l/2 - d, self.xbound[0], self.xbound[2]))
            xi1 = min(self.nX, to_idx(cx + l/2 + d, self.xbound[0], self.xbound[2]) + 1)
            yi0 = max(0, to_idx(cy - w/2 - d, self.ybound[0], self.ybound[2]))
            yi1 = min(self.nY, to_idx(cy + w/2 + d, self.ybound[0], self.ybound[2]) + 1)
            zi0 = max(0, to_idx(cz - h/2 - d, self.zbound[0], self.zbound[2]))
            zi1 = min(self.nZ, to_idx(cz + h/2 + d, self.zbound[0], self.zbound[2]) + 1)

            if xi0 < xi1 and yi0 < yi1 and zi0 < zi1:
                voxel[zi0:zi1, xi0:xi1, yi0:yi1] = cls  # 슬라이스 할당

        return torch.from_numpy(voxel)

    # ─────────────────────────────────────────────────
    def pre_cache_all(self, force=False):
        """
        학습 시작 전 전체 GT .npy 사전 생성

        force=True: 기존 캐시 무시하고 전부 재생성
        """
        from tqdm import tqdm
        pending = []
        for rec in self.samples:
            path = os.path.join(self.cache_dir, f'{rec["token"]}.npy')
            if force or not os.path.exists(path):
                pending.append(rec)

        if not pending:
            print(f'  ✅ GT 캐시 이미 완료 ({len(self.samples)}개)')
            return

        print(f'  GT 사전 캐싱: {len(pending)}/{len(self.samples)}개 생성 중...')
        for rec in tqdm(pending, desc='GT 캐싱', leave=False):
            self._get_gt_cached(rec)
        print(f'  ✅ GT 캐싱 완료 → {self.cache_dir}')


if __name__ == '__main__':
    ds = NuScenesV5Dataset('../data/sets/nuscenesmini', is_train=True, box_dilate=0)

    # 캐시 생성 테스트
    ds.pre_cache_all()

    imgs, Ks, s2e, gt = ds[0]
    print(f'imgs: {imgs.shape}, gt: {gt.shape}')

    # 분포 확인
    total = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for i in range(min(10, len(ds))):
        _, _, _, g = ds[i]
        for c in range(NUM_CLASSES):
            total[c] += (g == c).sum()
    tot = total.sum().item()
    print('\n=== V5 GT 분포 (10샘플, box_dilate=0) ===')
    for c, nm in enumerate(CLASS_NAMES):
        n = total[c].item()
        print(f'  {nm:<16}: {n:>10,}  ({n/tot*100:.3f}%)')
