"""
inference_rt.py — FPVNet 실시간 추론기
=======================================
학습된 FPVNet을 로드하여 단일 카메라 → 3D Semantic Occupancy 실시간 생성

LSS와의 차이:
  LSS  : 6카메라 필요, frustum pooling (복잡)
  FPVNet: 단일 전방 카메라, 기하학적 투영 (직관적)

사용:
  infer = RealTimeInference(model_path='../best_fpvnet.pth')
  voxel, fps = infer.infer(frame_bgr, fx=800, fy=800, cx=320, cy=240)
"""

import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_fpvnet import FPVNet
from dataset_nuscenes_v3 import NUM_CLASSES, CLASS_NAMES, IMG_H, IMG_W

# 이미지 전처리
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
_PREPROCESS = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    _NORMALIZE,
])


class RealTimeInference:
    """
    FPVNet 실시간 추론 래퍼

    Parameters
    ----------
    model_path : 학습된 가중치 (.pth)
    device     : 'cuda' | 'cpu'
    conf_thr   : 전경 최소 확률 (미사용 클래스 필터)
    """

    MODEL_CFG = dict(
        xbound=(-25.0, 25.0, 0.5),
        ybound=(-25.0, 25.0, 0.5),
        zbound=(-2.0,  6.0,  1.0),
        dbound=(1.0, 50.0),
        num_classes=NUM_CLASSES,
        fpn_ch=128,
    )

    def __init__(self,
                 model_path: str  = '../best_fpvnet.pth',
                 device:     str  = 'cuda',
                 conf_thr:   float = 0.3):

        self.device   = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.conf_thr = conf_thr
        self._fps_buf = []

        print(f'[RealTimeInference] 장치: {self.device}')
        self.model = FPVNet(**self.MODEL_CFG).to(self.device)

        # 가중치 로드
        try:
            state = torch.load(model_path, map_location=self.device,
                               weights_only=True)
            self.model.load_state_dict(state)
            print(f'  ✅ 모델 로드: {model_path}')
        except FileNotFoundError:
            print(f'  ⚠️  가중치 없음: {model_path} (랜덤 초기화)')

        self.model.eval()
        self._warmup()

    def _warmup(self):
        """TorchScript 첫 추론 JIT 준비"""
        dummy_img = torch.zeros(1, 3, IMG_H, IMG_W).to(self.device)
        dummy_K   = torch.eye(3).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for _ in range(2):
                self.model(dummy_img, dummy_K)
        print('  ✅ Warm-up 완료')

    def make_K(self, fx: float, fy: float,
               cx: float, cy: float) -> torch.Tensor:
        """
        카메라 내부 파라미터 → 텐서
        이미지 리사이즈 비율 자동 적용
        """
        # 원본 해상도 → IMG_W × IMG_H 리사이즈 보정은
        # dataset에서 하듯 사용자가 실제 해상도 맞게 조정 필요
        K = torch.tensor([[fx, 0., cx],
                          [0., fy, cy],
                          [0.,  0., 1.]], dtype=torch.float32)
        return K.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def infer(self,
              frame_bgr: np.ndarray,
              fx: float = 800.0,
              fy: float = 800.0,
              cx: float = IMG_W / 2,
              cy: float = IMG_H / 2,
              K:  torch.Tensor = None) -> tuple:
        """
        단일 카메라 프레임 → 3D Semantic Occupancy

        Parameters
        ----------
        frame_bgr : (H, W, 3) BGR numpy  (OpenCV 캡처 그대로)
        fx, fy    : focal length [px]  (리사이즈 후 기준)
        cx, cy    : principal point    (리사이즈 후 기준)
        K         : (1, 3, 3) 직접 제공 시 fx/fy/cx/cy 무시

        Returns
        -------
        voxel_cls : (nZ, nX, nY) int64 클래스 인덱스
        conf_map  : (nX, nY) float   BEV 최대 확률 (전경 신뢰도)
        fps       : float
        """
        t0 = time.perf_counter()

        # BGR → RGB → 전처리
        img_rgb = frame_bgr[:, :, ::-1].copy()
        img_t   = _PREPROCESS(img_rgb).unsqueeze(0).to(self.device)

        if K is None:
            K = self.make_K(fx, fy, cx, cy)

        with torch.amp.autocast('cuda',
                                enabled=(self.device.type == 'cuda')):
            vox_logits, depth, _ = self.model(img_t, K)
            # vox_logits: (1, C, nZ, nX, nY)

        vox_prob = vox_logits.softmax(dim=1)[0]          # (C, nZ, nX, nY)
        vox_cls  = vox_prob.argmax(dim=0).cpu()          # (nZ, nX, nY)

        # BEV 전경 신뢰도 (z 최대 풀링)
        fg_prob  = vox_prob[1:].max(dim=0).values.max(dim=0).values
        conf_map = fg_prob.cpu()                          # (nX, nY)

        fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
        self._fps_buf.append(fps)
        if len(self._fps_buf) > 30:
            self._fps_buf.pop(0)

        return vox_cls.numpy(), conf_map.numpy(), fps

    def get_obstacle_grid(self,
                          vox_cls: np.ndarray,
                          obs_classes=(2, 3, 4)) -> np.ndarray:
        """
        3D 복셀 → 2D BEV 장애물 격자 (A* 입력용)

        Parameters
        ----------
        vox_cls    : (nZ, nX, nY) int
        obs_classes: 장애물 클래스 ID tuple

        Returns
        -------
        grid : (nX, nY) uint8  0=자유 / 255=장애물
        """
        obs_mask = np.zeros(vox_cls.shape[1:], dtype=bool)
        for c in obs_classes:
            obs_mask |= (vox_cls == c).any(axis=0)
        return (obs_mask * 255).astype(np.uint8)

    @property
    def avg_fps(self) -> float:
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0
