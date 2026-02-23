"""
inference_rt.py - 실시간 3D Semantic Occupancy 추론기
======================================================
학습된 LSSModelV2를 로드하여 카메라 이미지 → 3D 점유 맵 실시간 변환
NuScenes 캘리브레이션 또는 커스텀 카메라 파라미터 지원
"""

import time
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
import sys

# 부모 디렉터리에 있는 model_v2, splat 접근
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_v2 import LSSModelV2


# ── 이미지 전처리 (NuScenes 학습 시와 동일) ──
_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

_PREPROCESS = T.Compose([
    T.ToPILImage(),
    T.Resize((384, 1056)),
    T.ToTensor(),
    _NORMALIZE,
])


class RealTimeInference:
    """
    실시간 추론 래퍼

    Parameters
    ----------
    model_path     : 학습된 가중치 경로 (.pth)
    device         : 'cuda' or 'cpu'
    conf_threshold : 전경 클래스 최소 확률 (0~1)
    """

    MODEL_CFG = dict(
        xbound=[-50, 50, 0.5],
        ybound=[-50, 50, 0.5],
        zbound=[-2.0, 6.0, 2.0],
        dbound=[4, 45, 1],
        num_classes=4,
        C=64,
    )

    def __init__(self,
                 model_path: str = "../best_v2_model.pth",
                 device: str = 'cuda',
                 conf_threshold: float = 0.3):

        self.device         = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold
        self._fps_buf       = []

        print(f"[RealTimeInference] 장치: {self.device}")
        self.model = LSSModelV2(**self.MODEL_CFG).to(self.device)

        try:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"  ✅ 모델 로드: {model_path}")
        except FileNotFoundError:
            print(f"  ⚠️  가중치 없음: {model_path} (랜덤 초기화)")

        self.model.eval()
        # TorchScript warm-up (첫 추론 지연 최소화)
        self._warmup()

    # ── Warm-up ──────────────────────────────
    def _warmup(self):
        B, N = 1, 6
        dummy_imgs = torch.zeros(B, N, 3, 384, 1056).to(self.device)
        dummy_rots = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).to(self.device)
        dummy_trans = torch.zeros(B, N, 3).to(self.device)
        dummy_intr  = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for _ in range(2):
                _ = self.model(dummy_imgs, dummy_rots, dummy_trans, dummy_intr)
        print("  ✅ Warm-up 완료")

    # ── 전처리 ───────────────────────────────
    def preprocess_images(self, images: list) -> torch.Tensor:
        """
        Parameters
        ----------
        images : list of N numpy arrays (H, W, 3) BGR or RGB

        Returns
        -------
        tensor : (1, N, 3, 384, 1056)
        """
        tensors = []
        for img in images:
            if img.ndim == 2:
                img = np.stack([img]*3, axis=2)
            if img.shape[2] == 3 and img.dtype == np.uint8:
                # BGR → RGB
                img = img[:, :, ::-1].copy()
            tensors.append(_PREPROCESS(img))
        return torch.stack(tensors).unsqueeze(0).to(self.device)  # (1,N,3,H,W)

    def make_calibration(self,
                         fx: float = 1260.0, fy: float = 1260.0,
                         cx: float = 528.0,  cy: float = 192.0,
                         num_cameras: int = 6) -> tuple:
        """
        간단한 캘리브레이션 (단안/단일 카메라 반복)
        실제 사용 시 각 카메라별 캘리브레이션으로 교체하세요.

        Returns
        -------
        (rots, trans, intrinsics) 각각 (1, N, 3, 3), (1, N, 3), (1, N, 3, 3)
        """
        K = torch.tensor([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=torch.float32)

        # 카메라를 전방 1개로 단순화 (실제 6-카메라 시스템에선 교체)
        rot   = torch.eye(3, dtype=torch.float32)
        trans = torch.zeros(3, dtype=torch.float32)

        rots      = rot.unsqueeze(0).unsqueeze(0).repeat(1, num_cameras, 1, 1)
        trans_b   = trans.unsqueeze(0).unsqueeze(0).repeat(1, num_cameras, 1)
        intrinsics = K.unsqueeze(0).unsqueeze(0).repeat(1, num_cameras, 1, 1)

        return (rots.to(self.device),
                trans_b.to(self.device),
                intrinsics.to(self.device))

    # ── 추론 ─────────────────────────────────
    @torch.no_grad()
    def infer(self,
              images: list,
              rots:       torch.Tensor = None,
              trans:      torch.Tensor = None,
              intrinsics: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        images     : list of N BGR numpy arrays
        rots       : (1, N, 3, 3) camera rotation   [선택]
        trans      : (1, N, 3)    camera translation [선택]
        intrinsics : (1, N, 3, 3) camera intrinsics  [선택]

        Returns
        -------
        pred_logits : (1, 4, nz, nx, ny) on CPU
        fps         : 추론 FPS
        """
        t0 = time.time()

        imgs = self.preprocess_images(images)

        if rots is None:
            rots, trans, intrinsics = self.make_calibration(
                num_cameras=len(images))

        with torch.amp.autocast('cuda'):
            out = self.model(imgs, rots, trans, intrinsics)

        fps = 1.0 / (time.time() - t0)
        self._fps_buf.append(fps)
        if len(self._fps_buf) > 30:
            self._fps_buf.pop(0)

        return out.cpu(), fps

    @property
    def avg_fps(self) -> float:
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0
