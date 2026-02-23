"""
inference_rt.py — FastOcc 실시간 추론기
========================================
학습된 FastOcc를 로드, 단일 전방 카메라 → 3D Semantic Occupancy 실시간 변환

사용:
  infer = RealTimeInference('../code3/best_fastocc_miou.pth')
  vox_cls, conf, fps = infer.infer(frame_bgr, K_tensor, s2e_tensor)
"""

import sys, time
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))
from model_fastocc import FastOcc
from dataset_nuscenes_v3 import NUM_CLASSES, CLASS_NAMES, IMG_H, IMG_W

_TF = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize([.485,.456,.406],[.229,.224,.225]),
])

MODEL_CFG = dict(
    xbound=(-25., 25., .5),
    ybound=(-25., 25., .5),
    zbound=(-2.,  6.,  .5),
    num_classes=NUM_CLASSES,
    fpn_ch=128, c2h_ch=64,
    img_h=IMG_H, img_w=IMG_W,
)


class RealTimeInference:
    def __init__(self, model_path='../best_fastocc_miou.pth',
                 device='cuda'):
        self.device  = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self._fps    = []
        self.model   = FastOcc(**MODEL_CFG).to(self.device)

        try:
            sd = torch.load(model_path, map_location=self.device,
                            weights_only=True)
            self.model.load_state_dict(sd)
            print(f'✅ FastOcc 로드: {model_path}')
        except FileNotFoundError:
            print(f'⚠️  가중치 없음 (학습 먼저 실행): {model_path}')

        self.model.eval()
        self._warmup()

    def _warmup(self):
        d = self.model
        dummy_img = torch.zeros(1,3,IMG_H,IMG_W).to(self.device)
        dummy_K   = torch.eye(3).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for _ in range(2): d(dummy_img, dummy_K)
        print('✅ Warm-up 완료')

    @torch.no_grad()
    def infer(self, frame_bgr, K: torch.Tensor,
              sensor2ego: torch.Tensor = None):
        """
        frame_bgr  : (H,W,3) BGR numpy (OpenCV)
        K          : (1,3,3) or (3,3) float tensor
        sensor2ego : (1,4,4) or (4,4) float tensor  [선택]
        """
        t0 = time.perf_counter()

        img = _TF(frame_bgr[:,:,::-1].copy()).unsqueeze(0).to(self.device)
        if K.dim() == 2: K = K.unsqueeze(0)
        K = K.float().to(self.device)
        if sensor2ego is not None:
            if sensor2ego.dim() == 2: sensor2ego = sensor2ego.unsqueeze(0)
            sensor2ego = sensor2ego.float().to(self.device)

        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            logits = self.model(img, K, sensor2ego)

        prob    = logits.softmax(1)[0]            # (C, nZ, nX, nY)
        vox_cls = prob.argmax(0).cpu().numpy()    # (nZ, nX, nY)
        conf    = prob[1:].max(0).values.max(0).values.cpu().numpy()

        fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
        self._fps.append(fps)
        if len(self._fps) > 30: self._fps.pop(0)

        return vox_cls, conf, fps

    def get_obstacle_grid(self, vox_cls, obs=(2,3,4)):
        """3D 복셀 → 2D BEV 장애물 격자 (A* 입력용)"""
        mask = np.zeros(vox_cls.shape[1:], bool)
        for c in obs: mask |= (vox_cls == c).any(0)
        return (mask * 255).astype(np.uint8)

    @property
    def avg_fps(self):
        return float(np.mean(self._fps)) if self._fps else 0.
