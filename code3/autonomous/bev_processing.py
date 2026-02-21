"""
bev_processing.py - 3D Occupancy → 2D 장애물 격자 변환
==========================================================
모델 출력 (B, 4classes, nz, nx, ny) → 2D BEV 장애물 맵 (nx, ny)
경로계획에 필요한 자유공간 / 장애물 격자 생성
"""

import numpy as np
import cv2
import torch


CLASS_COLORS = {
    0: (80,  80,  80),   # Empty  - 회색
    1: (70, 130, 180),   # Car    - 파랑
    2: (255, 160,  50),  # Truck  - 주황
    3: (50,  200,  80),  # Ped    - 초록
}
CLASS_NAMES = ["Empty", "Car", "Truck/Bus", "Pedestrian"]


class BEVProcessor:
    """
    3D Semantic Occupancy → 2D 장애물 격자 변환기

    Parameters
    ----------
    grid_size   : BEV 격자 크기 (기본 200×200)
    resolution  : 격자 셀당 실제 거리 [m] (기본 0.5m)
    obstacle_classes : 장애물로 간주할 클래스 인덱스 (기본: Car·Truck·Ped)
    inflate_r   : 장애물 팽창 반경 [cells] (안전 여유)
    """

    def __init__(self,
                 grid_size: int = 200,
                 resolution: float = 0.5,
                 obstacle_classes=(1, 2, 3),
                 inflate_r: int = 2):
        self.grid_size        = grid_size
        self.resolution       = resolution
        self.obstacle_classes = obstacle_classes
        self.inflate_r        = inflate_r
        self.ego_row          = grid_size // 2
        self.ego_col          = grid_size // 2

    # ──────────────────────────────────────────
    def occupancy_to_2d(self,
                        pred_logits: torch.Tensor,
                        smooth: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        pred_logits : (B, num_classes, nz, nx, ny) - 모델 raw 출력
        Returns
        -------
        grid2d : (nx, ny) uint8  0=자유 / 255=장애물  (첫 번째 배치)
        """
        # Argmax → 클래스 레이블 (B, nz, nx, ny)
        pred_cls = pred_logits[0].argmax(dim=0).cpu().numpy()   # (nz, nx, ny)

        # Z-축 max-pool: 한 층이라도 장애물이면 장애물
        is_obstacle = np.zeros((pred_cls.shape[1], pred_cls.shape[2]), dtype=np.uint8)
        for c in self.obstacle_classes:
            is_obstacle |= (pred_cls == c).any(axis=0).astype(np.uint8)

        grid2d = (is_obstacle * 255).astype(np.uint8)

        if smooth:
            # 잡음 제거: erosion → dilation
            k = np.ones((3, 3), np.uint8)
            grid2d = cv2.morphologyEx(grid2d, cv2.MORPH_OPEN, k)

        # 장애물 팽창 (안전 마진)
        if self.inflate_r > 0:
            r = self.inflate_r
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            grid2d = cv2.dilate(grid2d, k)

        return grid2d

    # ──────────────────────────────────────────
    def semantic_bev_image(self,
                           pred_logits: torch.Tensor,
                           path: list = None,
                           ego_mark: bool = True) -> np.ndarray:
        """
        컬러 BEV 시각화 이미지 생성 (BGR, 400×400)

        Parameters
        ----------
        pred_logits : (B, num_classes, nz, nx, ny)
        path        : [(row,col), ...] A* 경로 (선택)
        Returns
        -------
        img : (400, 400, 3) uint8 BGR
        """
        pred_cls = pred_logits[0].argmax(dim=0).cpu().numpy()     # (nz, nx, ny)
        top_cls  = np.zeros((pred_cls.shape[1], pred_cls.shape[2]), dtype=np.int32)

        # 우선순위: Ped > Car > Truck > Empty
        for c in [0, 2, 1, 3]:
            mask = (pred_cls == c).any(axis=0)
            top_cls[mask] = c

        H, W = top_cls.shape
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for c, color in CLASS_COLORS.items():
            img[top_cls == c] = color[::-1]   # RGB→BGR

        # 경로 그리기
        if path:
            for (r, c2) in path:
                if 0 <= r < H and 0 <= c2 < W:
                    cv2.circle(img, (c2, r), 2, (0, 255, 255), -1)

        # 자아 차량 마크
        if ego_mark:
            er, ec = self.ego_row, self.ego_col
            cv2.circle(img, (ec, er), 5, (255, 255, 255), -1)
            cv2.circle(img, (ec, er), 5, (0, 0, 0), 1)

        # 업스케일 (표시용)
        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)

        # 범례
        for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS.values())):
            cv2.rectangle(img, (5, 5 + i*22), (20, 20 + i*22), color[::-1], -1)
            cv2.putText(img, name, (25, 18 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        return img
