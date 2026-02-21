"""
main_demo.py — FPVNet 자율주행 실시간 데모
==========================================
파이프라인:
  웹캠/영상 → FPVNet 추론 → 3D 점유 맵
      → BEV 장애물 격자 → A* 경로 계획
      → 조향/속도 계산 → STM32 CAN 전송

실행:
  python code3/autonomous/main_demo.py [--source 0] [--can COM3] [--no-can]

  --source : 카메라 인덱스(0) 또는 영상 파일 경로
  --can    : CAN 포트 (기본 COM3 / Linux: can0)
  --no-can : CAN 없이 시뮬레이션만

설정:
  CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
  → 실제 카메라 캘리브레이션 값으로 교체하세요.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'autonomous'))

from autonomous.inference_rt import RealTimeInference, IMG_H, IMG_W, CLASS_NAMES
from autonomous.path_planner import AStarPlanner
from autonomous.can_interface import STM32CANInterface
from dataset_nuscenes_v3 import NUM_CLASSES

# ══ 카메라 파라미터 (실측값으로 교체) ══
CAMERA_FX = 800.0
CAMERA_FY = 800.0
CAMERA_CX = IMG_W / 2.0
CAMERA_CY = IMG_H / 2.0

# 색상 팔레트 (클래스별)
CMAP_BGR = {
    0: (20,  20,  20),    # Free
    1: (100, 100, 100),   # Road
    2: (255, 100,   0),   # Vehicle
    3: (0,   50,  255),   # Pedestrian
    4: (0,   220, 220),   # StaticObst
}


def bev_to_bgr(vox_cls: np.ndarray) -> np.ndarray:
    """(nZ, nX, nY) → BEV RGB (z-max projection)"""
    bev   = vox_cls.max(axis=0)  # (nX, nY)
    rgb   = np.zeros((*bev.shape, 3), dtype=np.uint8)
    for cid, color in CMAP_BGR.items():
        rgb[bev == cid] = color
    return rgb


def draw_hud(frame: np.ndarray,
             steering: float, speed: float,
             fps: float, avg_fps: float,
             mode_str: str) -> np.ndarray:
    """카메라 프레임에 HUD 오버레이"""
    h, w = frame.shape[:2]
    cv2.putText(frame, f'FPS: {fps:.0f} (avg {avg_fps:.0f})',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f'Steer: {steering:+.3f}  Speed: {speed:.3f}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Mode: {mode_str}',
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,165,0), 2)

    # 조향 바
    bar_cx = w // 2
    bar_y  = h - 30
    bar_w  = int(w * 0.4)
    cv2.rectangle(frame, (bar_cx-bar_w//2, bar_y-8),
                  (bar_cx+bar_w//2, bar_y+8), (50,50,50), -1)
    offset = int(steering * bar_w // 2)
    cv2.circle(frame, (bar_cx + offset, bar_y), 8, (0,200,255), -1)
    return frame


def main():
    parser = argparse.ArgumentParser(description='FPVNet 자율주행 데모')
    parser.add_argument('--source',  default='0',
                        help='카메라 인덱스 또는 영상 파일 (기본: 0)')
    parser.add_argument('--model',   default=str(ROOT / 'best_fpvnet.pth'))
    parser.add_argument('--can',     default='COM3')
    parser.add_argument('--no-can',  action='store_true')
    parser.add_argument('--goal-m',  type=float, default=8.0,
                        help='A* 전방 목표 거리 [m]')
    args = parser.parse_args()

    # ── 추론기 ──────────────────────────────
    print('\n[1/3] FPVNet 추론기 초기화')
    infer = RealTimeInference(model_path=args.model)

    # ── 경로 계획기 ─────────────────────────
    print('\n[2/3] A* 경로 계획기 초기화')
    nX = int((25*2) / 0.5)   # xbound (-25, 25, 0.5) → 100
    nY = int((25*2) / 0.5)
    planner = AStarPlanner(grid_size=nX, resolution=0.5, lookahead=15)

    # ── CAN 인터페이스 ──────────────────────
    print('\n[3/3] STM32 CAN 인터페이스')
    if args.no_can:
        can_if = None
        print('  시뮬레이션 모드 (--no-can)')
    else:
        can_if = STM32CANInterface(channel=args.can, verbose=False)

    # ── 카메라/영상 열기 ─────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'❌ 소스 열기 실패: {args.source}')
        sys.exit(1)

    print(f'\n▶  데모 시작 (q=종료, s=STOP, a=AUTO)')
    mode = 1  # AUTO

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # ── 1. FPVNet 추론 ──────────────────
            vox_cls, conf_bev, fps = infer.infer(
                frame,
                fx=CAMERA_FX, fy=CAMERA_FY,
                cx=CAMERA_CX, cy=CAMERA_CY)

            # ── 2. 장애물 격자 ──────────────────
            obs_grid = infer.get_obstacle_grid(vox_cls)

            # ── 3. A* 경로 계획 ─────────────────
            path = planner.plan(obs_grid, goal_ahead_m=args.goal_m)
            steering, speed = planner.path_to_command(path)
            if mode == 0:
                steering, speed = 0.0, 0.0

            # ── 4. CAN 전송 ─────────────────────
            if can_if is not None:
                can_if.send_control(steering, speed, mode=mode)
            else:
                mode_str = {0:'STOP', 1:'AUTO'}.get(mode, '?')
                print(f'[SIM] steer={steering:+.3f} speed={speed:.3f} '
                      f'mode={mode_str}')

            # ── 5. 시각화 ───────────────────────
            # BEV 뷰 (장애물 격자 + 경로)
            bev_img = bev_to_bgr(vox_cls)
            bev_img = cv2.resize(bev_img,
                                 (bev_img.shape[1]*3, bev_img.shape[0]*3),
                                 interpolation=cv2.INTER_NEAREST)
            bev_bgr = cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR)
            bev_bgr = planner.draw_path(bev_bgr, path, scale=3)

            # 카메라 뷰 리사이즈 + HUD
            cam_view = cv2.resize(frame, (IMG_W * 2, IMG_H * 2))
            mode_str = {0:'STOP', 1:'AUTO'}.get(mode, '?')
            draw_hud(cam_view, steering, speed,
                     fps, infer.avg_fps, mode_str)

            # 레이아웃: 상단=카메라, 하단=BEV
            bev_bgr_rs = cv2.resize(
                bev_bgr,
                (cam_view.shape[1], cam_view.shape[1]),  # 정사각형 BEV
                interpolation=cv2.INTER_NEAREST)
            combined = np.vstack([cam_view,
                                  bev_bgr_rs[:cam_view.shape[1]//2]])
            cv2.imshow('FPVNet Autonomous Driving', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                mode = 0
                if can_if:
                    can_if.send_stop()
            elif key == ord('a'):
                mode = 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if can_if is not None:
            can_if.close()
        print('\n데모 종료')


if __name__ == '__main__':
    main()
