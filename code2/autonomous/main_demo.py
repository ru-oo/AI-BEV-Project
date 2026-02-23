"""
main_demo.py - 자율주행 통합 데모
====================================
Pipeline:
  카메라/비디오 → [PC] 3D Occupancy 추론 → BEV 변환
  → A* 경로계획 → 조향·속도 명령 → CAN → STM32 → 모터

실행 방법:
  # 웹캠 (6개 동일 카메라 시뮬레이션)
  python autonomous/main_demo.py --source camera

  # 비디오 파일
  python autonomous/main_demo.py --source video --video path/to/video.mp4

  # NuScenes 미니 시각화 (GPU 추론 테스트)
  python autonomous/main_demo.py --source nuscenes

  # CAN 포트 지정
  python autonomous/main_demo.py --can COM3

STM32 CAN 메시지 포맷 (ID 0x100):
  Byte 0-1 : steering  int16  ×0.001 = -1.0~1.0
  Byte 2-3 : speed     uint16 ×0.001 =  0.0~1.0
  Byte 4   : mode      0=STOP 1=AUTO 2=MANUAL
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from autonomous.inference_rt  import RealTimeInference
from autonomous.bev_processing import BEVProcessor
from autonomous.path_planner  import AStarPlanner
from autonomous.can_interface import STM32CANInterface


# ── CAN 전송 주기 (모델 추론 FPS에 맞춤) ──
CAN_HZ = 20   # 최대 전송 빈도 [Hz]


def parse_args():
    p = argparse.ArgumentParser(description="3D Occupancy 자율주행 데모")
    p.add_argument("--source",  choices=["camera","video","nuscenes"],
                   default="camera")
    p.add_argument("--video",   default="", help="비디오 파일 경로")
    p.add_argument("--model",   default="../best_v2_model.pth")
    p.add_argument("--can",     default="SIM",
                   help="CAN 포트 (COM3, can0, SIM=시뮬)")
    p.add_argument("--goal-m",  type=float, default=8.0,
                   help="A* 목표 거리 [m]")
    p.add_argument("--save",    action="store_true",
                   help="결과 영상 저장")
    p.add_argument("--no-gui",  action="store_true")
    return p.parse_args()


def build_frame_generator(args):
    """소스에서 프레임 생성기 반환"""
    if args.source == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  웹캠 없음 → 가상 프레임 사용")
            while True:
                yield [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)] * 6
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield [frame] * 6    # 단안 카메라 6개 동일 사용 (테스트용)
            cap.release()

    elif args.source == "video":
        cap = cv2.VideoCapture(args.video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield [frame] * 6
        cap.release()

    elif args.source == "nuscenes":
        # NuScenes 미니 데이터셋에서 실제 6-카메라 프레임 로드
        try:
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes('v1.0-mini',
                            dataroot=str(ROOT.parent / 'data/sets/nuscenesmini'),
                            verbose=False)
            cam_names = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                         'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']

            for sample in nusc.sample:
                imgs = []
                for cam in cam_names:
                    tok  = sample['data'][cam]
                    path = nusc.get_sample_data_path(tok)
                    imgs.append(cv2.imread(path))
                yield imgs
        except Exception as e:
            print(f"NuScenes 로드 실패: {e} → 가상 프레임 사용")
            while True:
                yield [np.zeros((900, 1600, 3), np.uint8)] * 6


def overlay_hud(img, fps, steer, speed, mode, miou_info=""):
    """화면 HUD 오버레이"""
    h, w = img.shape[:2]
    # 배경
    cv2.rectangle(img, (0, h-80), (w, h), (30,30,30), -1)

    info = (f"FPS:{fps:5.1f}  |  Steer:{steer:+.3f}  "
            f"Speed:{speed:.3f}  |  Mode:{mode}  {miou_info}")
    cv2.putText(img, info, (10, h-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,200), 1)

    # 조향 막대
    cx = w // 2
    bar_len = int(steer * 100)
    cv2.rectangle(img, (cx-100, h-15), (cx+100, h-5), (60,60,60), -1)
    color = (0,255,0) if abs(steer) < 0.3 else (0,200,255)
    cv2.rectangle(img, (cx, h-15), (cx+bar_len, h-5), color, -1)
    return img


def main():
    args = parse_args()
    print("\n=== 3D Semantic Occupancy 자율주행 데모 ===")
    print(f"  소스: {args.source}  |  CAN: {args.can}")

    # ── 컴포넌트 초기화 ──
    infer   = RealTimeInference(model_path=args.model)
    bev_proc = BEVProcessor(inflate_r=2)
    planner  = AStarPlanner(lookahead=12)

    sim_mode = (args.can.upper() == "SIM")
    can_if   = STM32CANInterface(
        channel=args.can if not sim_mode else 'COM99',
        verbose=False
    )

    writer = None
    frame_gen = build_frame_generator(args)

    last_can_t = 0.0
    can_interval = 1.0 / CAN_HZ
    steer, speed, mode = 0.0, 0.0, 0

    print("\n[루프 시작] q 키: 종료 / s 키: STOP / a 키: AUTO\n")

    try:
        for camera_imgs in frame_gen:
            loop_t = time.time()

            # ① 추론
            pred_logits, fps = infer.infer(camera_imgs)

            # ② BEV 변환
            obs_grid = bev_proc.occupancy_to_2d(pred_logits)

            # ③ A* 경로 계획
            path = planner.plan(obs_grid, goal_ahead_m=args.goal_m)

            # ④ 명령 생성
            if path:
                steer, speed = planner.path_to_command(path)
                mode = 1  # AUTO
            else:
                steer, speed, mode = 0.0, 0.0, 0   # 경로 없으면 STOP

            # ⑤ CAN 전송 (주기 제한)
            now = time.time()
            if now - last_can_t >= can_interval:
                can_if.send_control(steer, speed, mode)
                last_can_t = now

            # ⑥ 시각화
            if not args.no_gui or args.save:
                bev_img = bev_proc.semantic_bev_image(pred_logits, path)
                bev_img = planner.draw_path(bev_img, path, scale=2)
                bev_img = overlay_hud(bev_img, fps, steer, speed,
                                      {0:"STOP",1:"AUTO",2:"MAN"}.get(mode,"?"),
                                      f"avgFPS:{infer.avg_fps:.1f}")

                # 카메라 앞면 이미지 (썸네일)
                front = cv2.resize(camera_imgs[0], (400, 150))
                bev_img[:150, :400] = front

                if args.save:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(
                            "results_v2/demo_output.mp4", fourcc, 10,
                            (bev_img.shape[1], bev_img.shape[0]))
                    writer.write(bev_img)

                if not args.no_gui:
                    cv2.imshow("Autonomous BEV Demo", bev_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        mode = 0
                        can_if.send_stop()
                    elif key == ord('a'):
                        mode = 1

            elapsed = time.time() - loop_t
            print(f"\r  FPS:{fps:5.1f}  steer:{steer:+.3f}  "
                  f"speed:{speed:.3f}  path:{len(path):4d}pts", end="")

    except KeyboardInterrupt:
        print("\n사용자 중단")

    finally:
        print("\n종료 처리...")
        can_if.send_stop()
        can_if.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ 데모 종료")


if __name__ == "__main__":
    main()
