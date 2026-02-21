# 3D Semantic Occupancy 자율주행 시스템

**PC 기반 실시간 3D Semantic Occupancy 생성 → 장애물 회피 경로 계획 → STM32 CAN 통신 → 자율주행**

단일 전방 카메라 이미지만으로 3D 의미론적 점유 공간을 예측하고,
A* 경로계획 결과를 STM32 보드에 CAN 통신으로 전달하는 완전한 자율주행 파이프라인입니다.

---

## 버전 이력

| 버전 | 디렉터리 | 모델 | 특징 |
|------|----------|------|------|
| v1 | `code2/` | LSS v1 (ResNet-18) | 초기 구현 |
| v2 | `code2/` | LSS v2 (EfficientNet-B0) | SE-Attention, Dynamic FocalLoss |
| **v3** | `code3/` | **FastOcc** (EfficientNet-B2) | **LSS 탈피 — 기하학적 복셀 샘플링 + Channel-to-Height** |

---

## 시스템 아키텍처 (v3 — FPVNet)

```
┌─────────────────────────────────────────────────────────────────┐
│                         PC (GPU 추론)                            │
│                                                                 │
│  1× 전방 카메라 → [EfficientNet-B2 + FPN]                       │
│                       │                                         │
│              ┌─────────┴──────────┐                             │
│         Depth Head           Sem Head                           │
│       (metric depth)    (2D semantic)                           │
│              │                   │                              │
│              └────── 기하학적 투영 ┘  ← 카메라 내부 파라미터 K     │
│                          │                                      │
│                   3D Voxel Grid                                 │
│              (nZ×100×100 @ 0.5m 해상도)                         │
│                          │                                      │
│                   [3D Refine CNN]                               │
│                          │                                      │
│             3D Semantic Occupancy Map                           │
│             (Free/Road/Vehicle/Ped/Obstacle)                    │
│                          │                                      │
│                   [A* Planner]                                  │
│                  경로 → 조향/속도                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │ CAN Bus (500kbps)
                      │ ID 0x100: [steer|speed|mode]
                 ┌────▼────┐
                 │  STM32  │ → 모터 드라이버 → 바퀴
                 └─────────┘
```

---

## FastOcc vs LSS 비교

| 항목 | LSS (v2) | **FastOcc (v3)** |
|------|----------|-----------------|
| 3D 투영 방식 | 학습된 D-bin 깊이 분포 → frustum voxel pooling | **복셀 중심 기하학적 투영 → grid_sample (깊이 학습 불필요)** |
| 3D 처리 | 3D voxel pool + 3D conv | **Channel-to-Height (C2H): 2D conv만으로 3D 표현** |
| 카메라 수 | 6 (멀티카메라) | **1 (단일 전방 카메라)** |
| 백본 | EfficientNet-B0 | **EfficientNet-B2 + FPN** |
| 핵심 연산 | splat (frustum pooling) | **grid_sample + C2H reshape** |
| VRAM | 높음 (frustum volume) | **낮음 (2D 연산 중심)** |
| 이미지 해상도 | 1056×384 | **400×224 (3.5배 절약)** |
| 클래스 | 4 | **5 (Road 추가)** |

---

## 모델 구조 (FastOcc v3)

```
1× 전방 카메라 (400×224)
       │
       ▼
[EfficientNet-B2] stage3→4→5
       │
       ▼
[FPN Neck] P3/P4/P5 → 128ch
       │
       ▼
[Voxel Query Sampler]  ★ LSS 아님 ★
  복셀 중심(x,y,z) → K로 이미지 투영
  → bilinear grid_sample (깊이 분포 없음)
  → (B, 128, nZ, nX, nY)
       │
       ▼
[Channel-to-Height (C2H) Refiner]
  (B, 128*nZ, nX, nY) → depthwise 2D conv
  → pointwise → (B, 64*nZ, nX, nY)
  → reshape → (B, 64, nZ, nX, nY)
       │
       ▼
[3D Classifier] 3D conv × 2
       │
       ▼
(B, 5class, 16, 100, 100)
```

---

## 파일 구조

```
code3/
├── model_fpvnet.py          # FPVNet 모델 (EfficientNet-B2 + FPN + GeomProj)
├── dataset_nuscenes_v3.py   # nuScenes 단일 전방 카메라 로더 (5클래스)
├── train_fpvnet.py          # 메모리 효율 학습 + 자동 git push
├── results_v3/              # 학습 결과 (자동 업데이트)
│   ├── bev_epoch???.jpg     # 에폭별 BEV 시각화 (GT vs Pred)
│   ├── loss_curve_v3.png    # 학습 곡선
│   ├── train_log_v3.csv     # 에폭별 로그
│   └── train_info_v3.json   # 최종 메트릭
└── autonomous/
    ├── inference_rt.py      # 실시간 추론 래퍼 (FPVNet)
    ├── path_planner.py      # A* 경로계획 (조향/속도 변환)
    ├── can_interface.py     # STM32 CAN 통신 (python-can)
    ├── bev_processing.py    # BEV 후처리 유틸
    └── main_demo.py         # 통합 데모 메인 루프

code2/                       # LSS v1/v2 (이전 버전, 참조용)
```

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install torch torchvision nuscenes-devkit pyquaternion
pip install python-can opencv-python matplotlib tqdm
```

### 2. 학습 (v3 — FPVNet)

```bash
cd code3
python train_fpvnet.py
# 5 epoch마다 mIoU 평가 + BEV JPG 저장 + 자동 git push
# mIoU 50% 달성 시 즉시 push
```

### 3. 자율주행 실시간 데모

```bash
# 웹캠 + 시뮬레이션 모드 (CAN 없이)
python code3/autonomous/main_demo.py --source 0 --no-can

# 웹캠 + 실제 STM32 CAN (COM3)
python code3/autonomous/main_demo.py --source 0 --can COM3

# 영상 파일
python code3/autonomous/main_demo.py --source drive.mp4 --no-can
```

### 4. CAN 메시지 포맷 (STM32)

```c
// ID 0x100 | 8 bytes | 제어 명령 (PC → STM32)
// Byte 0-1: steering  int16   ×0.001 = -1.0(좌) ~ +1.0(우)
// Byte 2-3: speed     uint16  ×0.001 =  0.0     ~ +1.0
// Byte 4  : mode      0=STOP, 1=AUTO, 2=MANUAL
// Byte 5-7: reserved

// ID 0x101 | 2 bytes | 상태 (STM32 → PC)
// Byte 0: ready flag
// Byte 1: error code
```

---

## 학습 설정 (v3)

| 파라미터 | 값 |
|----------|-----|
| 백본 | EfficientNet-B2 (ImageNet pretrained) |
| FPN 채널 | 128ch |
| 이미지 크기 | 400×224 (메모리 절약) |
| 복셀 범위 | x/y: ±25m @ 0.5m, z: -2~6m @ 1m |
| Batch Size | 2 (Gradient Accum 4 → effective 8) |
| Max LR | 2×10⁻⁴ |
| LR 스케줄러 | CosineAnnealingWarmRestarts (T₀=20) |
| 손실 | 3중 손실 (3D FocalLoss + 2D Sem + Depth 정규화) |
| 클래스 가중치 | Free=1, Road=3, Vehicle=10, Ped=15, Static=6 |
| Early Stopping | patience=30 |
| Max Epochs | 200 |

---

## 결과 (v3 — FPVNet, 학습 중)

> 클래스: Free / Road / Vehicle / Pedestrian / StaticObstacle
> 목표: 전경 mIoU ≥ 50%

![Loss Curve v3](code3/results_v3/loss_curve_v3.png)

![BEV Final](code3/results_v3/bev_final.jpg)

---

## 결과 (v2 — LSS EfficientNet-B0, 참조)

| 클래스 | IoU |
|--------|-----|
| Empty | 29.32% |
| Car | 2.13% |
| Truck/Bus | 1.31% |
| Pedestrian | 1.00% |
| 전경 mIoU | 1.48% |

![Loss Curve v2](code2/results_v2/loss_curve_v2.png)

---

## 클래스 색상 범례 (v3)

| 색상 | 클래스 |
|------|--------|
| 검정 | Free (빈 공간) |
| 회색 | Road (도로) |
| 파랑 | Vehicle (차량) |
| 빨강 | Pedestrian (보행자) |
| 노랑 | StaticObstacle (고정 장애물) |

---

## 참고 문헌

- [Lift, Splat, Shoot (ECCV 2020)](https://arxiv.org/abs/2008.05711) — v2 기반
- [EfficientNet (ICML 2019)](https://arxiv.org/abs/1905.11946)
- [Feature Pyramid Networks (CVPR 2017)](https://arxiv.org/abs/1612.03144)
- [NuScenes Dataset](https://www.nuscenes.org/)
- [python-can](https://python-can.readthedocs.io/) — STM32 CAN 통신
