import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from nuscenes_dataset import NuScenesDataset
from train import LSSModel

# 클래스별 BGR 색상 (OpenCV는 BGR 순서)
CLASS_BGR = np.array([
    [25,  25,  25],    # 0: Empty      → 짙은 회색
    [255, 100, 50],    # 1: Car        → 파랑 (BGR)
    [0,   140, 255],   # 2: Truck/Bus  → 주황 (BGR)
    [50,  205, 38],    # 3: Pedestrian → 초록 (BGR)
], dtype=np.uint8)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"NVIDIA GPU 사용: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple MPS 사용")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    return device


def pred_to_semantic_bev(preds, alpha=None, prev_map=None):
    """
    preds: (1, num_classes, nz, X, Y) tensor
    반환: (X, Y) ndarray - 클래스 ID (0~3), 평활화 적용 시 float
    """
    probs = torch.softmax(preds, dim=1)           # (1, 4, nz, X, Y)
    # Z 방향으로 max-pool: 각 (X, Y) 위치에서 가장 높은 확률의 클래스 결정
    probs_2d, _ = probs[0].max(dim=1)             # (4, X, Y) - Z축 max
    probs_2d = probs_2d.cpu().numpy()             # (4, X, Y)

    if alpha is not None and prev_map is not None:
        probs_2d = alpha * probs_2d + (1 - alpha) * prev_map

    class_map = np.argmax(probs_2d, axis=0).astype(np.int64)  # (X, Y)
    return class_map, probs_2d


def class_map_to_bgr(class_map):
    """class_map: (X, Y) → BGR image (X, Y, 3)"""
    return CLASS_BGR[class_map.clip(0, len(CLASS_BGR) - 1)]


def make_video():
    device = get_device()
    output_file = "driving_demo2.mp4"
    fps = 10
    alpha = 0.4   # temporal smoothing 계수

    model = LSSModel(
        xbound=[-50, 50, 0.5],
        ybound=[-50, 50, 0.5],
        zbound=[-2.0, 6.0, 2.0],
        dbound=[4, 45, 1],
        num_classes=4
    ).to(device)

    try:
        model.load_state_dict(torch.load("best_semantic_mini_model.pth", map_location=device))
        print("모델 로드 성공")
    except Exception as e:
        print(f"모델 파일 로드 실패: {e}")
        return

    model.eval()

    dataset = NuScenesDataset(dataroot='../data/sets/nuscenesmini', is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_file, fourcc, fps, (1200, 600))

    prev_probs = None

    print("영상 생성 중 (시맨틱 클래스 색상 + 스무딩)...")

    with torch.no_grad():
        for i, (imgs, intrinsics, sensor2ego, gt_bev) in enumerate(tqdm(loader)):
            imgs = imgs.to(device)
            intrinsics = intrinsics.float().to(device)
            rots = sensor2ego[:, :, :3, :3].float().to(device)
            trans = sensor2ego[:, :, :3, 3].float().to(device)

            preds = model(imgs, rots, trans, intrinsics)

            class_map, curr_probs = pred_to_semantic_bev(preds, alpha, prev_probs)
            prev_probs = curr_probs

            # === 캔버스 그리기 ===
            canvas = np.zeros((600, 1200, 3), dtype=np.uint8)

            # 1. 전방 카메라 복원 및 좌측 배치
            front_img = imgs[0, 1].permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            front_img = np.clip((front_img * std + mean) * 255, 0, 255).astype(np.uint8)
            front_img = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)

            h, w = front_img.shape[:2]
            scale = 600 / w
            dim = (600, int(h * scale))
            front_img = cv2.resize(front_img, dim)
            canvas[150: 150 + dim[1], 0:600] = front_img

            # 2. 시맨틱 BEV 맵 (우측 배치)
            bev_bgr = class_map_to_bgr(class_map)           # (X, Y, 3)
            bev_bgr = bev_bgr.transpose(1, 0, 2)            # (Y, X, 3) for imshow
            bev_bgr = cv2.resize(bev_bgr.astype(np.uint8), (600, 600),
                                 interpolation=cv2.INTER_NEAREST)
            bev_bgr = cv2.flip(bev_bgr, -1)                 # 좌표계 보정 (앞=위, 왼=왼)
            canvas[:, 600:] = bev_bgr

            # 에고카 위치 및 범례 오버레이
            cv2.arrowedLine(canvas, (900, 300), (900, 280), (255, 255, 255), 4)
            cv2.putText(canvas, f"Frame: {i}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # 범례 텍스트 (우측 하단)
            legend = [
                ("Empty",        (60, 60, 60)),
                ("Car",          (255, 100, 50)),
                ("Truck/Bus",    (0, 140, 255)),
                ("Pedestrian",   (50, 205, 38)),
            ]
            for j, (name, color) in enumerate(legend):
                y_pos = 540 - j * 25
                cv2.rectangle(canvas, (610, y_pos - 14), (628, y_pos + 2), color, -1)
                cv2.putText(canvas, name, (633, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            out_writer.write(canvas)

    out_writer.release()
    print(f"저장 완료: {output_file}")


if __name__ == "__main__":
    make_video()
