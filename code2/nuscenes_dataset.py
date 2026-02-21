import torch
import os
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix, box_in_image, view_points
from pyquaternion import Quaternion
from torchvision import transforms

class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, version='v1.0-mini', is_train=True):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.is_train = is_train
        self.samples = [samp for samp in self.nusc.sample]
        
        self.cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        # 이미지 전처리: 학습 시 Augmentation 적용, 평가 시 정규화만
        _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_train:
            self.normalize_img = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                _normalize,
            ])
        else:
            self.normalize_img = transforms.Compose([
                transforms.ToTensor(),
                _normalize,
            ])

        self.xbound = [-50.0, 50.0, 0.5]
        self.ybound = [-50.0, 50.0, 0.5]
        self.zbound = [-2.0, 6.0, 2.0] # 4개 층

        # === [심화] 클래스 정의 ===
        # 0: Empty, 1: Car, 2: Truck/Bus, 3: Pedestrian/Bike
        self.class_map = {
            'vehicle.car': 1,
            'vehicle.emergency': 1,
            'vehicle.truck': 2,
            'vehicle.bus': 2,
            'vehicle.construction': 2,
            'human.pedestrian': 3,
            'vehicle.bicycle': 3,
            'vehicle.motorcycle': 3,
        }
        self.num_classes = 4 # 0, 1, 2, 3

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_record = self.samples[index]
        
        imgs_list = []
        intrinsics_list = []
        sensor2ego_list = []

        # 1. 카메라 데이터 로드
        for cam_name in self.cams:
            cam_token = sample_record['data'][cam_name]
            cam_record = self.nusc.get('sample_data', cam_token)
            cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
            
            cam_path = self.nusc.get_sample_data_path(cam_token)
            img = Image.open(cam_path)
            # [수정] 해상도 1.5배 증가 (384, 1056)
            # 기존: new_w, new_h = 704, 256
            new_w, new_h = 1056, 384
            img = img.resize((new_w, new_h))
            img_tensor = self.normalize_img(img)
            
            intrinsics = np.array(cs_record['camera_intrinsic'])
            scale_x = new_w / 1600
            scale_y = new_h / 900
            intrinsics[0] *= scale_x
            intrinsics[1] *= scale_y
            
            rot = Quaternion(cs_record['rotation']).rotation_matrix
            trans = np.array(cs_record['translation'])
            
            s2e = np.eye(4)
            s2e[:3, :3] = rot
            s2e[:3, 3] = trans
            
            imgs_list.append(img_tensor)
            intrinsics_list.append(torch.from_numpy(intrinsics))
            sensor2ego_list.append(torch.from_numpy(s2e))

        imgs = torch.stack(imgs_list)
        intrinsics = torch.stack(intrinsics_list)
        sensor2ego = torch.stack(sensor2ego_list)

        # 2. Semantic GT 생성 (박스 기반)
        # ★ 핵심 버그 수정 ★
        # get_sample_data()는 기본적으로 LIDAR 센서 좌표계로 박스를 반환합니다.
        # 모델은 Ego 좌표계에서 예측하므로, 반드시 센서 -> Ego 변환이 필요합니다.
        lidar_token = sample_record['data']['LIDAR_TOP']
        lidar_sd    = self.nusc.get('sample_data', lidar_token)
        lidar_cs    = self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])

        _, boxes, _ = self.nusc.get_sample_data(lidar_token)

        # LIDAR 센서 좌표계 -> Ego 좌표계 변환 (rotate then translate)
        lidar_rot   = Quaternion(lidar_cs['rotation'])
        lidar_trans = np.array(lidar_cs['translation'])
        for box in boxes:
            box.rotate(lidar_rot)
            box.translate(lidar_trans)

        # 변환된 Ego 좌표계 박스를 복셀 그리드에 그리기
        gt_semantic = self.boxes_to_voxel(boxes)

        return imgs, intrinsics, sensor2ego, gt_semantic

    def boxes_to_voxel(self, boxes):
        dx, dy, dz = self.xbound[2], self.ybound[2], self.zbound[2]
        nx = int((self.xbound[1] - self.xbound[0]) / dx)
        ny = int((self.ybound[1] - self.ybound[0]) / dy)
        nz = int((self.zbound[1] - self.zbound[0]) / dz)

        # (Z, X, Y) 형태의 Semantic Grid (0으로 초기화 = Empty)
        # 학습을 위해 Long 타입 사용 (CrossEntropy용)
        voxel_grid = np.zeros((nz, nx, ny), dtype=np.int64)

        for box in boxes:
            # 1. 클래스 확인
            class_id = 0
            for name, id in self.class_map.items():
                if name in box.name:
                    class_id = id
                    break
            
            if class_id == 0: continue # 관심 없는 물체는 건너뜀

            # 2. 박스를 Ego 좌표계로 변환 (이미 get_sample_data에서 변환되어 옴)
            # 3. 박스 영역 내의 복셀 찾기 (간단한 AABB 방식 근사)
            # 박스 중심과 크기
            center = box.center
            w, l, h = box.wlh
            
            # 박스 영역 (Min ~ Max)
            min_x = center[0] - l/2
            max_x = center[0] + l/2
            min_y = center[1] - w/2
            max_y = center[1] + w/2
            min_z = center[2] - h/2
            max_z = center[2] + h/2

            # 인덱스로 변환
            x_min_idx = int((min_x - self.xbound[0]) / dx)
            x_max_idx = int((max_x - self.xbound[0]) / dx)
            y_min_idx = int((min_y - self.ybound[0]) / dy)
            y_max_idx = int((max_y - self.ybound[0]) / dy)
            z_min_idx = int((min_z - self.zbound[0]) / dz)
            z_max_idx = int((max_z - self.zbound[0]) / dz)

            # 범위 체크
            x_min_idx = max(0, x_min_idx)
            x_max_idx = min(nx, x_max_idx)
            y_min_idx = max(0, y_min_idx)
            y_max_idx = min(ny, y_max_idx)
            z_min_idx = max(0, z_min_idx)
            z_max_idx = min(nz, z_max_idx)

            # 해당 영역 채우기
            voxel_grid[z_min_idx:z_max_idx, x_min_idx:x_max_idx, y_min_idx:y_max_idx] = class_id

        return torch.from_numpy(voxel_grid)