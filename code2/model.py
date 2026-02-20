import torch
import torch.nn as nn
from torchvision.models import resnet18

class CamEncoder(nn.Module):
    def __init__(self, D, C, downsample=16):
        super(CamEncoder, self).__init__()
        self.D = D
        self.C = C
        
        # 1. ResNet50으로 교체
        self.trunk = resnet18(pretrained=True)
        
        self.trunk.fc = nn.Identity()
        self.trunk.avgpool = nn.Identity()
        
        # ResNet50의 layer4 출력 채널은 2048개입니다 (ResNet18은 512개)
        # 따라서 입력 채널을 512 -> 2048로 바꿔줘야 합니다.
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, D + C, kernel_size=1, padding=0) 
        )

    def get_cam_feats(self, x):
        """
        x: 입력 이미지 (Batch, 3, H, W)
        """
        # ResNet 통과 (이미지가 줄어들면서 특징만 남음)
        # ResNet 구조상 이미지가 1/32로 줄어드는데, 편의상 중간 과정을 생략하고 결과만 봅니다.
        # 실제로는 ResNet의 중간 레이어(layer1~4)를 거치며 사이즈가 줄어듭니다.
        
        # 여기서는 ResNet의 layer4까지 통과시킵니다.
        x = self.trunk.conv1(x)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x = self.trunk.layer2(x)
        x = self.trunk.layer3(x)
        x = self.trunk.layer4(x) 
        # 여기까지 오면 이미지가 1/32 크기로 작아지고 채널은 512개가 됩니다.

        # 1x1 Convolution을 통해 우리가 원하는 형태(D+C)로 바꿉니다.
        x = self.layer1(x)
        
        return x 

    def forward(self, x):
        # 1. 특징 추출
        x = self.get_cam_feats(x)
        
        # 2. 결과 쪼개기 (깊이 vs 특징)
        # x의 모양: (Batch, D+C, H_small, W_small)
        
        # 깊이(Depth) 정보: 앞쪽 D개 채널
        depth_logits = x[:, :self.D]
        
        # 특징(Context) 정보: 뒤쪽 C개 채널
        context = x[:, self.D:]
        
        # 깊이 값은 확률이므로 Softmax를 씌워서 0~1 사이 값(확률)으로 만듭니다.
        depth_probs = depth_logits.softmax(dim=1)
        
        return depth_probs, context