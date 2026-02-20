import torch
from model import CamEncoder

def get_geometry(B, D, H, W):
    """
    (학습용) 아까 frustum.py에서 했던 것과 비슷한 3D 좌표틀을 만듭니다.
    여기서는 텐서 모양만 확인하기 위해 가짜 데이터를 씁니다.
    """
    # 실제로는 여기서 카메라 파라미터로 3D 좌표(x,y,z)를 계산해야 합니다.
    # 지금은 모양(Shape)만 맞춘 가짜 좌표를 리턴합니다.
    # (Batch, Depth, Height, Width, 3)
    return torch.zeros(B, D, H, W, 3) 

def lift_features(depth_probs, context, geometry):
    """
    [핵심 함수] Depth와 Context를 곱해서 3D Feature를 만듭니다.
    """
    # depth_probs: (B, D, H, W)
    # context:     (B, C, H, W)
    
    # 1. 차원 늘리기 (Broadcasting 준비)
    # Depth: (B, D, 1, H, W)  <- C 차원을 1로 만듦
    # Context: (B, 1, C, H, W) <- D 차원을 1로 만듦
    context = context.unsqueeze(1)
    depth_probs = depth_probs.unsqueeze(2)
    
    # 2. 외적 (Outer Product)
    # 결과: (B, D, C, H, W)
    # 해석: 각 깊이(D)마다, 특징(C)이 확률만큼 곱해져서 배치됨
    frustum_features = context * depth_probs
    
    return frustum_features

# ------------------------------------------------
# 메인 실행
# ------------------------------------------------
# 1. 설정
H, W = 256, 704
D = 41   # 깊이 구간
C = 64   # 특징 채널

# 2. 모델 준비
model = CamEncoder(D=D, C=C)
model.eval()

# 3. 가짜 이미지 입력
dummy_image = torch.randn(1, 3, H, W)

print("1. 이미지를 AI에 넣습니다...")
with torch.no_grad():
    depth_probs, context = model(dummy_image)

print(f"   - Depth 모양: {depth_probs.shape}")
print(f"   - Context 모양: {context.shape}")

# 4. Lift 실행 (핵심!)
print("\n2. 두 정보를 결합(Lift)하여 3D 특징을 만듭니다...")
# 기하학적 정보(좌표)는 일단 가짜로 생성
geometry = get_geometry(1, D, depth_probs.shape[2], depth_probs.shape[3])
lifted_features = lift_features(depth_probs, context, geometry)

print(f"\n=== 최종 결과 (Lifted Features) ===")
print(f"텐서 크기: {lifted_features.shape}")
print("해석: (배치, 깊이, 특징, 높이, 너비)")
print(f"-> 총 포인트 개수: {lifted_features.numel()}개의 데이터가 3D 공간에 퍼져 있습니다.")

if lifted_features.shape[1] == D and lifted_features.shape[2] == C:
    print("\n✅ 성공! 2D 이미지가 3D 공간(Depth)으로 확장되었습니다.")
else:
    print("\n❌ 실패! 차원을 확인하세요.")