import torch
from model import CamEncoder

# ----------------------------------------
# 설정
# ----------------------------------------
H, W = 256, 704   # 입력 이미지 크기 (nuScenes 이미지를 줄인 것)
D = 41            # 깊이 구간 (4m ~ 45m, 1m 간격)
C = 64            # 문맥 정보 채널 수

# 1. 모델 생성 (화가 고용)
print("모델을 생성하고 있습니다...")
model = CamEncoder(D=D, C=C)
model.eval() # 학습 모드 끄기 (테스트만 할 거니까)

# 2. 가짜 이미지 생성 (테스트용)
# (Batch=1, RGB=3, Height=256, Width=704)
dummy_image = torch.randn(1, 3, H, W)
print(f"입력 이미지 크기: {dummy_image.shape}")

# 3. 모델 실행 (추론)
print("AI가 이미지를 분석 중입니다...")
with torch.no_grad():
    depth_probs, context = model(dummy_image)

# 4. 결과 확인
print("\n=== 결과 분석 ===")
print(f"1. 깊이 확률(Depth) 크기: {depth_probs.shape}")
print(f"   -> (배치, 깊이구간, 높이, 너비)")
print(f"   -> 해석: 256x704 이미지가 {depth_probs.shape[2]}x{depth_probs.shape[3]}로 줄어들었고,")
print(f"            각 픽셀마다 {D}개의 거리 구간에 대한 확률을 계산했습니다.")

print(f"\n2. 특징 정보(Context) 크기: {context.shape}")
print(f"   -> 해석: 각 픽셀마다 {C}개의 시각적 특징(색깔, 모양 등)을 요약했습니다.")

if depth_probs.shape[1] == D and context.shape[1] == C:
    print("\n✅ 성공! 모델이 정상적으로 작동합니다.")
else:
    print("\n❌ 실패! 차원이 맞지 않습니다.")