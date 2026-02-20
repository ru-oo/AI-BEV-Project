"""
학습 완료 후 평가 + 시각화를 한번에 실행하는 파이프라인 스크립트.
train.py 학습이 끝난 뒤 실행하세요:
  python run_pipeline.py
"""
import subprocess
import sys
import os

PYTHON = sys.executable

def run(script, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    result = subprocess.run([PYTHON, script], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[오류] {script} 실행 중 문제 발생 (returncode={result.returncode})")
    return result.returncode == 0

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    steps = [
        ("evaluate_all_3d.py", "클래스별 3D mIoU 평가"),
        ("visualize_3d.py",    "시맨틱 BEV 시각화 (PNG)"),
        ("check_result_3d.py", "Pred vs GT vs Diff 비교 (PNG)"),
    ]

    success = 0
    for script, label in steps:
        if run(script, label):
            success += 1

    print(f"\n파이프라인 완료: {success}/{len(steps)} 성공")
    print("결과 파일: code2/results/ 폴더 확인")
