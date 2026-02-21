"""학습 로그 CSV에서 손실 그래프와 train_info.json을 생성합니다."""
import csv, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_path = "results/train_log.csv"
epochs, losses = [], []
with open(log_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        losses.append(float(row["loss"]))

best_loss = min(losses)

plt.figure(figsize=(10, 4))
plt.plot(epochs, losses, color='royalblue', linewidth=1.8, marker='o', markersize=4)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss  (best={best_loss:.4f}, epochs={len(epochs)})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=120)
plt.close()
print(f"손실 그래프 저장: results/loss_curve.png")

info = {
    "model": "LSS (Lift-Splat-Shoot)",
    "backbone": "ResNet18",
    "dataset": "NuScenes mini (v1.0-mini)",
    "num_classes": 4,
    "classes": ["Empty", "Car", "Truck/Bus", "Pedestrian/Bike"],
    "img_size": [384, 1056],
    "batch_size": 4,
    "accumulation_steps": 2,
    "effective_batch": 8,
    "learning_rate": 3e-4,
    "epochs_trained": len(epochs),
    "best_loss": round(best_loss, 6),
    "device": "cuda (NVIDIA GeForce RTX 5070 Laptop GPU)",
}
with open("results/train_info.json", "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print("학습 정보 저장: results/train_info.json")
