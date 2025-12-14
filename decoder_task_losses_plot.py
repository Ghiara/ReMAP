import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # ===== 修改路径 =====
    CSV_PATH = "/home/ubuntu/yuanmeng/bo/MRL-Inference-Reutilization/output/toy1d-multi-task/2025_12_07_15_01_42_default_true_gmm_timesteps_48/TRAIN_STRIDE/logs/decoder_tasks_loss/decoder_task_losses.csv"

    df = pd.read_csv(CSV_PATH)

    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df[col], label=col)

    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Decoder Per-Task Loss Curves")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(os.path.dirname(CSV_PATH), "decoder_task_losses.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"[Saved] {save_path}")