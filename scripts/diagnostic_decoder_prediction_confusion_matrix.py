# diagnostic_confusion_matrix.py
#这个脚本是为了分析 decoder 在 subgoal prediction 上的表现，生成一个 confusion matrix 来看看哪些任务容易被混淆了。
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_eval_logs(log_dir):
    """
    只使用 logs 目录下 `subgoals_ep{N}.csv` 里 N 最大的那一个文件，
    作为最新的 episode 日志来做分析。
    """
    candidates = []

    for f in os.listdir(log_dir):
        if f.startswith("subgoals_ep") and f.endswith(".csv"):
            # 文件名形如：subgoals_ep12.csv
            core = f[len("subgoals_ep"):-4]   # 去掉前缀和 .csv，得到 "12"
            try:
                ep_idx = int(core)
            except ValueError:
                # 如果中间不是纯数字，就跳过
                continue
            candidates.append((ep_idx, f))

    if not candidates:
        raise FileNotFoundError(" No valid subgoals_ep*.csv files found in log directory.")

    # 选 episode 编号最大的那个文件
    max_ep, latest_file = max(candidates, key=lambda x: x[0])
    full_path = os.path.join(log_dir, latest_file)

    print(f"[Using latest CSV file]: {full_path} (ep={max_ep})")

    # 读取这个最新 episode 的日志
    df = pd.read_csv(full_path)

    if "true_task_idx" not in df or "pred_task_idx" not in df:
        raise ValueError(" CSV file does not contain true_task_idx or pred_task_idx columns.")

    true_labels = df["true_task_idx"].to_numpy()
    pred_labels = df["pred_task_idx"].to_numpy()

    return true_labels, pred_labels



def plot_confusion_matrix(true, pred, save_path=None, task_names=None):
    labels = np.unique(true)
    cm = confusion_matrix(true, pred, labels=labels)

    plt.figure(figsize=(8, 6))
    #blend the number in the heatmap
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=task_names, yticklabels=task_names)
    
    #make the font size larger
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True-Tasks", fontsize=16)
    plt.title("Decoder Prediction Confusion Matrix", fontsize=18)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    # ===== 修改这个路径 =====
    LOG_DIR = "/root/bayes-tmp/bowang/Inference-reutilization-MRL/output/toy1d-multi-task/2026_01_06_20_48_56_default_dpmm_seed0_regular_loss_true_time_steps48_tsne_test/DECODER_EVAL/logs"   # 例如 /home/.../EVAL/logs

    # 你的 cheetah 任务顺序（你在代码里写过）
    task_names = ["goal_front", "goal_back", "forward_vel", "backward_vel"]

    true, pred = load_eval_logs(LOG_DIR)
    print(f"Loaded {len(true)} samples.")

    save_path = os.path.join(LOG_DIR, "confusion_matrix.png")
    plot_confusion_matrix(true, pred, save_path, task_names)
    print(f"[Saved] {save_path}")