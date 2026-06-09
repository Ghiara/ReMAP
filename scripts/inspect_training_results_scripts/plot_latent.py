
#这个脚本是为了可视化训练过程中 latent space 的变化，特别是对于多任务学习中的不同任务在 latent space 中的分布情况。
# 它会从指定路径下的 tensorboard 数据中读取 metadata 和 tensors，然后使用 t-SNE 或 PCA 将高维 latent space 降维到 2D 或 3D 进行可视化。每个任务的数据点会根据其规格（specs）进行颜色编码，以便更好地观察不同任务之间的关系和变化趋势。最终的图像会保存到指定的目录中，方便后续分析和展示。
#主要是根据 metadata 中的任务信息和规格信息，对 latent space 中的数据点进行分类和颜色编码，从而观察不同任务在 latent space 中的分布情况，以及它们之间的关系和变化趋势。

import sys, os, re, time, platform
import json, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
import matplotlib
from sklearn.manifold import TSNE
from pathlib import Path
import matplotlib.colors as mcolors



PLOT_LIST = []

MARKERS = ['.', '^', 's', 'p', '*', 'X', 'h', 'd', '+', 'P']

# f'{os.getcwd()}/output/toy1d-multi-task/2025_02_14_15_42_58_default_stick_breaking_run2_seed499'
def main(path, run_name=None, show_='last', save=True, use_tsne=True, DIM_RED=2, hardcoded=None):
   
    data_dir = os.path.join(path,'DECODER_EVAL','tensorboard_transfer')
    # data_dir = '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_07_11_15_51_58_default_true_gmm'
    fig_dir = os.path.join(path,'DECODER_EVAL','log', 'encodings')
    # run_name = DATA_DIR
    # if run_name is not None and os.path.exists(os.path.join(run_name, 'tensorboard')):
    #         DATA_DIR = os.path.join(run_name, 'tensorboard')
    # else:
    #     raise FileNotFoundError('Not a valid path %s' % run_name)
    data_dir_run = os.path.join(path,'tensorboard_transfer')
    run_name = Path(data_dir_run).parent
    # run_name = Path(data_dir)

    fig_folder = os.path.join(fig_dir, f'{os.path.split(data_dir)[-1]}')
    if not os.path.isdir(fig_folder) and save:
        os.makedirs(fig_folder, exist_ok=True)

    epoch_dict = {}
    if hardcoded:
        fig_folder = os.path.join(hardcoded)
        step_ = '0'
        data_dir = hardcoded
        with open(os.path.join(data_dir, 'metadata.tsv'), newline='') as f:
            r = csv.reader(f, delimiter='\t')
            metadata = [row_ for row_ in r]
        with open(os.path.join(data_dir, 'tensors.tsv'), newline='') as f:
            r = csv.reader(f, delimiter='\t')
            data = [row_ for row_ in r]

        # Convert to np
        metadata = np.array(metadata)
        data = np.array(data, dtype=float)

        # Bring data to 3D
        pcad = ''
        if data.shape[1] <= 2:
            temp = np.zeros([data.shape[0], 3])
            temp[:, 0:data.shape[1]] = data
            data = temp

        elif data.shape[1] > 2:
            if use_tsne:
                print(f'Performing T-SNE from {data.shape[1]} DIM to {DIM_RED}')
                pcad = f' (Using T-SNE From {data.shape[1]} To {DIM_RED} Dimensions)'
                true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
                pattern = r'\[([-+]?\d*\.\d+|\d+)\]'
                specs = np.array([float(re.findall(pattern, row[0])[0]) for row in metadata])
                true_tasks[true_tasks=='1']='2'
                true_tasks[(specs<0) & (true_tasks=='0')]='1'
                true_tasks[(specs<0) & (true_tasks=='2')]='3'
                unique_tasks = np.sort(np.unique(true_tasks))
                data = TSNE(n_components=DIM_RED, init='pca').fit_transform(data)
            else:
                print(f'Performing PCA from {data.shape[1]} DIM to {DIM_RED}')
                pcad = f' (Using PCA From {data.shape[1]} To {DIM_RED} Dimensions)'
                data = perform_pca(data)

        epoch_dict[step_] = [metadata, data, pcad]
    else:
        folders_ = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        maxlen = max([len(f) for f in folders_])
        folders_ = sorted([f.zfill(maxlen) for f in folders_])
        for folder in (folders_ if show_ == 'all' else [folders_[-1]]):
            if re.match('[0-9]+', folder):

                step_ = re.findall('([0-9]+)', folder)[0]

                with open(os.path.join(data_dir, folder, 'default', 'metadata.tsv'), newline='') as f:
                    r = csv.reader(f, delimiter='\t')
                    metadata = [row_ for row_ in r]
                with open(os.path.join(data_dir, folder, 'default', 'tensors.tsv'), newline='') as f:
                    r = csv.reader(f, delimiter='\t')
                    data = [row_ for row_ in r]

                # Convert to np
                metadata = np.array(metadata)
                data = np.array(data, dtype=float)

                # Bring data to 3D
                pcad = ''
                if data.shape[1] <= 2:
                    temp = np.zeros([data.shape[0], 3])
                    temp[:, 0:data.shape[1]] = data
                    data = temp

                elif data.shape[1] > 2:
                    if use_tsne:
                        print(f'Performing T-SNE from {data.shape[1]} DIM to {DIM_RED}')
                        pcad = f' (Using T-SNE From {data.shape[1]} To {DIM_RED} Dimensions)'
                        true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
                        unique_tasks = np.sort(np.unique(true_tasks))
                        data = TSNE(n_components=DIM_RED, init='pca').fit_transform(data)
                    else:
                        print(f'Performing PCA from {data.shape[1]} DIM to {DIM_RED}')
                        pcad = f' (Using PCA From {data.shape[1]} To {DIM_RED} Dimensions)'
                        data = perform_pca(data)

                epoch_dict[step_] = [metadata, data, pcad]

        m_l = max([len(k) for k in epoch_dict.keys()])

    if os.path.exists(os.path.join(run_name, 'task_dict.json')):
        with open(os.path.join(run_name, 'task_dict.json'), 'r') as f:
            # Copy so not both changed during updates
            task_dict = json.load(f)
            task_dict = {str(el): key for key, el in task_dict.items()} if task_dict is not None else None

    # Map
    class_map = {'flip_forward': 'Front flip', 'jump': 'Jump',
                 'goal_backward': 'Goal in back', 'goal_forward': 'Goal in front',
                 'stand_back': 'Back stand', 'stand_front': 'Front stand',
                 'velocity_backward': 'Run backward', 'velocity_forward': 'Run forward',
                 'direction_backward': 'Run forward', 'direction_forward': 'Run backward',
                 'reach-v2': 'reach-v2', 'push-v2': 'push-v2', 
                 'velocity_up': 'Velocity Up', 'velocity_down': 'Velocity Down', 'velocity_right': 'Velocity Right', 'velocity_left': 'Velocity Left',
                 'goal_up': 'Goal Up', 'goal_down': 'Goal Down', 'goal_right': 'Goal Right', 'goal_left': 'Goal Left'}

    # Plotting

    plt.style.use('seaborn-v0_8')

    # Use Latex text
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    size_ = 31
    plt.rc('font', size=size_)  # controls default text sizes
    plt.rc('figure', titlesize=size_)  # fontsize of the figure title
    plt.rc('axes', titlesize=size_)  # fontsize of the axes title
    plt.rc('axes', labelsize=size_)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size_*0.8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size_*0.8)  # legend fontsize

    for step_ in sorted(epoch_dict.keys()):

        metadata, values, pcad = epoch_dict[step_]

        metadata_strings = np.array([s[0] for s in metadata])
        true_tasks = np.array([s.split('[')[0].strip() for s in metadata_strings])
        unique_tasks = np.sort(np.unique(true_tasks))
        max_len_task = max([len(t) for t in unique_tasks])

        parsed_numbers = [re.findall('[0-9]+[.]*[0-9]*', s) for s in metadata_strings]
        specs = np.array([float(nums[1]) for nums in parsed_numbers])
        if parsed_numbers and len(parsed_numbers[0]) == 4:
            specs = np.zeros(len(parsed_numbers))
            for i, nums in enumerate(parsed_numbers):
                pair = [float(s) for s in nums[1:3]]
                nz = np.nonzero(pair)[0]
                specs[i] = pair[nz[0].item()] if len(nz) > 0 else pair[0]
        predicted_tasks = np.array([s.split('->')[1].strip() for s in metadata_strings])

        fig = plt.figure()

        if values.shape[1] == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_aspect('auto')
        else:
            ax = plt.gca()
            ax.set_title('Cheetah multi-task after deploying toy dpmm inference module')

        legend_elements = []
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors += ['#db8233', '#8c564b', '#e377c2']

        for i, target_class in enumerate(unique_tasks):
            matching_indices = np.where(true_tasks == target_class)[0]
            true_match_values = values[matching_indices]
            specs_filtered = specs[matching_indices]
            max_spec = np.max(specs_filtered)
            if max_spec > 0:
                specs_filtered = specs_filtered / max_spec
            else:
                specs_filtered = np.ones_like(specs_filtered)

            rgb = mcolors.to_rgb(colors[i % len(colors)])
            hsv = mcolors.rgb_to_hsv(rgb)
            point_colors = np.zeros((len(specs_filtered), 3))
            for j, brightness in enumerate(specs_filtered):
                hsv_j = hsv.copy()
                hsv_j[2] = brightness * 0.5 + 0.5
                point_colors[j] = mcolors.hsv_to_rgb(hsv_j)

            label = class_map[task_dict[target_class]]
            if true_match_values.shape[1] == 3:
                el = ax.scatter(
                    true_match_values[:, 0],
                    true_match_values[:, 1],
                    true_match_values[:, 2],
                    label=label,
                    marker=MARKERS[0],
                    color=point_colors,
                )
            else:
                el = ax.scatter(
                    true_match_values[:, 0],
                    true_match_values[:, 1],
                    label=label,
                    marker=MARKERS[0],
                    color=point_colors,
                )

            legend_elements.append(el)


        if values.shape[1] == 3:
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
            ax.set_zlabel('Latent Dim 3')
            # Equal scaling
            x_lim_min, x_lim_max = ax.get_xlim()
            y_lim_min, y_lim_max = ax.get_ylim()
            z_lim_min, z_lim_max = ax.get_zlim()
            max_ = np.array([(x_lim_max - x_lim_min), (y_lim_max - y_lim_min), (z_lim_max - z_lim_min)]).max()
            ax.set_xlim(x_lim_min - (max_ - (x_lim_max - x_lim_min)) / 2, x_lim_max + (max_ - (x_lim_max - x_lim_min)) / 2)
            ax.set_ylim(y_lim_min - (max_ - (y_lim_max - y_lim_min)) / 2, y_lim_max + (max_ - (y_lim_max - y_lim_min)) / 2)
            ax.set_zlim(z_lim_min - (max_ - (z_lim_max - z_lim_min)) / 2, z_lim_max + (max_ - (z_lim_max - z_lim_min)) / 2)
        else:
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')

            ts = ax.get_xticks()
            if len(ts) < 5:
                ax.set_xticks(np.linspace(min(ts), max(ts), 5))
            ts = ax.get_yticks()
            if len(ts) < 5:
                ax.set_yticks(np.linspace(min(ts), max(ts), 5))

        # ax.view_init(-90, 90)

        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.9, 0.5), markerscale=1.5)

        fig.set_size_inches(10., 7.1) #8 task

        if save:
            plt.savefig(os.path.join(fig_folder, f'encodings.pdf'), format='pdf', dpi=100, bbox_inches='tight')
            plt.savefig(os.path.join(fig_folder, f'encodings.png'), format='png', dpi=100, bbox_inches='tight')

        else:
            plt.show()

        plt.close()

        print(f'Created plot for {data_dir}')

        print("Loaded steps:", list(epoch_dict.keys()))


# help function to compute and plot drift trajectory

def compute_and_plot_drift(epoch_dict, fig_dir, task_dict, class_map):
    """
    计算每个任务在不同 step 下的质心 (centroid) 并绘制轨迹。
    epoch_dict: {step: [metadata, data, pcad]}
    """
    # ========== 收集质心数据 ==========
    centroids = []  # 每行：[step, task_name, x, y]
    for step_ in sorted(epoch_dict.keys(), key=lambda x: int(x)):
        metadata, values, _ = epoch_dict[step_]
        true_tasks = np.array([s[0].split('[')[0].strip() for s in metadata])
        unique_tasks = np.unique(true_tasks)

        for task in unique_tasks:
            idx = np.where(true_tasks == task)[0]
            if len(idx) == 0:
                continue
            centroid = np.mean(values[idx, :2], axis=0)  # 用 t-SNE 的前两维
            centroids.append([int(step_), task, centroid[0], centroid[1]])

    centroids = np.array(centroids, dtype=object)

    # ========== 转换为可绘制形式 ==========
    plt.figure(figsize=(8,6))
    plt.title("Task Centroid Drift over Time")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")

    unique_tasks = np.unique(centroids[:,1])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, task in enumerate(unique_tasks):
        data_t = centroids[centroids[:,1] == task]
        data_t = data_t[np.argsort(data_t[:,0])]  # 按 step 排序
        steps = data_t[:,0].astype(int)
        xs = data_t[:,2].astype(float)
        ys = data_t[:,3].astype(float)

        # 绘制轨迹线
        plt.plot(xs, ys, '-o', color=colors[i % len(colors)],
                 label=class_map.get(task_dict.get(task, task), task))

        # 起点和终点标注
        plt.annotate(f'start ({steps[0]})', (xs[0], ys[0]), textcoords="offset points", xytext=(5,5), fontsize=8)
        plt.annotate(f'end ({steps[-1]})', (xs[-1], ys[-1]), textcoords="offset points", xytext=(5,5), fontsize=8)

    plt.legend(loc='best', frameon=True)
    drift_path = os.path.join(fig_dir, "encodings_drift.png")
    plt.savefig(drift_path, dpi=200, bbox_inches='tight')
    plt.savefig(drift_path.replace('.png','.pdf'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Drift trajectory plot saved to {drift_path}")

def perform_pca(values):

    # sample points equally for all gaussians
    x = np.copy(values)

    # centering the data
    x -= np.mean(x, axis=0)

    cov = np.cov(x, rowvar=False)

    evals, evecs = LA.eigh(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx[:3]]

    return np.dot(x, evecs)
if __name__ == '__main__':

    paths = [
            
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_16_15_34_37_default_true_gmm',     # keep training var0.1 step 32, with var 0.1 ??
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_21_11_05_23_default_true_gmm',     # var 0.05
    # '/home/ubuntu/juan/melts/output/toy1d-multi-task/2024_09_21_11_06_00_default_true_gmm',     # var 0.02
    '/root/bayes-tmp/bowang/Inference-reutilization-MRL/output/toy1d-multi-task/2026_01_06_20_48_56_default_dpmm_seed0_regular_loss_true_time_steps48_tsne_test',     # var 0.05


        ]
    for path in paths:
        main(path, *sys.argv[1:], hardcoded=None)