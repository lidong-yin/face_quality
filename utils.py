import gc
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
from collections import defaultdict


def l2norm(vec):
    "向量归一化"
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def is_l2norm(features, size):
    "判断一组features是否满足归一化, 随机一部分向量验证"
    rand_i = random.choice(range(size))
    norm_ = np.dot(features[rand_i, :], features[rand_i, :])
    return abs(norm_ - 1) < 1e-6


def sampling_cluster(labels, feats=None, rate=0.1):
    '按比例筛选部分簇, 返回节点idx及对应的簇标签'

    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    sorted_clusters = sorted(label_indices.keys(), key=lambda x: len(label_indices[x]), reverse=True)
    sorted_cluster_sizes = [len(label_indices[cluster]) for cluster in sorted_clusters]
    
    num_blocks = int(1 // rate)
    parts = [[] for _ in range(num_blocks)] # 维护每块的节点索引
    block_sizes = [0] * num_blocks # 维护每块的节点数
    for cluster, size in tqdm(zip(sorted_clusters, sorted_cluster_sizes)):
        min_block_index = np.argmin(block_sizes)
        parts[min_block_index].extend(label_indices[cluster])
        block_sizes[min_block_index] += size
    
    part_indices = parts[0]
    if feats is None:
        return part_indices, None

    # 将部分簇的特征提取出来
    feats_block = np.empty((len(part_indices), 1024), dtype=np.float16)
    offset = 0
    chunk_size = 10000  # 减少内存
    for i in trange(0, len(part_indices), chunk_size):
        sub_indices = part_indices[i:i+chunk_size]
        sub_block = feats[sub_indices]
        n = sub_block.shape[0]
        feats_block[offset:offset+n] = sub_block
        offset += n
    
    del feats
    gc.collect()
    
    return part_indices, feats_block


def plot_distribution(arr: np.ndarray, title: str, fname: str, bins: int = 100):
    """
    绘制给定一维数组 arr 的分布曲线。
    """
    if len(arr) > 10000000:
        arr = np.random.choice(arr, size=10000000, replace=False)

    # 设置风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # 绘制图形
    sns.histplot(arr, kde=True, bins=bins, stat="density", label="Histogram (Density)")
    sns.kdeplot(arr, color='crimson', linewidth=2.5, label="KDE Curve")

    # 添加统计线
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    plt.axvline(mean_val, color='darkorange', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}')

    # 设置标题和标签
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.05)