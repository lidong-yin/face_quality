import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm


def gen_cr_labels(W, feats, labels, batch_size=4096):
    """
    人脸特征计算CR质量伪标签, 参考论文: CR-FIQA
    Args:
        w_path (str): 训练好的W矩阵 (.npy文件) 的路径。
        feats_path (str): 特征数据 (.npy文件) 的路径。
        labels_path (str): 标签数据 (.npy文件) 的路径。
        output_path (str): 保存CR质量分 (.npy文件) 的路径。
        batch_size (int): 每个批次处理的样本数量，根据GPU显存调整。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    W_tensor = torch.from_numpy(W).float().to(device, non_blocking=True)
    W_norm = F.normalize(W_tensor, p=2, dim=0) # (feature_dim, num_classes)
    print(f"W matrix loaded. Shape: {W_tensor.shape}")

    num_samples = feats.shape[0]
    assert W_tensor.shape[0] == feats.shape[1], \
        f"W feature dim ({W_tensor.shape[0]}) != feats dim ({feats.shape[1]})"
    print(f"Found {num_samples} samples.")

    all_cr_scores = []
    print(f"Starting CR score generation with batch size {batch_size}...")
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
        end_idx = min(i + batch_size, num_samples)
        batch_feats_np = feats[i:end_idx]
        batch_labels_np = labels[i:end_idx]

        batch_feats = torch.from_numpy(batch_feats_np.copy()).float().to(device, non_blocking=True)
        batch_labels = torch.from_numpy(batch_labels_np.copy()).long().to(device, non_blocking=True)

        batch_feats_norm = F.normalize(batch_feats, p=2, dim=1)

        # 核心计算：一个矩阵乘法得到所有样本与所有类别
        cos_theta = torch.mm(batch_feats_norm, W_norm)
        # a) 计算 CCS (Class Center Similarity)
        # .gather() 精准地挑出每个样本对应其正确类别的那个余弦相似度值
        # batch_labels.view(-1, 1) 将标签[N]变为[N, 1]以满足gather的索引要求
        ccs = cos_theta.gather(1, batch_labels.view(-1, 1)).squeeze()
        # b) 计算 NNCCS (Nearest Negative Class Center Similarity)
        # 先把正确类别的cos_theta值替换成一个极小值，
        # 然后再取每行的最大值，就自然得到了最近的“负类”中心的相似度。
        cos_theta.scatter_(1, batch_labels.view(-1, 1), -1e9)
        nnccs, _ = cos_theta.max(dim=1)
        
        epsilon = 1e-12 # ε 是一个很小的数，防止分母为零
        cr_scores = ccs / (nnccs + 1 + epsilon)

        all_cr_scores.append(cr_scores.cpu().numpy())

    print("Concatenating results...")
    final_cr_scores = np.concatenate(all_cr_scores)
    print("Done!")
    return final_cr_scores


if __name__ == "__main__":
    data = pd.read_parquet("data/lyg_blur3_15_data_high_quality.parquet")
    data['cluster_id'] = pd.factorize(data['cluster_id'])[0]
    W = np.load("data/lyg_blur3_15_data_high_quality_W.npy")
    feats = np.vstack(data['feature'])
    # feats = np.load("data/lyg_final_high_quality_v1_feats.npy", mmap_mode='r')
    labels = np.array(data['cluster_id'])
    cr_quality_score = gen_cr_labels(W, feats, labels)

    print(f"Max Score: {cr_quality_score.max()}")
    print(f"Min Score: {cr_quality_score.min()}")
    print(f"Avg Score: {cr_quality_score.mean()}")
    data['cr_label'] = list(cr_quality_score)
    