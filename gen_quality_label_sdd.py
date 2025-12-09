# 2025-10-21
import os
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional
from collections import defaultdict


# 定义一个全局变量，用于在每个子进程中存储 FaceQuality 实例
g_face_quality_instance: Optional['FaceQualitySDD'] = None


def init_worker(instance: 'FaceQualitySDD'):
    """
    子进程直接初始化加载全局变量, 共享全局变量实例, 零开销
    """
    global g_face_quality_instance
    g_face_quality_instance = instance


def _worker_function(target_idx: int) -> float:
    """
    为单个样本计算原始wasserstein距离, 由多进程池中的每个worker调用。
    """
    w_distances_k = []

    instance = g_face_quality_instance
    for _ in range(instance.k):
        pos_sims = instance.pos_sampler(target_idx)
        neg_sims = instance.neg_sampler(target_idx)

        if pos_sims is not None and neg_sims is not None:
            distance = wasserstein_distance(pos_sims, neg_sims)
            w_distances_k.append(distance)
    if w_distances_k:
        q_i = np.mean(w_distances_k)
    else:
        q_i = np.nan

    return (target_idx, q_i)


class FaceQualitySDD:
    """
    基于簇内正样本与簇外负样本的Wasserstein距离计算人脸质量伪标签。
    参考论文: SDD-FIQA
    """
    def __init__(self, data: pd.DataFrame, feats: np.ndarray, cluster_id: np.ndarray):
        self.pos_m = 15     # 正样本采样量
        self.neg_m = 25     # 负样本采样量
        self.min_m = 30     # 簇最小样本量
        self.k = 5          # 采样次数
        
        self.n = len(data)
        self.feats = feats
        self.cluster_id = cluster_id
        self.blur = np.array(data['blur'])
        self.isface = np.array(data['isface'])
        self.eye_occ = np.array(data['eye_occ'])
        self.mouth_occ = np.array(data['mouth_occ'])
        self.yaw = np.array(data['yaw'])
        self.pitch = np.array(data['pitch'])
        
        self.num_work = cpu_count() // 2
        print(f"Initializing with {self.n} samples on {self.num_work} workers.")
        self._preprocess_and_normalize()

    def _preprocess_and_normalize(self):
        """
        进行所有耗时的一次性预处理：
        1. L2归一化特征，以便后续用内积计算余弦相似度。
        2. 创建从簇ID到样本索引的高效查找字典。
        """
        # print("Feats Normalize...")
        # faiss.normalize_L2(self.feats) # 原地修改数组，比numpy高效
        
        print("Building a cluster_id to indices map for fast sampling...")
        pid_to_indices_list = defaultdict(list)
        for idx, pid in enumerate(tqdm(self.cluster_id, desc="Aggregating clusters")):
            pid_to_indices_list[pid].append(idx)
        
        self.pid_to_indices: Dict[int, np.ndarray] = {
            pid: np.array(indices, dtype=np.int32) 
            for pid, indices in pid_to_indices_list.items()
        }
        print("Preprocessing finished.")

    def pos_sampler(self, target_idx: int) -> Optional[np.ndarray]:
        """
        为单个目标样本采样正样本，返回相似度分数。
        """
        target_pid = self.cluster_id[target_idx]
        all_partners_indices = self.pid_to_indices[target_pid]
        partners = all_partners_indices[all_partners_indices != target_idx]
        
        if len(partners) < self.min_m - 1:
            return None # 如果同簇邻居太少，放弃计算
        
        pick_num = min(self.pos_m, len(partners)) # 采样数
        sampled_indices = np.random.choice(partners, size=pick_num, replace=False)
        
        target_feat = self.feats[target_idx]
        sampled_feats = self.feats[sampled_indices]
        sims = sampled_feats @ target_feat
        
        return sims

    def neg_sampler(self, target_idx: int) -> Optional[np.ndarray]:
        """
        为单个目标样本采样负样本，返回相似度分数。
        随机生成一批索引，然后剔除掉与目标同簇索引。
        """
        target_pid = self.cluster_id[target_idx]
        needed = self.neg_m
        max_retries = 5 # 最大重试上限，防止死循环
        
        for _ in range(max_retries):
            sample_pool_size = int(needed * 1.5)
            random_indices = np.random.randint(0, self.n, size=sample_pool_size)
            random_pids = self.cluster_id[random_indices]

            valid_mask = (random_pids != target_pid)
            valid_negative_indices = random_indices[valid_mask]
            
            if len(valid_negative_indices) >= needed:
                final_indices = valid_negative_indices[:needed]
                target_feat = self.feats[target_idx]
                sampled_feats = self.feats[final_indices]
                sims = sampled_feats @ target_feat
                return sims
                
        return None

    def gen_labels(self) -> np.ndarray:
        """
        使用多进程并行生成所有样本的质量伪标签。
        """
        print(f"Starting pseudo-label generation with {self.num_work} processes...")
        tasks = range(self.n)
        raw_w_distances = np.full(self.n, np.nan, dtype=np.float16)

        # 创建多进程池
        with Pool(processes=self.num_work, 
                  initializer=init_worker, 
                  initargs=(self,)) as pool:
            with tqdm(total=self.n, desc="Calculating W-Distances") as pbar:
                for idx, q_i in pool.imap_unordered(_worker_function, tasks, chunksize=1024):
                    raw_w_distances[idx] = q_i
                    pbar.update(1)

        print("Raw Wasserstein distances calculated. Normalizing scores...")
        # 归一化, 计算归一化系数sigma系数 找到非nan值的最大和最小值
        valid_scores = raw_w_distances[~np.isnan(raw_w_distances)]
        print(f"Valid num: {len(valid_scores)}")
        if len(valid_scores) == 0:
            print("Warning: No valid scores were calculated.")
            return raw_w_distances

        min_score = np.min(valid_scores)
        max_score = np.max(valid_scores)
        final_scores = (raw_w_distances - min_score) / (max_score - min_score) * 100 # np.nan是簇大小不够或采样本
        print(f"Label stats: Min={min_score:.2f}, Max={max_score:.2f}, Mean={np.mean(valid_scores):.2f}")
        print("Pseudo-label generation finished.")

        return final_scores


if __name__ == "__main__":
    data_path = "data/lyg_blur_sampler_10_data.pkl"
    data = pd.read_pickle(data_path)
    feats = np.load("data/lyg_blur_sampler_10_feats.npy")
    cluster_id = np.array(data['cluster_id'])

    face_quality = FaceQualitySDD(data=data, feats=feats, cluster_id=cluster_id)
    quality_labels = face_quality.gen_labels()
    
    print("\n--- Results ---")
    print(f"Successfully generated {len(quality_labels)} quality labels.")

    data['sdd_label'] = list(quality_labels)
    breakpoint()