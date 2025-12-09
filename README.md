# Face Quality

This repository contains the code for the paper "SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance" (https://arxiv.org/abs/2103.05977). "CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability" (https://arxiv.org/abs/2112.06592).

## 训练
### 1. 生成标签
生成标签的方式很多，可以使用sdd-fiqa或cr-fiqa生成伪标签，也可以使用其他方法生成标签。

- **sdd-fiqa**：使用sdd-fiqa生成伪标签
  
  ```python
  python generate_sdd_label.py
  ```
  
- **cr-fiqa**：使用cr-fiqa生成伪标签

  ```python
  # 训练arcface header，得到 W 矩阵 或 根据自己筛选后的数据手动计算簇中心(推荐)
  bash train_arcface_header.sh
  ```
  
  ```python
  # 使用W 矩阵生成伪标签
  python generate_cr_label.py
  ```
  

**注意**：生成伪标签的好坏和数据有较强的相关性，详细参考: [face_quality](./assets/face_quality.md)

### 2. 训练模型
```python
bash train.sh
```

## 推理
```python
bash inference.sh
```

