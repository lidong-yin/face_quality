import os
import hydra
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import models
from dataset import FaceDataset, FaceFeatureDataset

log = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    pbar = tqdm(loader, desc="Training")
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device).float().view(-1, 1)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 记录 Loss (加权平均)
        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        
    return total_loss / num_samples


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_mse += torch.abs(outputs - labels).sum().item()
            num_samples += batch_size

    avg_loss = total_loss / num_samples
    avg_mse = total_mse / num_samples
    return avg_loss, avg_mse


def create_scheduler(optimizer, cfg, steps_per_epoch):
    if not cfg.trainer.get("scheduler") or cfg.trainer.scheduler.name is None:
        return None

    scheduler_cfg = cfg.trainer.scheduler
    num_epoch = cfg.trainer.num_epoch
    max_lr = cfg.trainer.max_lr

    if scheduler_cfg.name == 'cosine_with_warmup':
        warmup_steps = scheduler_cfg.warmup_epochs * steps_per_epoch
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=scheduler_cfg.warmup_lr_init / max_lr,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        decay_steps = (num_epoch * steps_per_epoch) - warmup_steps
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=scheduler_cfg.eta_min
        )
        main_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
        return main_scheduler
    return None


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # 2. 初始化数据
    log.info("Loading data...")
    if cfg.data.input_type == 'img':
        df = pd.read_parquet(cfg.data.gt_file)
        imgs = df['fpath'].values
        labels = df['q_v1'].values
        X_train, X_val, y_train, y_val = train_test_split(imgs, labels, test_size=cfg.data.split_size, shuffle=True, random_state=42)
        train_dataset = FaceDataset(X_train, y_train, img_size=(cfg.model.input_size, cfg.model.input_size))
        val_dataset = FaceDataset(X_val, y_val, img_size=(cfg.model.input_size, cfg.model.input_size))
    elif cfg.data.input_type == 'feat':
        df = pd.read_parquet(cfg.data.gt_file)
        features = np.load(cfg.data.data_file, mmap_mode='r')
        labels = df['q_v1'].values
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=cfg.data.split_size, shuffle=True, random_state=42)
        train_dataset = FaceFeatureDataset(X_train, y_train)
        val_dataset = FaceFeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.trainer.batchsize, shuffle=False, num_workers=8, pin_memory=True)
    log.info(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples.")

    # 3. 初始化模型
    log.info(f"Initializing model: {cfg.model.model_name}")
    if cfg.model.model_name == 'FaceQualityModel':
        model = models.FaceQualityModel(
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone,
            backbone=cfg.model.backbone
        )
    elif cfg.model.model_name == 'QualityRegressorHeder':
        model = models.QualityRegressorHeder(input_dim=cfg.data.input_dim)
    
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # 4. 优化器与 Loss
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
    if cfg.trainer.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.max_lr, weight_decay=cfg.trainer.weight_decay)
    elif cfg.trainer.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.trainer.max_lr, momentum=cfg.trainer.momentum, weight_decay=cfg.trainer.weight_decay)
    
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))
    writer = SummaryWriter(log_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # 5. 循环训练
    best_val_mse = float('inf')
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    log.info("Starting training...")
    for epoch in range(1, cfg.trainer.num_epoch + 1):
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_mse = validate_one_epoch(model, val_loader, criterion, device)
        
        log.info(f"Epoch {epoch}/{cfg.trainer.num_epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("MSE/validation", val_mse, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

        model_path = os.path.join(output_dir, f"epoch{epoch}.pth")
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, model_path)
        log.info(f"Checkpoint saved to {model_path}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            model_path = os.path.join(output_dir, "best_model.pth")
            # 保存时处理 DataParallel 的 .module 封装
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, model_path)
            log.info(f"New best model saved to {model_path} with MSE: {best_val_mse:.4f}")

    writer.close()
    log.info("Training finished!")


if __name__ == "__main__":
    main()