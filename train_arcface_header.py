# train_header.py
import os
import math
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter


class ArcFaceHeader(nn.Module):
    """
    ArcFace Header, acting as a trainable classification layer.
    Its forward pass computes the final logits for CrossEntropyLoss.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFaceHeader, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # The learnable weight matrix W, representing class centers, d * C
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.kernel)

        # Constants for ArcFace margin application
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = self.sin_m * m

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)

        # Get cos(theta_y) for the ground truth class
        target_logit = cos_theta.gather(1, labels.view(-1, 1))

        # Calculate cos(theta + m) using trigonometric identity
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m

        # Apply margin penalty with the paper's trick
        final_target_logit = torch.where(
            target_logit > self.th,
            cos_theta_m,
            target_logit - self.mm
        )

        # Substitute the modified target logit back into the cos_theta matrix
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output_logits = (one_hot * final_target_logit) + ((1.0 - one_hot) * cos_theta)
        # Scale the logits
        output_logits *= self.s
        
        return output_logits


class FeatureDataset(Dataset):
    """
    Custom dataset to load pre-extracted features and labels.
    - Assumes features are in a .npy file (or memmap) of shape [num_samples, feature_dim]
    - Assumes labels are in a .npy file of shape [num_samples]
    """
    def __init__(self, features_path, labels_path):
        super(FeatureDataset, self).__init__()
        logging.info(f"Loading features from {features_path} (memmap mode)")
        self.features = np.load(features_path, mmap_mode='r')
        logging.info(f"Loading labels from {labels_path} (memmap mode)")
        self.labels = np.load(labels_path, mmap_mode='r')
        
        assert self.features.shape[0] == self.labels.shape[0], "Features and labels must have the same number of samples"
        self.num_samples = self.features.shape[0]
        logging.info(f"Dataset loaded with {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        feature = torch.from_numpy(self.features[index].copy()).float()
        label = torch.from_numpy(np.array(self.labels[index])).long()
        return feature, label


def train_arcface_header(args):
    # DDP 初始化
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 日志设置
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - RANK:{rank} - %(levelname)s - %(message)s')

    # 创建输出目录
    writer = None
    val_loader = None
    if rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        log_dir = os.path.join(args.output_dir, "logs")
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"TensorBoard logs will be saved to: {log_dir}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if args.val_features_path is not None and args.val_labels_path is not None:
            val_dataset = FeatureDataset(args.val_features_path, args.val_labels_path)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size * 2, # 验证时可以用更大的batch size
                shuffle=False, # 验证集不需要打乱
                num_workers=args.num_workers,
                pin_memory=True
            )

    # 加载数据集
    train_dataset = FeatureDataset(args.train_features_path, args.train_labels_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 初始化模型 (ArcFaceHeader)
    header = ArcFaceHeader(
        in_features=args.feature_dim, 
        out_features=args.num_classes,
        s=args.s,
        m=args.m
    ).to(local_rank)

    # 将模型包装为 DDP 模型
    header = DistributedDataParallel(module=header, device_ids=[local_rank])
    header.train()

    optimizer = optim.SGD(
        params=header.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    criterion = CrossEntropyLoss()

    logging.info("Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    warmup_steps = args.warmup_steps
    target_lr = args.lr

    for epoch in range(args.num_epoch):
        train_sampler.set_epoch(epoch) # DDP 要求每个 epoch 设置一次
        for step, (features, labels) in enumerate(train_loader):
            features = features.cuda(local_rank, non_blocking=True)
            labels = labels.cuda(local_rank, non_blocking=True)

            if global_step < warmup_steps:
                lr_scale = global_step / warmup_steps # 线性增加学习率
                # 为optimizer中的每个参数组设置新的学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = target_lr * lr_scale

            logits = header(features, labels)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(header.parameters(), max_norm=5.0) # 梯度裁切,防止梯度爆炸
            optimizer.step()

            # 日志记录 (只在 rank 0 进程打印)
            if rank == 0 and step % args.log_freq == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch: [{epoch+1}/{args.num_epoch}] | "
                    f"Step: [{step}/{len(train_loader)}] | "
                    f"Step_global: {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.5f}"
                )
                if writer is not None:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    writer.add_scalar('Learning_Rate/lr', current_lr, global_step)
            global_step += 1
        
        if epoch >= (warmup_steps / len(train_loader)):
            scheduler.step()

        check_kernel = header.module.kernel.detach().cpu().numpy()
        check_kernel_save_path = os.path.join(args.output_dir, f"epoch{epoch+1}_kernel_W.npy")
        np.save(check_kernel_save_path, check_kernel)

        if rank == 0 and val_loader is not None:
            header.eval()
            total_val_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.cuda(local_rank, non_blocking=True)
                    labels = labels.cuda(local_rank, non_blocking=True)
                    
                    logits = header(features, labels)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item() * features.size(0)
                    
                    _, predicted = torch.max(logits, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += features.size(0)

            avg_val_loss = total_val_loss / total_samples
            val_accuracy = total_correct / total_samples
            
            logging.info(f"--- Epoch {epoch+1} Validation ---")
            logging.info(f"Avg Validation Loss: {avg_val_loss:.4f}")
            logging.info(f"Validation Accuracy: {val_accuracy:.4f}")

            header.train()

            if writer is not None:
                writer.add_scalar('Loss/validation', avg_val_loss, global_step)
                writer.add_scalar('Accuracy/validation', val_accuracy, global_step)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                kernel_save_path = os.path.join(args.output_dir, "best_kernel_W.npy")
                best_kernel = header.module.kernel.detach().cpu().numpy()
                np.save(kernel_save_path, best_kernel)
                best_model_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(header.module.state_dict(), best_model_path)
                logging.info(f"*** New best model saved to {kernel_save_path} with val_loss: {best_val_loss:.4f} ***")

    if rank == 0:
        # 训练结束后，单独保存最终的权重矩阵 W
        final_kernel = header.module.kernel.detach().cpu().numpy()
        kernel_save_path = os.path.join(args.output_dir, "final_kernel_W.npy")
        np.save(kernel_save_path, final_kernel)
        logging.info(f"Final Kernel (W) saved to {kernel_save_path}")
        logging.info(f"Shape of W: {final_kernel.shape}")
        final_model_path = os.path.join(args.output_dir, f"final_model.pth")
        torch.save(header.module.state_dict(), final_model_path)
        logging.info(f"Final Model saved to {final_model_path}")

        if writer is not None:
            writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ArcFace Header on Fixed Features')
    parser.add_argument('--train_features_path', type=str, required=True, help="Path to the .npy file containing features")
    parser.add_argument('--train_labels_path', type=str, required=True, help="Path to the .npy file containing labels")
    parser.add_argument('--val_features_path', type=str, default=None, help="Path to the .npy file containing features")
    parser.add_argument('--val_labels_path', type=str, default=None, help="Path to the .npy file containing labels")
    parser.add_argument('--output_dir', type=str, default="./header_output", help="Directory to save checkpoints and final kernel")

    # 训练超参数
    parser.add_argument('--feature_dim', type=int, default=1024, help="Dimension of input features")
    parser.add_argument('--num_classes', type=int, required=True, help="Total number of classes (identities)")
    parser.add_argument('--num_epoch', type=int, default=16, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size per GPU")
    parser.add_argument('--lr', type=float, default=0.01, help="Initial learning rate")
    parser.add_argument('--lr_step', type=int, default=5, help="Epochs after which to decay learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay")
    parser.add_argument('--warmup_steps', type=int, default=2000, help="Number of linear warmup steps.")
    
    # ArcFace 超参数
    parser.add_argument('--s', type=float, default=64.0, help="Scale parameter in ArcFace")
    parser.add_argument('--m', type=float, default=0.5, help="Margin parameter in ArcFace")

    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loading workers per GPU")
    parser.add_argument('--log_freq', type=int, default=100, help="Frequency of logging (in steps)")
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    
    args = parser.parse_args()
    train_arcface_header(args)
