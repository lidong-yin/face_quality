import torch
import torch.nn as nn
from torchvision import models


class QualityRegressorHeder(nn.Module):
    '''
    人脸特征质量回归 header
    '''
    def __init__(self, input_dim=1024, dropout_prob=0.5):
        super(QualityRegressorHeder, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.regressor(x)


class FaceQualityModel(nn.Module):
    '''
    人脸质量回归, backbone + header
    '''
    def __init__(self, pretrained=True, freeze_backbone=False, dropout_prob=0.5, backbone='mobilenet_v3_small'):
        super(FaceQualityModel, self).__init__()
        # backbone
        if pretrained:
            # 预训练权重 torchvision 0.13后更新 为weights 传参
            if backbone == 'mobilenet_v3_small':
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
            elif backbone == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.DEFAULT
            elif backbone == "squeezenet":
                weights = models.SqueezeNet1_1_Weights.DEFAULT
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        else:
            weights = None

        if backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity() # 移除原始的分类头
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "squeezenet":
            self.backbone = models.squeezenet1_1(weights=weights)
            in_features = 512 # SqueezeNet 1.1 的最终输出通道数是 512, 移除它的分类器
            self.backbone.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        else:
            raise ValueError("Unsupported backbone: {self.backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # header
        self.header = QualityRegressorHeder(input_dim=in_features, dropout_prob=dropout_prob)
        
    def forward(self, image):
        features = self.backbone(image)
        quality_score = self.header(features)
        
        return quality_score