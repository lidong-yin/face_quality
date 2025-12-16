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
        
        if pretrained:
            weights_mapping = {
                'mobilenet_v3_small': models.MobileNet_V3_Small_Weights.DEFAULT,
                'mobilenet_v3_large': models.MobileNet_V3_Large_Weights.DEFAULT,
                'efficientnet_b0': models.EfficientNet_B0_Weights.DEFAULT,
                'efficientnet_b1': models.EfficientNet_B1_Weights.DEFAULT,
                'efficientnet_b2': models.EfficientNet_B2_Weights.DEFAULT,
                'resnet18': models.ResNet18_Weights.DEFAULT,
                'resnet34': models.ResNet34_Weights.DEFAULT,
                'densenet121': models.DenseNet121_Weights.DEFAULT,
                'shufflenet_v2_x1_0': models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
                'regnet_y_400mf': models.RegNet_Y_400MF_Weights.DEFAULT,
                'regnet_y_800mf': models.RegNet_Y_800MF_Weights.DEFAULT,
            }
            weights = weights_mapping.get(backbone)
        else:
            weights = None
        
        # 初始化backbone
        if backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = 960
            self.backbone.classifier = nn.Identity()
            
        elif backbone.startswith('efficientnet'):
            if backbone == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(weights=weights)
            elif backbone == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(weights=weights)
            elif backbone == 'efficientnet_b2':
                self.backbone = models.efficientnet_b2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone in ['resnet18', 'resnet34']:
            if backbone == 'resnet18':
                self.backbone = models.resnet18(weights=weights)
            else:
                self.backbone = models.resnet34(weights=weights)
            in_features = 512
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(weights=weights)
            in_features = 1024
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'shufflenet_v2_x1_0':
            self.backbone = models.shufflenet_v2_x1_0(weights=weights)
            in_features = 1024
            self.backbone.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            
        elif backbone.startswith('regnet'):
            if backbone == 'regnet_y_400mf':
                self.backbone = models.regnet_y_400mf(weights=weights)
                in_features = 440
            elif backbone == 'regnet_y_800mf':
                self.backbone = models.regnet_y_800mf(weights=weights)
                in_features = 784
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # header
        self.header = QualityRegressorHeder(input_dim=in_features, dropout_prob=dropout_prob)
    
    def forward(self, image):
        features = self.backbone(image)
        quality_score = self.header(features)
        return quality_score