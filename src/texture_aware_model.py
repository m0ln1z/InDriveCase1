"""
Улучшенная модель для решения проблем с различением грязи и повреждений.

Основные улучшения:
1. Более сбалансированная архитектура с вниманием к различению текстур
2. Специальные аугментации для лучшего различения dirt/damage  
3. Контрастный loss для лучшего разделения классов
4. Калибровка порогов для более сбалансированных предсказаний
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import random

class AttentionModule(nn.Module):
    """Модуль внимания для фокусировки на важных регионах изображения"""
    
    def __init__(self, in_features: int, reduction: int = 16):
        super(AttentionModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TextureAwareClassifier(nn.Module):
    """
    Улучшенная модель с фокусом на различение текстур и повреждений.
    """
    
    def __init__(self, 
                 backbone='resnet50', 
                 num_cleanliness_classes=4, 
                 num_damage_classes=3,
                 dropout_rate=0.3,  # Уменьшенный dropout
                 use_attention=True):
        super(TextureAwareClassifier, self).__init__()
        
        # Backbone с предтренированными весами
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Attention механизм
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(feature_dim)
        
        # Более мощный shared блок
        self.shared = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),  # Начальный слабый dropout
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate)
        )
        
        # Специализированные головы с разными архитектурами
        
        # Голова для чистоты - более простая, так как это в основном глобальная характеристика
        self.cleanliness_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_cleanliness_classes)
        )
        
        # Голова для повреждений - более сложная, для детального анализа текстур
        self.damage_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_damage_classes)
        )
        
        # Инициализация весов
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Инициализация весов для лучшей сходимости"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Извлечение признаков
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Применение attention
        if self.use_attention:
            features = self.attention(features)
        
        # Global Average Pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Shared представления
        shared_features = self.shared(features)
        
        return {
            'cleanliness': self.cleanliness_head(shared_features),
            'damage': self.damage_head(shared_features)
        }

class ContrastiveLoss(nn.Module):
    """
    Контрастивный loss для лучшего разделения классов чистоты и повреждений
    """
    
    def __init__(self, temperature=0.5, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features, labels):
        """
        Args:
            features: [N, D] - представления
            labels: [N] - метки классов
        """
        # Нормализация признаков
        features = F.normalize(features, dim=1)
        
        # Вычисление матрицы сходства
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Маска для позитивных пар (одинаковые классы)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Удаление диагонали (избегаем сравнения с самим собой)
        mask = mask - torch.eye(mask.size(0)).to(mask.device)
        
        # Контрастивный loss
        exp_sim = torch.exp(sim_matrix)
        sum_exp_sim = torch.sum(exp_sim * (1 - torch.eye(exp_sim.size(0)).to(exp_sim.device)), dim=1, keepdim=True)
        
        pos_sim = torch.sum(exp_sim * mask, dim=1, keepdim=True)
        loss = -torch.log(pos_sim / (sum_exp_sim + 1e-8))
        
        return loss.mean()

class ImprovedAugmentation:
    """Специальные аугментации для лучшего различения грязи и повреждений"""
    
    def __init__(self, image_size=224, severity='medium'):
        self.image_size = image_size
        self.severity = severity
        
        # Базовые аугментации
        base_transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
        ]
        
        if severity == 'light':
            augment_p = 0.3
            strength = 0.1
        elif severity == 'medium':
            augment_p = 0.5
            strength = 0.2
        else:  # heavy
            augment_p = 0.7
            strength = 0.3
            
        # Специальные аугментации для различения текстур
        texture_transforms = [
            # Помогают модели различать грязь от повреждений
            A.RandomBrightnessContrast(brightness_limit=strength*2, contrast_limit=strength*2, p=augment_p),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=augment_p*0.7),
            
            # Симуляция различных условий освещения
            A.RandomGamma(gamma_limit=(80, 120), p=augment_p*0.6),
            A.RandomToneCurve(scale=strength, p=augment_p*0.5),
            
            # Текстурные искажения
            A.GaussNoise(var_limit=(10, 50), p=augment_p*0.4),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=augment_p*0.3),
            
            # Цветовые вариации для лучшего обобщения
            A.HueSaturationValue(
                hue_shift_limit=int(strength*100), 
                sat_shift_limit=int(strength*150),
                val_shift_limit=int(strength*100), 
                p=augment_p*0.6
            ),
            
            # Размытие и резкость для имитации условий съемки
            A.OneOf([
                A.Blur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
            ], p=augment_p*0.4),
        ]
        
        # Нормализация
        normalize_transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
        
        self.transform = A.Compose(
            base_transforms + texture_transforms + normalize_transforms
        )
        
        # Валидационные трансформации (без аугментации)
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def get_train_transform(self):
        return self.transform
        
    def get_val_transform(self):
        return self.val_transform

class FocalLoss(nn.Module):
    """Focal Loss для решения проблемы дисбаланса классов"""
    
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return focal_loss.mean()
        else:
            return focal_loss

def create_improved_model(
    backbone='resnet50',
    num_cleanliness_classes=4,
    num_damage_classes=3,
    dropout_rate=0.3,
    use_attention=True
):
    """Создание улучшенной модели"""
    return TextureAwareClassifier(
        backbone=backbone,
        num_cleanliness_classes=num_cleanliness_classes,
        num_damage_classes=num_damage_classes,
        dropout_rate=dropout_rate,
        use_attention=use_attention
    )

def get_improved_transforms(image_size=224, severity='medium'):
    """Получение улучшенных трансформаций"""
    aug = ImprovedAugmentation(image_size=image_size, severity=severity)
    return {
        'train': aug.get_train_transform(),
        'val': aug.get_val_transform()
    }

if __name__ == "__main__":
    # Тест модели
    model = create_improved_model()
    x = torch.randn(2, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        
    print("Выходы модели:")
    print(f"Cleanliness: {outputs['cleanliness'].shape}")
    print(f"Damage: {outputs['damage'].shape}")
    
    print("\nМодель создана успешно!")