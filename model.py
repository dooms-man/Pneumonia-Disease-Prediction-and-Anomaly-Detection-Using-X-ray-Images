import os
import zipfile
import random
import shutil
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms, models


from transformers import ViTModel
class HybridCNNViT(nn.Module):
    def __init__(self):
        super(HybridCNNViT, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.cnn_features = densenet.features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224',output_attentions=True)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1024 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        cnn_out = self.cnn_features(x)
        cnn_out = self.cnn_pool(cnn_out).view(x.size(0), -1)
        vit_out = self.vit(pixel_values=x).last_hidden_state[:, 0, :]
        out = torch.cat((cnn_out, vit_out), dim=1)
        return self.fc(out)