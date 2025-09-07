import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


class MobileNetV3Style(nn.Module):
  def __init__(self,embedding_dim=512):
    super().__init__()

    self.backbone = models.mobilenet_v3_small(pretrained = True)
    self.backbone.classifier = nn.Identity()
    self.embedding_head = nn.Linear(self.backbone.features[-1].out_channels, embedding_dim)

  def forward(self, x):
    features = self.backbone(x)
    embedding = self.embedding_head(features)
    embedding = F.normalize(embedding, p=2,dim=1)
    return embedding
