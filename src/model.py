import torch.nn as nn
import torchvision.models as models

class DualHeadFaceNet(nn.Module):
    def __init__(self, num_id_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, in_features)
        self.gender_head = nn.Linear(in_features, 1)
        self.id_head = nn.Linear(in_features, num_id_classes)

    def forward(self, x):
        x = x.float()  
        features = self.backbone(x)
        gender_logits = self.gender_head(features)
        id_logits = self.id_head(features)
        return gender_logits, id_logits

