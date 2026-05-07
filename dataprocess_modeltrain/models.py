import timm
import torch
from torch import nn

MODEL_IMAGE_SIZE = 320
OUTPUT_CLASSES = ["P", "R", "S"]


class RPSClassifier(nn.Module):
    def __init__(
        self,
        image_size: int = MODEL_IMAGE_SIZE,
        head_hidden_dim: int = 256,
        dropout: float = 0.2,
        num_classes: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv1_100",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        feature_channels, feature_height, feature_width = self._infer_feature_shape(
            image_size
        )
        self.head = nn.Sequential(
            nn.Conv2d(feature_channels, head_hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(
                head_hidden_dim,
                num_classes,
                kernel_size=(feature_height, feature_width),
                bias=True,
            ),
            nn.Flatten(1),
        )

    def _infer_feature_shape(self, image_size: int):
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            features = self.backbone(dummy)
        if was_training:
            self.backbone.train()
        return features.shape[1], features.shape[2], features.shape[3]

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits
