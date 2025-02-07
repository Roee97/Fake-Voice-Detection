import torch
from wavencoder.models.rawnet import RawNet2Model


class RawNet2BinaryClassifier(RawNet2Model):
    def __init__(self, pretrained=True, pretrained_path="pretrained_models_pt/rawnet2_best_weights.pt", device="cpu"):
        super().__init__(pretrained=pretrained, class_dim=1, pretrained_path=pretrained_path, device=device, return_code=False)
        # class_dim=1 ensures the output is a single scalar (logit)

    def forward(self, x):
        x = super().forward(x)
        # Apply sigmoid for binary classification
        x = torch.sigmoid(x)
        return x