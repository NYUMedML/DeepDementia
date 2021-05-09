
from efficientnet_pytorch_3d import EfficientNet3D
import torch
import torch.nn as nn



class EfficientNet(nn.Module):

    def __init__(self, arch_type = "efficientnet-b0", num_classes=3,in_channel=1):
        super(EfficientNet, self).__init__()

        self.model = EfficientNet3D.from_name(arch_type, override_params={'num_classes': num_classes}, in_channels=in_channel)

    def forward(self, x, age_id):
        z = self.model(x)
        return z
