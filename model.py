import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class DepthEncoder(nn.Module):
    """
    Depth Encoder
    * ResNet 18
    * Use pretrained weights
    """    
    def __init__(self, pretrained):
        super(DepthEncoder, self).__init__()
        
        self.model = models.resnet18(pretrained=pretrained)
        
        self.layer0 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu
        )

    def forward(self, x):

        features = []
        x = self.layer0(x); features.append(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x); features.append(x)
        x = self.model.layer2(x); features.append(x)
        x = self.model.layer3(x); features.append(x)
        x = self.model.layer4(x); features.append(x)

        return features


class DepthDecoder(nn.Module):
    """
    Depth Decoder
    * disp : Sigmoid
    * conv : ELU 
    * Reflection Padding
    
    """
    def __init__(self):
        super(DepthDecoder, self).__init__()

        num_ch_enc = [512, 256, 128, 64, 64]
        num_ch_dec = [256, 128, 64, 32, 16]

        self.upconvs = []
        self.iconvs = []
        self.dispconvs = []

        for i in range(5):
            in_ch = num_ch_enc[i]; out_ch = num_ch_dec[i]
            if i == 4:
                in_ch = num_ch_dec[i-1]
            self.upconvs.append(self._make_conv(in_ch, out_ch))
            if i < 4:
                in_ch = num_ch_enc[i+1] + num_ch_dec[i]
            else:
                in_ch = num_ch_dec[i]
            self.iconvs.append(self._make_conv(in_ch, out_ch))
            if i > 0:
                self.dispconvs.append(self._make_dispconv(out_ch, 1))

        self.upconvs = nn.ModuleList(self.upconvs)
        self.dispconvs = nn.ModuleList(self.dispconvs)


    def _make_conv(self, in_channel, out_channel):
        layer = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(int(in_channel), int(out_channel), kernel_size=3),
            nn.ELU(inplace=True)
            )
        return layer

    def _make_dispconv(self, in_channel, out_channel):
        layer = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(int(in_channel), int(out_channel), kernel_size=3),
            nn.Sigmoid()
        )
        return layer
    
    def forward(self, features):
        disps = []
        x = features[-1]
        for i, layer in enumerate(self.upconvs):
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")

            if i < 4:
                x = torch.cat([x, features[3-i]], dim=1)

            x = self.iconvs[i](x)

            if i > 0:
                disps.append(self.dispconvs[i-1](x))
        
        return disps         
