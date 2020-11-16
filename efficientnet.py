import torch
import torch.nn as nn
from efficientnet_pytorch import efficientnet

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Conv 3x3 block
class Conv3x3(nn.Module):
    def __init__(self, C_in, C_out, k, s, p):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        x = self.conv(x)
        return x


# Mobile Inverted Bottleneck MBConv block
class MBConv(nn.Module):
    def __init__(self, k, f, s, C_in, C_out, se):
        super(MBConv, self).__init__()
        self.se = se
        self.f = f
        self.stride = s
        self.cin = C_in
        self.cout = C_out

        if f != 1:
            self.expand = nn.Sequential(
                # Conv1x1, BN, Relu     HxWxF -> HxWx channel_factor*F
                nn.Conv2d(C_in, C_in * f, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(C_in * f, eps=1e-3),
                Swish()
            )
        if not se:
            self.block = nn.Sequential(
                # DWConv3x3, BN, Relu   HxWx channel_factor*F -> HxWx channel_factor*F
                nn.Conv2d(C_in * f, C_in * f, kernel_size=k, stride=s, padding=k // 2, groups=C_in * f, bias=False),
                nn.BatchNorm2d(C_in * f, eps=1e-3),
                Swish(),
                # Conv1x1, BN   HxWx channel_factor*F -> HxWxF
                nn.Conv2d(C_in * f, C_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(C_out, eps=1e-3),
            )
        else:
            self.block_se1 = nn.Sequential(
                # DWConv3x3, BN, Relu   HxWx channel_factor*F -> HxWx channel_factor*F
                nn.Conv2d(C_in * f, C_in * f, kernel_size=k, stride=s, padding=k // 2, groups=C_in * f, bias=False),
                nn.BatchNorm2d(C_in * f, eps=1e-3),
                Swish(),
                # squeeze-and-excitation， SE_Ratio = 0.25, SE_channel = C_in//4
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(C_in * f, C_in // 4, kernel_size=1, stride=1),
                Swish(),
                nn.Conv2d(C_in // 4, C_in * f, kernel_size=1, stride=1),
                nn.Sigmoid(),
            )
            self.block_se2 = nn.Sequential(
                # Conv1x1, BN   HxWx channel_factor*F -> HxWxF
                nn.Conv2d(C_in * f, C_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(C_out, eps=1e-3),
            )

    def forward(self, inputs):
        x = inputs

        if self.f != 1:
            x = self.expand(x)

        if self.se:
            y = self.block_se1(x) * x
            x = self.block_se2(y)
        else:
            x = self.block(x)

        #TODO: implement drop connect

        #skip connection
        if self.stride == 1 and self.cin == self.cout:
            x = x + inputs
        return x


# Conv1x1 & Pooling & FC
class Final_Layer(nn.Module):
    def __init__(self, C_in, C_out, n_class):
        super(Final_Layer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm2d(C_out,eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1), #flatten here
            nn.Dropout(p=0.2),
            nn.Linear(C_out, n_class),
        )

    def forward(self, x):
        x = self.block(x)
        return x


# EfficientNet-B0 baseline network
class EfficientNet_B0(nn.Module):
    def __init__(self, n_class=21):
        super().__init__()
        # Conv3x3, 224x224x3 -> 112x112x32, kernel_size=3, stride=2, padding=1
        self.stage1 = Conv3x3(C_in=3, C_out=32, k=3, s=2, p=1)
        # MBConv1, k3x3, 112x112x32 -> 112x112x16, kernel_size=3, channel_factor=1
        self.stage2 = MBConv(k=3, f=1, s=1, C_in=32, C_out=16, se=True)
        # MBConv6, k3x3, 112x112x16 -> 56x56x24, channel_factor=6   (x2)
        self.stage3 = nn.Sequential(MBConv(k=3, f=6, s=2, C_in=16, C_out=24, se=True),
                                    MBConv(k=3, f=6, s=1, C_in=24, C_out=24, se=True))
        # MBConv6, k5x5, 56x56x24 -> 28x28x40, channel_factor=6     (x2)
        self.stage4 = nn.Sequential(MBConv(k=5, f=6, s=2, C_in=24, C_out=40, se=True),
                                    MBConv(k=5, f=6, s=1, C_in=40, C_out=40, se=True))
        # MBConv6, k3x3, 28x28x40 -> 28x28x80, channel_factor=6     (x3)
        self.stage5 = nn.Sequential(MBConv(k=3, f=6, s=1, C_in=40, C_out=80, se=True),
                                    MBConv(k=3, f=6, s=1, C_in=80, C_out=80, se=True),
                                    MBConv(k=3, f=6, s=1, C_in=80, C_out=80, se=True))
        # MBConv6, k5x5, 28x28x80 -> 14x14x112, channel_factor=6     (x3)
        self.stage6 = nn.Sequential(MBConv(k=5, f=6, s=2, C_in=80, C_out=112, se=True),
                                    MBConv(k=5, f=6, s=1, C_in=112, C_out=112, se=True),
                                    MBConv(k=5, f=6, s=1, C_in=112, C_out=112, se=True))
        # MBConv6, k5x5, 14x14x112 -> 14x14x192, channel_factor=6     (x4)
        self.stage7 = nn.Sequential(MBConv(k=5, f=6, s=1, C_in=112, C_out=192, se=True),
                                    MBConv(k=5, f=6, s=1, C_in=192, C_out=192, se=True),
                                    MBConv(k=5, f=6, s=1, C_in=192, C_out=192, se=True),
                                    MBConv(k=5, f=6, s=1, C_in=192, C_out=192, se=True))
        # MBConv6, k5x5, 14x14x192 -> 7x7x320, channel_factor=6 
        self.stage8 = MBConv(k=3, f=6, s=2, C_in=192, C_out=320, se=True)
        # Conv1x1 & Pooling & FC, 7x7x320 -> 7x7x1280
        self.stage9 = Final_Layer(C_in=320, C_out=1280, n_class=n_class)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaimin initialization on conv layer, normal initialization on fc layer,
        constant initialization on batchnorm layer
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        return x
