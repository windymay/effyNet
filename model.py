import torch
import torch.nn as nn
from torch.nn import functional as F

# drop connect function
def drop_connect(inputs, p):
    batch_size = inputs.shape[0]
    survival = 1 - p 

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = survival
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / survival * binary_tensor
    return output


# Mobile Inverted Bottleneck MBConv block
class MBConv(nn.Module):
    def __init__(self, k, f, s, C_in, C_out, se, drop_rate):
        super(MBConv, self).__init__()
        self.se = se # Squeeze and Excitation flag
        self.f = f  # expansion ratio
        self.stride = s
        self.cin = C_in
        self.cout = C_out
        self.drop_rate = drop_rate

        # Expansion phase (Inverted Bottleneck)
        # expansion ratio not 1
        if f != 1:
            # Conv1x1, BN, Relu     HxWxF -> HxWx channel_factor*F
            self.expand = nn.Conv2d(C_in, C_in*f, kernel_size=1, stride=1, bias=False)
            self.bn0 = nn.BatchNorm2d(C_in*f, momentum=0.01, eps=1e-3)
            
        # Depthwise convolution phase
        self.depthwise = nn.Conv2d(C_in*f, C_in*f, kernel_size=k, stride=s, padding=k//2, groups=C_in*f, bias=False)
        self.bn1 = nn.BatchNorm2d(C_in*f, momentum=0.01, eps=1e-3)

        # Squeeze and Excitation, if True
        # SE_Ratio = 0.25, SE_channel = C_in//4
        if se:
            self.squeeze= nn.Conv2d(C_in*f, C_in//4, kernel_size=1, stride=1)
            self.excite = nn.Conv2d(C_in//4, C_in*f, kernel_size=1, stride=1)
        
        # Pointwise convolution phase
        self.pointwise = nn.Conv2d(C_in*f, C_out, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out, momentum=0.01, eps=1e-3)

        # SiLU activation
        self.silu = nn.SiLU()


    def forward(self, inputs):
        x = inputs

        if self.f != 1:
            x = self.silu(self.bn0(self.expand(x)))
        
        x = self.silu(self.bn1(self.depthwise(x)))

        if self.se:
            squeeze = F.adaptive_avg_pool2d(x,1)
            excite = self.silu(self.squeeze(squeeze))
            excite = self.excite(excite)
            x = torch.sigmoid(excite) * x

        x = self.bn2(self.pointwise(x))
        
        # drop connect and skip connection
        if self.stride == 1 and self.cin == self.cout:
            drop_connect(x, p=self.drop_rate)
            x = x + inputs

        return x



class EfficientNet_B0(nn.Module):
    def __init__(self, n_class=21):
        super().__init__()
        # first stage 
        # Conv3x3, 224x224x3 -> 112x112x32, kernel_size=3, stride=2, padding=1
        self.conv3x3 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)

        # building blocks
        self.MB_blocks = nn.ModuleList([])
        # stage 2
        self.MB_blocks.append(MBConv(k=3, f=1, s=1, C_in=32, C_out=16, se=True, drop_rate=0.2*0/16))
        # stage 3
        self.MB_blocks.append(MBConv(k=3, f=6, s=2, C_in=16, C_out=24, se=True, drop_rate=0.2*1/16))
        self.MB_blocks.append(MBConv(k=3, f=6, s=1, C_in=24, C_out=24, se=True, drop_rate=0.2*2/16))
        # stage 4
        self.MB_blocks.append(MBConv(k=5, f=6, s=2, C_in=24, C_out=40, se=True, drop_rate=0.2*3/16))
        self.MB_blocks.append(MBConv(k=5, f=6, s=1, C_in=40, C_out=40, se=True, drop_rate=0.2*4/16))
        # stage 5
        self.MB_blocks.append(MBConv(k=3, f=6, s=1, C_in=40, C_out=80, se=True, drop_rate=0.2*5/16))
        self.MB_blocks.append(MBConv(k=3, f=6, s=1, C_in=80, C_out=80, se=True, drop_rate=0.2*6/16))
        self.MB_blocks.append(MBConv(k=3, f=6, s=1, C_in=80, C_out=80, se=True, drop_rate=0.2*7/16))
        # stage 6 
        self.MB_blocks.append(MBConv(k=5, f=6, s=2, C_in=80, C_out=112, se=True, drop_rate=0.2*8/16))
        self.MB_blocks.append(MBConv(k=5, f=6, s=1, C_in=112, C_out=112, se=True, drop_rate=0.2*9/16))
        self.MB_blocks.append(MBConv(k=5, f=6, s=1, C_in=112, C_out=112, se=True, drop_rate=0.2*10/16))
        # stage 7
        self.MB_blocks.append(MBConv(k=5, f=6, s=1, C_in=112, C_out=192, se=True, drop_rate=0.2*11/16))
        self.MB_blocks.append(MBConv(k=5, f=6, s=1, C_in=192, C_out=192, se=True, drop_rate=0.2*12/16))
        self.MB_blocks.append(MBConv(k=5, f=6, s=1, C_in=192, C_out=192, se=True, drop_rate=0.2*13/16))
        self.MB_blocks.append(MBConv(k=5, f=6, s=1, C_in=192, C_out=192, se=True, drop_rate=0.2*14/16))
        # stage 8
        self.MB_blocks.append(MBConv(k=3, f=6, s=2, C_in=192, C_out=320, se=True, drop_rate=0.2*15/16))
        
        # Final stage and FC layer
        self.conv_f = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.bn_f = nn.BatchNorm2d(1280, momentum=0.01, eps=1e-3)
        # FC layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout_f = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1280, n_class)
        self.silu = nn.SiLU()

    def forward(self, inputs):
        # Convolution layers
        x = self.silu(self.bn(self.conv3x3(inputs)))

        for idx, block in enumerate(self.MB_blocks):
            x = block(x)

        x = self.silu(self.bn_f(self.conv_f(x)))

        # Pooling and final linear layer
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout_f(x)
        x = self.fc(x)

        return x