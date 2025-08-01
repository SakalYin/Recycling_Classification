import torch
import torch.nn as nn
import torch.nn.functional as F

class MediumCustomNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.15):
        super(MediumCustomNet, self).__init__()
        
        # Stem with moderate initial channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2, 2)  # 112x112
        )
        
        # Mixed architecture: some residual, some regular blocks
        self.block1 = self._make_separable_conv(32, 64, stride=2)    # 56x56
        self.block2 = self._make_residual_sep_conv(64, 96, stride=2) # 28x28, with residual
        self.block3 = self._make_separable_conv(96, 128, stride=2)   # 14x14
        self.block4 = self._make_residual_sep_conv(128, 192, stride=1) # 14x14, with residual
        
        # Squeeze-and-Excitation for the last block
        self.se = SqueezeExcitation(192, reduction=8)
        
        # Classifier with moderate complexity
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(96, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_separable_conv(self, in_channels, out_channels, stride=1):
        """Standard separable convolution block"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def _make_residual_sep_conv(self, in_channels, out_channels, stride=1):
        """Separable convolution with residual connection when possible"""
        return ResidualSepConv(in_channels, out_channels, stride)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.se(x)
        x = self.classifier(x)
        return x


class SqueezeExcitation(nn.Module):
    """Lightweight Squeeze-and-Excitation module"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, channels // reduction)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        scale = self.se(x)
        return x * scale


class ResidualSepConv(nn.Module):
    """Separable convolution with optional residual connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualSepConv, self).__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Main path
        self.sepconv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection for dimension matching
        if not self.use_residual and stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None
    
    def forward(self, x):
        out = self.sepconv(x)
        
        if self.use_residual:
            out = out + x
        elif self.shortcut is not None:
            out = out + self.shortcut(x)
        
        return F.relu6(out, inplace=True)