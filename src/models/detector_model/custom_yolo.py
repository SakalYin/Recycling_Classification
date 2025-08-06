import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    @staticmethod
    def autopad(k, p=None, d=1):
        """Auto-padding for 'same' padding"""
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depthwise convolution"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class PSA(nn.Module):
    """Polarized Self-Attention module"""
    def __init__(self, c, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.c = c
        self.head_dim = c // n_heads
        
        self.conv1 = Conv(c, c * 2, 1)
        self.conv2 = Conv(c * 2, c, 1)
        self.attn = nn.MultiheadAttention(c, n_heads, batch_first=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Channel attention
        y = self.conv1(x)
        y1, y2 = y.chunk(2, dim=1)
        
        # Spatial attention via MultiheadAttention
        y1_flat = y1.flatten(2).transpose(1, 2)  # B, HW, C
        attn_out, _ = self.attn(y1_flat, y1_flat, y1_flat)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)
        
        # Combine
        y = attn_out * y2
        return self.conv2(y) + x


class SCDown(nn.Module):
    """Spatial-Channel Decoupled Downsampling"""
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c1, k=k, s=s, g=c1)  # Depthwise
        self.cv2 = Conv(c1, c2, k=1, s=1)        # Pointwise
        
    def forward(self, x):
        return self.cv2(self.cv1(x))


class C2fPSA(nn.Module):
    """C2f module with PSA attention"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(PSA(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f(nn.Module):
    """Faster implementation of CSP Bottleneck with 2 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Detect(nn.Module):
    """YOLOv10 Detect head with one-to-many and one-to-one heads"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        # One-to-many head (training)
        self.one2many_cv2 = nn.ModuleList(nn.Sequential(Conv(x, 64, 3), Conv(64, 64, 3), nn.Conv2d(64, 4 * self.reg_max, 1)) for x in ch)
        self.one2many_cv3 = nn.ModuleList(nn.Sequential(Conv(x, 64, 3), Conv(64, 64, 3), nn.Conv2d(64, self.nc, 1)) for x in ch)
        
        # One-to-one head (inference)
        self.one2one_cv2 = nn.ModuleList(nn.Sequential(Conv(x, 64, 3), Conv(64, 64, 3), nn.Conv2d(64, 4 * self.reg_max, 1)) for x in ch)
        self.one2one_cv3 = nn.ModuleList(nn.Sequential(Conv(x, 64, 3), Conv(64, 64, 3), nn.Conv2d(64, self.nc, 1)) for x in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        if self.training:
            # One-to-many head for training
            return self.forward_one2many(x)
        else:
            # One-to-one head for inference
            return self.forward_one2one(x)

    def forward_one2many(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.one2many_cv2[i](x[i]), self.one2many_cv3[i](x[i])), 1)
        return x

    def forward_one2one(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.export:
            return x
        
        # Inference path
        return self.postprocess(x, shape)

    def postprocess(self, preds, img_shape):
        """Post-process predictions"""
        anchors, strides = (x.transpose(0, 1) for x in self.make_anchors(preds, self.stride, 0.5))
        
        for i, pred in enumerate(preds):
            b, c, h, w = pred.shape
            pred = pred.view(b, self.no, h * w).permute(0, 2, 1)  # box, cls
            
            # Decode boxes
            if self.reg_max > 1:
                dbox = self.dfl(pred[..., :self.reg_max * 4].view(b, h * w, 4, self.reg_max))
            else:
                dbox = pred[..., :4]
            
            # Apply anchors and strides
            dbox = self.dist2bbox(dbox, anchors[i], xywh=True, dim=1) * strides[i]
            cls = pred[..., 4 * self.reg_max:].sigmoid()
            preds[i] = torch.cat([dbox, cls], dim=-1)
            
        return preds

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)
        return torch.cat((x1y1, x2y2), dim)


class DFL(nn.Module):
    """Distribution Focal Loss (DFL)"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class YOLOv10n(nn.Module):
    """YOLOv10 Nano model"""
    def __init__(self, nc=80, ch=3):
        super().__init__()
        
        # Model scaling parameters for nano version
        depth_multiple = 0.33  # model depth multiple
        width_multiple = 0.25  # layer channel multiple
        
        # Calculate channel sizes
        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor
        
        # Backbone channels
        ch_list = [64, 128, 256, 512, 1024]
        ch_list = [make_divisible(x * width_multiple) for x in ch_list]
        
        # Backbone
        self.backbone = nn.ModuleList([
            # P1/2
            Conv(ch, ch_list[0], 3, 2),  # 0
            # P2/4  
            Conv(ch_list[0], ch_list[1], 3, 2),  # 1
            C2f(ch_list[1], ch_list[1], max(round(3 * depth_multiple), 1), True),  # 2
            # P3/8
            Conv(ch_list[1], ch_list[2], 3, 2),  # 3  
            C2f(ch_list[2], ch_list[2], max(round(6 * depth_multiple), 1), True),  # 4
            # P4/16
            SCDown(ch_list[2], ch_list[3], 3, 2),  # 5
            C2fPSA(ch_list[3], ch_list[3], max(round(6 * depth_multiple), 1)),  # 6
            # P5/32
            SCDown(ch_list[3], ch_list[4], 3, 2),  # 7
            C2fPSA(ch_list[4], ch_list[4], max(round(3 * depth_multiple), 1)),  # 8
            SPPF(ch_list[4], ch_list[4], 5),  # 9
        ])
        
        # Neck - FPN
        self.neck = nn.ModuleList([
            nn.Upsample(None, 2, 'nearest'),  # 10
            C2f(ch_list[4] + ch_list[3], ch_list[3], max(round(3 * depth_multiple), 1)),  # 11
            nn.Upsample(None, 2, 'nearest'),  # 12
            C2f(ch_list[3] + ch_list[2], ch_list[2], max(round(3 * depth_multiple), 1)),  # 13
            Conv(ch_list[2], ch_list[2], 3, 2),  # 14
            C2f(ch_list[2] + ch_list[3], ch_list[3], max(round(3 * depth_multiple), 1)),  # 15
            SCDown(ch_list[3], ch_list[3], 3, 2),  # 16
            C2f(ch_list[3] + ch_list[4], ch_list[4], max(round(3 * depth_multiple), 1)),  # 17
        ])
        
        # Detection head
        self.head = Detect(nc, (ch_list[2], ch_list[3], ch_list[4]))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone
        x = self.backbone[0](x)  # P1
        x = self.backbone[1](x)  # P2
        x = self.backbone[2](x)
        p3 = self.backbone[3](x)  # P3
        p3 = self.backbone[4](p3)
        p4 = self.backbone[5](p3)  # P4
        p4 = self.backbone[6](p4)
        p5 = self.backbone[7](p4)  # P5
        p5 = self.backbone[8](p5)
        p5 = self.backbone[9](p5)
        
        # Neck - FPN
        x = self.neck[0](p5)  # upsample
        x = torch.cat([x, p4], 1)
        x = self.neck[1](x)  # C2f
        p4_out = x
        
        x = self.neck[2](x)  # upsample
        x = torch.cat([x, p3], 1)
        p3_out = self.neck[3](x)  # C2f
        
        x = self.neck[4](p3_out)  # downsample
        x = torch.cat([x, p4_out], 1)
        p4_out = self.neck[5](x)  # C2f
        
        x = self.neck[6](p4_out)  # downsample
        x = torch.cat([x, p5], 1)
        p5_out = self.neck[7](x)  # C2f
        
        # Detection
        return self.head([p3_out, p4_out, p5_out])


def yolov10n(num_classes=80, pretrained=False):
    """Create YOLOv10n model"""
    model = YOLOv10n(nc=num_classes)
    
    if pretrained:
        # You would load pretrained weights here
        # checkpoint = torch.load('yolov10n.pt')
        # model.load_state_dict(checkpoint['model'])
        pass
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model for 7 classes (your use case)
    model = yolov10n(num_classes=7)
    
    # Test with dummy input
    x = torch.randn(1, 3, 640, 640)
    model.eval()
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shapes: {[out.shape for out in outputs]}")
    
    # Training mode
    model.train()
    train_outputs = model(x)
    print(f"Training output shapes: {[out.shape for out in train_outputs]}")