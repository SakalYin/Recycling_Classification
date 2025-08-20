import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=20, num_anchors=3, grid_size=7):
        super(ObjectDetectionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size

        # Backbone: Feature extractor (e.g., simplified CNN for demonstration)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Detection Head: Outputs bounding boxes, confidence scores, and class probabilities
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        if x.shape[2] != self.grid_size or x.shape[3] != self.grid_size:
            x = F.interpolate(
                x, 
                size=(self.grid_size, self.grid_size),
                mode='bicubic',  # bilinear or bicubic works fine for features
                align_corners=False
            )

        x = self.detector(x)  # shape: [B, A*(5+C), S, S]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, S, S, A*(5+C)]

        return x
    
    def count_parameters(self):
        model = self
        """
        Count total and trainable parameters in a PyTorch model
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print(f"{'='*50}")
        print(f"MODEL PARAMETER SUMMARY")
        print(f"{'='*50}")
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")
        print(f"Non-trainable params:  {non_trainable_params:,}")
        print(f"{'='*50}")

class MobileObjectDetectionModel(nn.Module):
    """Lightweight object detection model optimized for Raspberry Pi 5"""
    def __init__(self, num_classes=20, num_anchors=3, grid_size=7, width_mult=0.5):
        super(MobileObjectDetectionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size
        
        # Calculate channel widths based on multiplier
        ch64 = int(64 * width_mult)
        ch128 = int(128 * width_mult)
        ch256 = int(256 * width_mult)
        
        # Extremely lightweight backbone using depthwise separable convolutions
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, ch64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch64),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable blocks
            DepthwiseSeparableConv(ch64, ch128, stride=2),
            DepthwiseSeparableConv(ch128, ch128, stride=1),
            DepthwiseSeparableConv(ch128, ch256, stride=2),
            DepthwiseSeparableConv(ch256, ch256, stride=1),
            
            # Global context with reduced spatial size
            nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )
        
        # Ultra-light detection head
        self.detector = nn.Sequential(
            # Single refinement layer
            DepthwiseSeparableConv(ch256, ch128, stride=1),
            nn.Dropout2d(0.1),
            
            # Direct prediction
            nn.Conv2d(ch128, num_anchors * (5 + num_classes), kernel_size=1)
        )
        
        # Initialize for mobile deployment
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Single forward pass through backbone (includes pooling)
        features = self.backbone(x)
        
        # Direct prediction
        predictions = self.detector(features)
        predictions = predictions.permute(0, 2, 3, 1).contiguous()
        
        return predictions

    def count_parameters(self):
        model = self
        """
        Count total and trainable parameters in a PyTorch model
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print(f"{'='*50}")
        print(f"MODEL PARAMETER SUMMARY")
        print(f"{'='*50}")
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")
        print(f"Non-trainable params:  {non_trainable_params:,}")
        print(f"{'='*50}")

class DepthwiseSeparableConv(nn.Module):
    """Memory and compute efficient depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu6(self.bn1(self.depthwise(x)), inplace=True)
        x = F.relu6(self.bn2(self.pointwise(x)), inplace=True)
        return x
    

class AdaptiveObjectDetectionHead(nn.Module):
    """
    Adaptive Detection Head based on grid size
    """
    def __init__(self, input_channel=128, num_classes=20,  num_anchors=3, grid_size=7):
        super(AdaptiveObjectDetectionHead).__init__()
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.grid_size = grid_size

        self.detector_input = nn.Sequential(
            nn.Conv2d(self.input_channel, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.confidence_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_anchors, kernel_size=1)            
        )

        self.class_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_anchors * self.num_classes)
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_anchors * 4)
        )

        self._initialize_heads()

    def _initialize_heads(self):
        """Init detection heads"""
        nn.init.normal_(self.confidence_head[-1].weight, std=0.01)
        nn.init.constant_(self.confidence_head[-1].bias, -2.0)  # Start with low objectness
        
        # Initialize bbox head  
        nn.init.normal_(self.bbox_head[-1].weight, std=0.01)
        nn.init.constant_(self.bbox_head[-1].bias, 0.0)
        
        # Initialize class head
        nn.init.normal_(self.class_head[-1].weight, std=0.01)
        nn.init.constant_(self.class_head[-1].bias, 0.0)

    def forward(self, x):
        feature = self.detector_input(x)
        confidences = self.confidence_head(feature)
        classes = self.class_head(feature)
        bboxes = self.bbox_head(feature)

        B, HS, H, W = confidences.shape

        confidences = confidences.view(B, self.num_anchors, 1, H, W)
        classes = classes.view(B, self.num_anchors, self.num_classes, H, W)
        bboxes = bboxes.view(B, self.num_anchors, 4, H, W)

        output = torch.cat([bboxes, confidences, classes], dim=2)  # [B, A, 5+C, H, W]
        if H != self.grid_size or W != self.grid_size:
            output = F.interpolate(
                output,
                size=(self.grid_size, self.grid_size),
                mode='bilinear',
                align_corners=False
            )

        return output
    
# class AdaptiveObjectDetectionModel(nn.Module):
#     super()