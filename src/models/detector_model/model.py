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