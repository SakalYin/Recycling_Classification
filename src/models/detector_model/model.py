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
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Detection Head: Outputs bounding boxes, confidence scores, and class probabilities
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)), 
            nn.Conv2d(256, num_anchors * (5 + num_classes), kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.detector(x)  # shape: [B, A*(5+C), S, S]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, S, S, A*(5+C)]
        x = x.view(x.size(0), x.size(1), x.size(2), self.num_anchors * (5 + self.num_classes))
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
    