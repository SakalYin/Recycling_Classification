import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SIoU(nn.Module):
    def __init__(self, x1y1x2y2=True, eps=1e-7):
        super(SIoU, self).__init__()
        self.x1y1x2y2 = x1y1x2y2
        self.eps = eps
  
    def forward(self, box1, box2):
        # Get the coordinates of bounding boxes
        if self.x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
    
        # IoU value of the bounding boxes
        iou = inter / union
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5) + self.eps
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            
        # Angle Cost
        angle_cost = 1 - 2 * torch.pow( torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            
        # Distance Cost
        rho_x = (s_cw / (cw + self.eps)) ** 2
        rho_y = (s_ch / (ch + self.eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            
        # Shape Cost
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        return 1 - (iou + 0.5 * (distance_cost + shape_cost))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=None):
        """
        Focal Loss implementation for both binary and multi-class classification.
        
        Args:
            alpha (float, list, or None): 
                - For binary: single float value (default: None)
                - For multi-class: list of per-class weights or None for uniform weighting
            gamma (float): focusing parameter (default: 2)
            num_classes (int or None): 
                - None for binary classification
                - Number of classes for multi-class classification
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        if num_classes is None:
            # Binary classification
            self.alpha = alpha
        else:
            # Multi-class classification
            if alpha is None:
                self.alpha = None
            elif isinstance(alpha, (list, tuple)):
                if len(alpha) != num_classes:
                    raise ValueError(f"Alpha list length ({len(alpha)}) must match num_classes ({num_classes})")
                self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            else:
                raise ValueError("For multi-class, alpha must be None or a list of per-class weights")
    
    def forward(self, inputs, targets):
        """
        Forward pass for focal loss calculation.
        
        Args:
            inputs: Model predictions
                - Binary: (N,) or (N, 1) raw logits
                - Multi-class: (N, C) raw logits where C is number of classes
            targets: Ground truth labels
                - Binary: (N,) with values 0 or 1
                - Multi-class: (N,) with class indices or (N, C) one-hot encoded
        
        Returns:
            torch.Tensor: Focal loss value
        """
        if self.num_classes is None:
            return self._binary_focal_loss(inputs, targets)
        else:
            return self._multiclass_focal_loss(inputs, targets)
    
    def _binary_focal_loss(self, inputs, targets):
        """Binary classification focal loss."""
        # Ensure targets are float for BCE
        targets = targets.float()
        
        # Calculate BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate p_t (probability of true class)
        pt = p * targets + (1 - p) * (1 - targets)
        
        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
    
    def _multiclass_focal_loss(self, inputs, targets):
        """
        Multi-class focal loss using PyTorch's cross_entropy for stability.
        `inputs` shape: (N, C) - raw logits
        `targets` shape: (N,) - class indices
        """
        if targets.dim() != 1:
            targets = torch.argmax(targets, dim=1)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")

        # Get cross-entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities of true class (pt)
        probs = F.softmax(inputs, dim=1)
        pt = probs[torch.arange(len(probs)), targets]

        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # shape: (N,)
            loss = alpha_t * loss

        return loss.mean()


class ObjectDetectionLoss:
    def __init__(self, processor, classes_alpha):
        self.processor = processor
        self.SIoU_loss = SIoU(x1y1x2y2=False)
        self.binary_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.multi_focal_loss = FocalLoss(alpha=classes_alpha, gamma=2.0, num_classes=processor.num_classes)

    def compute_loss(self, outputs, targets):
        """
        outputs: Tensor of shape [B, S, S, A, 5 + C]
        targets: Tensor of shape [B, S, S, A, 5 + C]
        """
        batch_size = outputs.shape[0]
        device = outputs.device if hasattr(outputs, 'device') else torch.device('cpu')

        total_siou = torch.tensor(0., device=device)
        total_obj = torch.tensor(0., device=device)
        total_cls = torch.tensor(0., device=device)

        for i in range(batch_size):
            output = outputs[i]
            target = targets[i]

            pred_data = self.processor.convert_yolo_output_to_bboxes(output, class_tensor=True, is_training=True)
            gt_data = self.processor.convert_yolo_output_to_bboxes(target, class_tensor=True, is_training=True)

            loss_dict = self._compute_single_image_loss(pred_data, gt_data)
            total_siou += loss_dict["siou"]
            total_obj += loss_dict["objectness"]
            total_cls += loss_dict["classification"]

        return (total_siou + total_obj + total_cls) / batch_size

    def _compute_single_image_loss(self, output, target):
        pred_data = output
        gt_data = target

        device = None
        # Try to get device from a tensor in pred_data (e.g., class_tensor)
        if len(pred_data) > 0 and 'class_tensor' in pred_data[0]:
            device = pred_data[0]['class_tensor'].device
        
        total_siou_loss = torch.tensor(0., device=device)
        total_obj_loss = torch.tensor(0., device=device)
        total_cls_loss = torch.tensor(0., device=device)
        num_matched = 0

        for gt in gt_data:
            best_iou = 0
            best_pred = None
            for pred in pred_data:
                iou = self._bbox_iou(gt['bbox'], pred['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred

            if best_iou > 0.5:
                print('matched')
                # Use torch.as_tensor to avoid breaking gradient tracking
                pred_bbox = best_pred['bbox'].to(device)
                gt_bbox = torch.as_tensor(gt['bbox'], dtype=torch.float32, device=device)
                pred_conf = best_pred['conf'].unsqueeze(0).to(device)
                pred_class = best_pred['class_tensor'].to(device)

                total_siou_loss += self.SIoU_loss(pred_bbox, gt_bbox)
                print(self.SIoU_loss(pred_bbox, gt_bbox).requires_grad)
                total_obj_loss += self.binary_focal_loss(pred_conf, torch.tensor([1.0], device=device))
                
                target_class = gt['class_id']
                if not isinstance(target_class, torch.Tensor):
                    target_class = torch.tensor([target_class], device=device)
                total_cls_loss += self.multi_focal_loss(pred_class, target_class)

                num_matched += 1

        for pred in pred_data:
            matched = any(self._bbox_iou(pred['bbox'], gt['bbox']) > 0.5 for gt in gt_data)
            if not matched:
                pred_conf = pred['conf'].unsqueeze(0).to(device)
                total_obj_loss += self.binary_focal_loss(pred_conf, torch.tensor([0.0], device=device))
        if num_matched > 0:
            total_siou_loss /= num_matched
            total_cls_loss /= num_matched

        total_loss = total_siou_loss + total_obj_loss + total_cls_loss
        return {
            "total": total_loss,
            "siou": total_siou_loss,
            "objectness": total_obj_loss,
            "classification": total_cls_loss
        }

    def _bbox_iou(self, box1, box2, eps=1e-7):
        # Convert [xc, yc, w, h] -> [x1, y1, x2, y2]
        box1 = self._xywh_to_xyxy(box1)
        box2 = self._xywh_to_xyxy(box2)

        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter + eps
        return inter / union

    def _xywh_to_xyxy(self, box):
        x, y, w, h = box
        return [x - w/2, y - h/2, x + w/2, y + h/2]
