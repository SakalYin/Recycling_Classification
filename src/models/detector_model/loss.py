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
        angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            
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
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        if num_classes is None:
            # Binary classification - store as float, not tensor
            self.alpha_value = alpha
        else:
            # Multi-class classification
            if alpha is None:
                self.alpha_value = None
            elif isinstance(alpha, (list, tuple)):
                if len(alpha) != num_classes:
                    raise ValueError(f"Alpha list length ({len(alpha)}) must match num_classes ({num_classes})")
                self.alpha_value = alpha  # Store as list
            else:
                raise ValueError("For multi-class, alpha must be None or a list of per-class weights")
    
    def forward(self, inputs, targets, device=None):
        if device is None:
            device = inputs.device
            
        if self.num_classes is None:
            return self._binary_focal_loss(inputs, targets, device)
        else:
            return self._multiclass_focal_loss(inputs, targets, device)
    
    def _binary_focal_loss(self, inputs, targets, device):
        targets = targets.float()
        
        # Calculate BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate probabilities
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        
        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce_loss
        
        # Apply alpha weighting if specified
        if self.alpha_value is not None:
            alpha = torch.tensor(self.alpha_value, device=device, dtype=torch.float32)
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
    
    def _multiclass_focal_loss(self, inputs, targets, device):
        if targets.dim() != 1:
            targets = torch.argmax(targets, dim=1)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # Get cross-entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities of true class (pt)
        probs = F.softmax(inputs, dim=1)
        pt = probs[torch.arange(len(probs)), targets]

        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        # Apply alpha weighting if specified
        if self.alpha_value is not None:
            alpha = torch.tensor(self.alpha_value, device=device, dtype=torch.float32)
            alpha_t = alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


class ObjectDetectionLoss(nn.Module):
    def __init__(self, processor, classes_alpha, eps=1e-7):
        """
        Gradient-safe object detection loss implementation.
        
        Args:
            processor: Object detection processor
            classes_alpha: Alpha values for multi-class focal loss
            eps: Small epsilon value for numerical stability
        """
        super(ObjectDetectionLoss, self).__init__()
        self.processor = processor
        self.eps = eps
        self.siou_loss = SIoU(x1y1x2y2=False)
        self.binary_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.multi_focal_loss = FocalLoss(alpha=classes_alpha, gamma=2.0, num_classes=processor.num_classes)
    
    def forward(self, outputs, targets):
        """
        outputs: Tensor of shape [B, S, S, A, 5 + C]
        targets: Tensor of shape [B, S, S, A, 5 + C]
        """
        batch_size = outputs.shape[0]
        device = outputs.device
        
        # Initialize loss accumulators with proper gradient tracking
        total_siou = torch.zeros(1, device=device, requires_grad=True)
        total_obj = torch.zeros(1, device=device, requires_grad=True)
        total_cls = torch.zeros(1, device=device, requires_grad=True)

        for i in range(batch_size):
            output = outputs[i]
            target = targets[i]

            pred_data = self.processor.convert_yolo_output_to_bboxes(output, class_tensor=True, is_training=True)
            gt_data = self.processor.convert_yolo_output_to_bboxes(target, class_tensor=True, is_training=True)

            loss_dict = self._compute_single_image_loss(pred_data, gt_data, device)
            
            # Use proper addition to maintain gradient flow
            total_siou = total_siou + loss_dict["siou"]
            total_obj = total_obj + loss_dict["objectness"]
            total_cls = total_cls + loss_dict["classification"]

        # Average over batch
        batch_size_tensor = torch.tensor(batch_size, device=device, dtype=torch.float32)
        total_loss = (total_siou + total_obj + total_cls) / batch_size_tensor
        return total_loss.squeeze()

    def _compute_single_image_loss(self, pred_data, gt_data, device):
        """
        Compute loss for a single image with gradient-safe operations.
        """
        # Initialize loss tensors with gradient tracking
        total_siou_loss = torch.zeros(1, device=device, requires_grad=True)
        total_obj_loss = torch.zeros(1, device=device, requires_grad=True)
        total_cls_loss = torch.zeros(1, device=device, requires_grad=True)
        num_matched = 0

        if len(gt_data) == 0:
            # Handle empty cases
            # print('Empty')
            for pred in pred_data:
                pred_conf = pred['conf'].unsqueeze(0)
                neg_target = torch.zeros_like(pred_conf, device=device)
                total_obj_loss = total_obj_loss + self.binary_focal_loss(pred_conf, neg_target, device=device)
                
            return {
                "total": total_siou_loss + total_obj_loss + total_cls_loss,
                "siou": total_siou_loss,
                "objectness": total_obj_loss,
                "classification": total_cls_loss
            }

        # Compute IoU matrix between all predictions and ground truths
        iou_matrix = self._compute_iou_matrix(pred_data, gt_data, device)
        
        # Find best matches for each ground truth
        matches = []
        used_preds = set()
        
        for gt_idx in range(len(gt_data)):
            best_iou_val = torch.tensor(0.0, device=device)
            best_pred_idx = -1
            
            for pred_idx in range(len(pred_data)):
                if pred_idx not in used_preds:
                    iou_val = iou_matrix[pred_idx, gt_idx]
                    if iou_val > best_iou_val:
                        best_iou_val = iou_val
                        best_pred_idx = pred_idx
            
            # Use differentiable threshold check
            if best_pred_idx >= 0:
                threshold = torch.tensor(0.1, device=device)
                is_match = (best_iou_val >= threshold).float()
                
                if is_match > 0:
                    matches.append((best_pred_idx, gt_idx))
                    used_preds.add(best_pred_idx)
                    num_matched += 1

        # Process positive matches
        if matches:
            for pred_idx, gt_idx in matches:
                pred = pred_data[pred_idx]
                gt = gt_data[gt_idx]
                
                # Ensure tensors are on correct device and maintain gradients
                pred_bbox = pred['bbox'].to(device)
                gt_bbox = self._ensure_tensor(gt['bbox'], device)
                pred_conf = pred['conf'].unsqueeze(0).to(device)
                pred_class = pred['class_tensor'].to(device)
                
                # Compute losses
                siou_loss = self.siou_loss(pred_bbox, gt_bbox)
                obj_loss = self.binary_focal_loss(pred_conf, torch.ones_like(pred_conf, device=device), device)
                target_class = self._ensure_tensor([gt['class_id']], device, dtype=torch.long)
                cls_loss = self.multi_focal_loss(pred_class, target_class, device)
                
                # Accumulate losses
                total_siou_loss = total_siou_loss + siou_loss
                total_obj_loss = total_obj_loss + obj_loss
                total_cls_loss = total_cls_loss + cls_loss

        # ------------------------------------------------------------------------
        # TODO: how to penalize for non match
        # ------------------------------------------------------------------------

        # Process negative samples (unmatched predictions)
        for pred_idx in range(len(pred_data)):
            if pred_idx not in used_preds:
                pred_conf = pred_data[pred_idx]['conf'].unsqueeze(0).to(device)
                neg_target = torch.zeros_like(pred_conf, device=device)
                total_obj_loss = total_obj_loss + self.binary_focal_loss(pred_conf, neg_target, device)

                siou_loss = self.siou_loss(pred_bbox, gt_bbox)
                target_class = self._ensure_tensor([gt['class_id']], device, dtype=torch.long)
                cls_loss = self.multi_focal_loss(pred_class, target_class, device)

                total_siou_loss = total_siou_loss + siou_loss
                total_cls_loss = total_cls_loss + cls_loss

        # Normalize losses by number of matches (avoid division by zero)
        if num_matched > 0:
            num_matched_tensor = torch.tensor(float(num_matched), device=device)
            total_siou_loss = total_siou_loss / num_matched_tensor
            total_cls_loss = total_cls_loss / num_matched_tensor

        total_loss = total_siou_loss + total_obj_loss + total_cls_loss
        # print(f'Matched: {num_matched}')
        
        return {
            "total": total_loss,
            "siou": total_siou_loss,
            "objectness": total_obj_loss,
            "classification": total_cls_loss
        }

    def _compute_iou_matrix(self, pred_data, gt_data, device):
        """
        Compute IoU matrix between all predictions and ground truths.
        Returns a differentiable tensor of shape (num_preds, num_gts).
        """
        num_preds = len(pred_data)
        num_gts = len(gt_data)
        
        iou_matrix = torch.zeros(num_preds, num_gts, device=device)
        
        for i, pred in enumerate(pred_data):
            for j, gt in enumerate(gt_data):
                pred_bbox = pred['bbox'].to(device)
                gt_bbox = self._ensure_tensor(gt['bbox'], device)
                iou_matrix[i, j] = self._bbox_iou_differentiable(pred_bbox, gt_bbox)
        
        return iou_matrix

    def _bbox_iou_differentiable(self, box1, box2):
        """
        Differentiable IoU computation for bounding boxes in [xc, yc, w, h] format.
        """
        # Convert to [x1, y1, x2, y2] format
        box1_xyxy = self._xywh_to_xyxy_differentiable(box1)
        box2_xyxy = self._xywh_to_xyxy_differentiable(box2)

        # Compute intersection using differentiable operations
        x1 = torch.max(box1_xyxy[0], box2_xyxy[0])
        y1 = torch.max(box1_xyxy[1], box2_xyxy[1])
        x2 = torch.min(box1_xyxy[2], box2_xyxy[2])
        y2 = torch.min(box1_xyxy[3], box2_xyxy[3])

        # Clamp to ensure non-negative intersection
        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Compute areas
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        
        # Union with epsilon for numerical stability
        union = area1 + area2 - inter + self.eps
        
        return inter / union

    def _xywh_to_xyxy_differentiable(self, box):
        """
        Differentiable conversion from [xc, yc, w, h] to [x1, y1, x2, y2] format.
        """
        xc, yc, w, h = box[0], box[1], box[2], box[3]
        return torch.stack([
            xc - w / 2,  # x1
            yc - h / 2,  # y1  
            xc + w / 2,  # x2
            yc + h / 2   # y2
        ])

    def _ensure_tensor(self, data, device, dtype=torch.float32):
        """
        Ensure data is a tensor on the correct device with correct dtype.
        Maintains gradient tracking if input is already a tensor with gradients.
        """
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        else:
            return torch.tensor(data, device=device, dtype=dtype)
