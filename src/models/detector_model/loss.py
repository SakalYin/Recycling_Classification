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
        if self.x1y1x2y2:  
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
    
        iou = inter / union
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5) + self.eps
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            
        angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            
        rho_x = (s_cw / (cw + self.eps)) ** 2
        rho_y = (s_ch / (ch + self.eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            
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
            self.alpha_value = alpha
        else:
            if alpha is None:
                self.alpha_value = None
            elif isinstance(alpha, (list, tuple)):
                if len(alpha) != num_classes:
                    raise ValueError(f"Alpha list length ({len(alpha)}) must match num_classes ({num_classes})")
                self.alpha_value = alpha
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
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce_loss
        
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

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        probs = F.softmax(inputs, dim=1)
        pt = probs[torch.arange(len(probs)), targets]
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha_value is not None:
            alpha = torch.tensor(self.alpha_value, device=device, dtype=torch.float32)
            alpha_t = alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


class ObjectDetectionLoss(nn.Module):
    def __init__(self, processor, classes_alpha, eps=1e-7):
        super(ObjectDetectionLoss, self).__init__()
        self.processor = processor
        self.eps = eps
        self.siou_loss = SIoU(x1y1x2y2=False)
        self.binary_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.multi_focal_loss = FocalLoss(alpha=classes_alpha, gamma=2.0, num_classes=processor.num_classes)
    
    def forward(self, outputs, targets):
        """
        outputs: Tensor [B, N, 5 + C], N = number of predictions per image (all predictions concatenated, no grid)
        targets: Tensor [B, M, 5 + C], M = number of GT boxes per image
        Each bbox format assumed to be [xc, yc, w, h]
        """
        batch_size = outputs.shape[0]
        device = outputs.device
        
        total_siou = torch.zeros(1, device=device, requires_grad=True)
        total_obj = torch.zeros(1, device=device, requires_grad=True)
        total_cls = torch.zeros(1, device=device, requires_grad=True)

        for i in range(batch_size):
            pred_data = self.processor.convert_output_to_bboxes(outputs[i], grid=False, class_tensor=True)
            gt_data = self.processor.convert_output_to_bboxes(targets[i], grid=False, class_tensor=True)
            
            loss_dict = self._compute_single_image_loss_global(pred_data, gt_data, device)
            
            total_siou += loss_dict["siou"]
            total_obj += loss_dict["objectness"]
            total_cls += loss_dict["classification"]
        
        batch_size_tensor = torch.tensor(batch_size, device=device, dtype=torch.float32)
        total_loss = (total_siou + total_obj + total_cls) / batch_size_tensor
        return total_loss.squeeze()
    
    def _compute_single_image_loss_global(self, pred_data, gt_data, device):
        """
        Match predictions to ground truths globally (not grid-based).
        pred_data and gt_data are lists of dicts with keys:
            'bbox': tensor of shape [4]
            'conf': tensor scalar
            'class_tensor': tensor of shape [num_classes]
            'class_id': int
        """
        siou_losses = []
        obj_losses = []
        cls_losses = []
        
        if len(gt_data) == 0:
            # No GTs: all predictions should be negative (objectness=0)
            for pred in pred_data:
                pred_conf = pred['conf'].unsqueeze(0)
                neg_target = torch.zeros_like(pred_conf, device=device)
                obj_loss = self.binary_focal_loss(pred_conf, neg_target, device=device)
                obj_losses.append(obj_loss)
            return {
                "siou": siou_losses,
                "objectness": obj_losses,
                "classification": cls_losses
            }
        
        # Compute IoU matrix between all preds and GTs
        iou_matrix = torch.zeros(len(pred_data), len(gt_data), device=device)
        for i, pred in enumerate(pred_data):
            pred_bbox = pred['bbox'].to(device)
            for j, gt in enumerate(gt_data):
                gt_bbox = self._ensure_tensor(gt['bbox'], device)
                iou_matrix[i, j] = self._bbox_iou_differentiable(pred_bbox, gt_bbox)
        
        # Match predictions to GTs greedily (max IoU) with threshold
        matches = []
        used_preds = set()
        used_gts = set()
        iou_threshold = 0.2
        
        for _ in range(min(len(pred_data), len(gt_data))):
            max_iou = 0
            max_pred_idx = -1
            max_gt_idx = -1
            for pred_idx in range(len(pred_data)):
                if pred_idx in used_preds:
                    continue
                for gt_idx in range(len(gt_data)):
                    if gt_idx in used_gts:
                        continue
                    iou_val = iou_matrix[pred_idx, gt_idx]
                    if iou_val > max_iou:
                        max_iou = iou_val
                        max_pred_idx = pred_idx
                        max_gt_idx = gt_idx
            if max_iou >= iou_threshold:
                matches.append((max_pred_idx, max_gt_idx, max_iou))
                used_preds.add(max_pred_idx)
                used_gts.add(max_gt_idx)
            else:
                break
        
        # Process matched pairs
        for pred_idx, gt_idx, iou_val in matches:
            pred = pred_data[pred_idx]
            gt = gt_data[gt_idx]
            
            pred_bbox = pred['bbox'].to(device)
            gt_bbox = self._ensure_tensor(gt['bbox'], device)
            pred_conf = pred['conf'].unsqueeze(0).to(device)
            pred_class = pred['class_tensor'].to(device)
            
            siou_losses.append(self.siou_loss(pred_bbox, gt_bbox))
            obj_losses.append(self.binary_focal_loss(pred_conf, torch.ones_like(pred_conf, device=device), device))
            target_class = self._ensure_tensor([gt['class_id']], device, dtype=torch.long)
            cls_losses.append(self.multi_focal_loss(pred_class, target_class, device))
        
        # Process unmatched GTs (missed detections)
        unmatched_gts = [i for i in range(len(gt_data)) if i not in used_gts]
        unused_preds = [i for i in range(len(pred_data)) if i not in used_preds]
        for gt_idx in unmatched_gts:
            gt = gt_data[gt_idx]
            
            # Assign best unused prediction by IoU if available
            best_pred_idx = None
            best_iou = torch.tensor(0.0, device=device)
            for pred_idx in unused_preds:
                pred_bbox = pred_data[pred_idx]['bbox'].to(device)
                gt_bbox = self._ensure_tensor(gt['bbox'], device)
                iou_val = self._bbox_iou_differentiable(pred_bbox, gt_bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None:
                best_pred = pred_data[best_pred_idx]
                pred_bbox = best_pred['bbox'].to(device)
                gt_bbox = self._ensure_tensor(gt['bbox'], device)
                pred_conf = best_pred['conf'].unsqueeze(0).to(device)
                pred_class = best_pred['class_tensor'].to(device)
                
                siou_losses.append(self.siou_loss(pred_bbox, gt_bbox))
                soft_target = torch.clamp(best_iou, min=0.3, max=0.8)
                obj_losses.append(self.binary_focal_loss(pred_conf, soft_target.unsqueeze(0), device))
                target_class = self._ensure_tensor([gt['class_id']], device, dtype=torch.long)
                cls_losses.append(self.multi_focal_loss(pred_class, target_class, device))
                
                used_preds.add(best_pred_idx)
                unused_preds.remove(best_pred_idx)
            else:
                # No prediction available - penalty
                obj_losses.append(torch.tensor(2.0, device=device, requires_grad=True))
        
        # Process unmatched predictions (negatives)
        for pred_idx in range(len(pred_data)):
            if pred_idx not in used_preds:
                pred_conf = pred_data[pred_idx]['conf'].unsqueeze(0).to(device)
                obj_losses.append(self.binary_focal_loss(pred_conf, torch.zeros_like(pred_conf, device=device), device))
        
        total_siou_loss = sum(siou_losses) if siou_losses else torch.tensor(0.0, device=device, requires_grad=True)
        total_obj_loss = sum(obj_losses) if obj_losses else torch.tensor(0.0, device=device, requires_grad=True)
        total_cls_loss = sum(cls_losses) if cls_losses else torch.tensor(0.0, device=device, requires_grad=True)
        
        num_positive = len(siou_losses)
        if num_positive > 0:
            num_pos_tensor = torch.tensor(float(num_positive), device=device)
            total_siou_loss = total_siou_loss / num_pos_tensor
            total_cls_loss = total_cls_loss / num_pos_tensor
        
        return {
            "total": total_siou_loss + total_obj_loss + total_cls_loss,
            "siou": total_siou_loss,
            "objectness": total_obj_loss,
            "classification": total_cls_loss,
            "num_positive": num_positive
        }

    def _bbox_iou_differentiable(self, box1, box2):
        box1_xyxy = self._xywh_to_xyxy_differentiable(box1)
        box2_xyxy = self._xywh_to_xyxy_differentiable(box2)

        x1 = torch.max(box1_xyxy[0], box2_xyxy[0])
        y1 = torch.max(box1_xyxy[1], box2_xyxy[1])
        x2 = torch.min(box1_xyxy[2], box2_xyxy[2])
        y2 = torch.min(box1_xyxy[3], box2_xyxy[3])

        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - inter + self.eps
        return inter / union

    def _xywh_to_xyxy_differentiable(self, box):
        xc, yc, w, h = box[0], box[1], box[2], box[3]
        return torch.stack([
            xc - w / 2,
            yc - h / 2,
            xc + w / 2,
            yc + h / 2
        ])

    def _ensure_tensor(self, data, device, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        else:
            return torch.tensor(data, device=device, dtype=dtype)
