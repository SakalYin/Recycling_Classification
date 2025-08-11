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
        Grid-based object detection loss implementation.
        
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
        
        # Get grid dimensions from processor
        self.grid_size = processor.grid_size  # Assuming processor has grid_size attribute
        self.num_anchors = processor.num_anchors  # Assuming processor has num_anchors
    
    def forward(self, outputs, targets):
        """
        Grid-based loss calculation
        outputs: Tensor of shape [B, S, S, A, 5 + C]
        targets: Tensor of shape [B, S, S, A, 5 + C]
        """
        batch_size = outputs.shape[0]
        device = outputs.device
        
        # Initialize loss accumulators
        total_siou = torch.zeros(1, device=device, requires_grad=True)
        total_obj = torch.zeros(1, device=device, requires_grad=True)
        total_cls = torch.zeros(1, device=device, requires_grad=True)

        for i in range(batch_size):
            output = outputs[i]
            target = targets[i]

            # Convert to grid-based format with grid coordinates
            pred_data = self.processor.convert_yolo_output_to_bboxes(
                output, grid=True, class_tensor=True, is_training=True
            )
            gt_data = self.processor.convert_yolo_output_to_bboxes(
                target, grid=True, class_tensor=True, is_training=True
            )

            loss_dict = self._compute_single_image_loss_grid_based(pred_data, gt_data, device)
            
            total_siou = total_siou + loss_dict["siou"]
            total_obj = total_obj + loss_dict["objectness"]
            total_cls = total_cls + loss_dict["classification"]

        # Average over batch
        batch_size_tensor = torch.tensor(batch_size, device=device, dtype=torch.float32)
        total_loss = (total_siou + total_obj + total_cls) / batch_size_tensor
        return total_loss.squeeze()

    def _compute_single_image_loss_grid_based(self, pred_data, gt_data, device):
        """
        Compute loss for a single image using grid-based matching.
        Each prediction and ground truth has a 'grid' key indicating its grid cell (i, j).
        """
        # Organize data by grid cells for efficient matching
        pred_by_grid = self._organize_by_grid(pred_data)
        gt_by_grid = self._organize_by_grid(gt_data)
        
        # Initialize loss lists
        siou_losses = []
        obj_losses = []
        cls_losses = []
        
        # Get all unique grid cells that have either predictions or ground truths
        all_grids = set(pred_by_grid.keys()) | set(gt_by_grid.keys())
        
        for grid_coord in all_grids:
            grid_preds = pred_by_grid.get(grid_coord, [])
            grid_gts = gt_by_grid.get(grid_coord, [])
            
            # Compute loss for this specific grid cell
            grid_losses = self._compute_grid_cell_loss(grid_preds, grid_gts, grid_coord, device)
            
            # Accumulate losses
            siou_losses.extend(grid_losses["siou"])
            obj_losses.extend(grid_losses["objectness"])
            cls_losses.extend(grid_losses["classification"])
        
        # Aggregate losses
        total_siou_loss = sum(siou_losses) if siou_losses else torch.tensor(0.0, device=device, requires_grad=True)
        total_obj_loss = sum(obj_losses) if obj_losses else torch.tensor(0.0, device=device, requires_grad=True)
        total_cls_loss = sum(cls_losses) if cls_losses else torch.tensor(0.0, device=device, requires_grad=True)
        
        # Normalize regression and classification losses by number of positive samples
        num_positive = len([loss for loss in siou_losses if loss > 0])
        if num_positive > 0:
            num_positive_tensor = torch.tensor(float(num_positive), device=device)
            total_siou_loss = total_siou_loss / num_positive_tensor
            total_cls_loss = total_cls_loss / num_positive_tensor
        
        total_loss = total_siou_loss + total_obj_loss + total_cls_loss
        
        return {
            "total": total_loss,
            "siou": total_siou_loss,
            "objectness": total_obj_loss,
            "classification": total_cls_loss,
            "num_positive": num_positive,
            "num_grids_processed": len(all_grids)
        }

    def _organize_by_grid(self, data_list):
        """
        Organize predictions/ground truths by their grid coordinates.
        Returns: dict where key is (grid_i, grid_j) and value is list of objects in that cell
        """
        grid_dict = {}
        for item in data_list:
            grid_coord = item['grid']  # Should be tuple like (1, 2)
            if grid_coord not in grid_dict:
                grid_dict[grid_coord] = []
            grid_dict[grid_coord].append(item)
        return grid_dict

    def _compute_grid_cell_loss(self, grid_preds, grid_gts, grid_coord, device):
        """
        Compute loss for a specific grid cell.
        In YOLO, each grid cell is responsible for detecting objects whose center falls in that cell.
        """
        siou_losses = []
        obj_losses = []
        cls_losses = []
        
        if len(grid_gts) == 0:
            # No ground truth in this grid cell - all predictions should be negative
            for pred in grid_preds:
                pred_conf = pred['conf'].unsqueeze(0)
                neg_target = torch.zeros_like(pred_conf, device=device)
                obj_loss = self.binary_focal_loss(pred_conf, neg_target, device=device)
                obj_losses.append(obj_loss)
        else:
            # There are ground truths in this grid cell
            # Match predictions to ground truths using IoU
            matches = self._match_predictions_to_gts_in_grid(grid_preds, grid_gts, device)
            
            matched_preds = set()
            matched_gts = set()
            
            # Process positive matches
            for pred_idx, gt_idx, iou_val in matches:
                pred = grid_preds[pred_idx]
                gt = grid_gts[gt_idx]
                
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)
                
                # Compute losses for positive match
                pred_bbox = pred['bbox'].to(device)
                gt_bbox = self._ensure_tensor(gt['bbox'], device)
                pred_conf = pred['conf'].unsqueeze(0).to(device)
                pred_class = pred['class_tensor'].to(device)
                
                # Regression loss
                siou_loss = self.siou_loss(pred_bbox, gt_bbox)
                siou_losses.append(siou_loss)
                
                # Objectness loss (positive)
                obj_loss = self.binary_focal_loss(pred_conf, torch.ones_like(pred_conf, device=device), device)
                obj_losses.append(obj_loss)
                
                # Classification loss
                target_class = self._ensure_tensor([gt['class_id']], device, dtype=torch.long)
                cls_loss = self.multi_focal_loss(pred_class, target_class, device)
                cls_losses.append(cls_loss)
            
            # Process unmatched ground truths (missed detections in this grid)
            for gt_idx, gt in enumerate(grid_gts):
                if gt_idx not in matched_gts:
                    # Option 2: Assign to best available prediction (if any unused ones exist)
                    unused_preds = [i for i in range(len(grid_preds)) if i not in matched_preds]
                    if unused_preds:
                        # Find best unused prediction for this GT based on IoU
                        best_pred_idx = None
                        best_iou = torch.tensor(0.0, device=device)
                        
                        for pred_idx in unused_preds:
                            pred = grid_preds[pred_idx]
                            pred_bbox = pred['bbox'].to(device)
                            gt_bbox = self._ensure_tensor(gt['bbox'], device)
                            iou_val = self._bbox_iou_differentiable(pred_bbox, gt_bbox)
                            
                            if iou_val > best_iou:
                                best_iou = iou_val
                                best_pred_idx = pred_idx
                        
                        if best_pred_idx is not None:
                            best_pred = grid_preds[best_pred_idx]
                            
                            # Calculate losses
                            pred_bbox = best_pred['bbox'].to(device)
                            gt_bbox = self._ensure_tensor(gt['bbox'], device)
                            pred_conf = best_pred['conf'].unsqueeze(0).to(device)
                            pred_class = best_pred['class_tensor'].to(device)
                            
                            # SIoU loss
                            siou_loss = self.siou_loss(pred_bbox, gt_bbox)
                            siou_losses.append(siou_loss)
                            
                            # Objectness loss with soft target based on IoU
                            soft_target = torch.clamp(best_iou, min=0.3, max=0.8)
                            obj_loss = self.binary_focal_loss(pred_conf, soft_target.unsqueeze(0), device)
                            obj_losses.append(obj_loss)
                            
                            # Classification loss
                            target_class = self._ensure_tensor([gt['class_id']], device, dtype=torch.long)
                            cls_loss = self.multi_focal_loss(pred_class, target_class, device)
                            cls_losses.append(cls_loss)
                            
                            # Mark this prediction as assigned
                            matched_preds.add(best_pred_idx)
                        else:
                            # If no prediction available, add penalty
                            missed_penalty = torch.tensor(2.0, device=device, requires_grad=True)
                            obj_losses.append(missed_penalty)
                    else:
                        # No unused predictions available, add penalty
                        missed_penalty = torch.tensor(2.0, device=device, requires_grad=True)
                        obj_losses.append(missed_penalty)

            # Process negative predictions (unmatched predictions in this grid)
            # Only process predictions that were not matched AND not assigned in Option 2
            for pred_idx, pred in enumerate(grid_preds):
                if pred_idx not in matched_preds:
                    pred_conf = pred['conf'].unsqueeze(0).to(device)
                    neg_target = torch.zeros_like(pred_conf, device=device)
                    obj_loss = self.binary_focal_loss(pred_conf, neg_target, device)
                    obj_losses.append(obj_loss)
        
        return {
            "siou": siou_losses,
            "objectness": obj_losses,
            "classification": cls_losses
        }

    def _match_predictions_to_gts_in_grid(self, grid_preds, grid_gts, device):
        """
        Match predictions to ground truths within a single grid cell using IoU.
        Returns list of (pred_idx, gt_idx, iou_value) tuples for matches above threshold.
        """
        matches = []
        iou_threshold = 0.5  # Higher threshold since we're in the same grid cell
        
        if len(grid_preds) == 0 or len(grid_gts) == 0:
            return matches
        
        # Compute IoU matrix for this grid cell
        iou_matrix = torch.zeros(len(grid_preds), len(grid_gts), device=device)
        for i, pred in enumerate(grid_preds):
            for j, gt in enumerate(grid_gts):
                pred_bbox = pred['bbox'].to(device)
                gt_bbox = self._ensure_tensor(gt['bbox'], device)
                iou_matrix[i, j] = self._bbox_iou_differentiable(pred_bbox, gt_bbox)
        
        # Greedy matching: assign each GT to best prediction above threshold
        used_preds = set()
        for gt_idx in range(len(grid_gts)):
            best_iou = torch.tensor(0.0, device=device)
            best_pred_idx = -1
            
            for pred_idx in range(len(grid_preds)):
                if pred_idx not in used_preds:
                    iou_val = iou_matrix[pred_idx, gt_idx]
                    if iou_val > best_iou and iou_val > iou_threshold:
                        best_iou = iou_val
                        best_pred_idx = pred_idx
            
            if best_pred_idx >= 0:
                matches.append((best_pred_idx, gt_idx, best_iou))
                used_preds.add(best_pred_idx)
        
        return matches

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
        """
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        else:
            return torch.tensor(data, device=device, dtype=dtype)