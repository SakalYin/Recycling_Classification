from torchvision.ops import complete_box_iou_loss, sigmoid_focal_loss
import torch
import torch.nn as nn
from munkres import Munkres
import numpy as np

class ObjectDetectionLoss(nn.Module):
    def __init__(self, processor, alpha=None, eps=1e-7, obj_loss_weight=1.5, 
                 cls_loss_weight=1.5, bbox_loss_weight=0.5, neg_obj_weight=0.5, pos_obj_weight=1.5):
        super(ObjectDetectionLoss, self).__init__()
        self.processor = processor
        self.matching_algorithm = Munkres()
        self.alpha = alpha
        self.eps = eps
        self.obj_loss_weight = obj_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.neg_obj_weight = neg_obj_weight  # Weight for negative objectness loss
        self.pos_obj_weight = pos_obj_weight
    
    def forward(self, outputs, targets):
        """
        outputs: Tensor [B, N, 5 + C], N = number of predictions per image
        targets: Tensor [B, M, 5 + C], M = number of GT boxes per image
        """
        batch_size = outputs.shape[0]
        device = outputs.device
        
        total_ciou = 0.0
        total_obj = 0.0
        total_cls = 0.0
        total_positive_obj_loss = []
        valid_samples = 0

        for i in range(batch_size):
            try:
                pred_data = self.processor.convert_output_to_bboxes(
                    outputs[i], grid=True, class_tensor=True, conf_threshold=None
                )
                gt_data = self.processor.convert_output_to_bboxes(
                    targets[i], grid=True, class_tensor=True, conf_threshold=0.5
                )
                
                loss_dict = self._compute_single_image_loss(pred_data, gt_data, device)
                total_ciou += loss_dict["ciou"]
                total_obj += loss_dict["objectness"]
                total_cls += loss_dict["classification"]
                
                # Safe append for positive objectness loss
                if "positive_objectness" in loss_dict and loss_dict["positive_objectness"] is not None:
                    total_positive_obj_loss.append(loss_dict['positive_objectness'])

                valid_samples += 1
                
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                continue
        
        if valid_samples == 0:
            # Return zero losses if no valid samples
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "total": zero_loss,
                "ciou": zero_loss,
                "objectness": zero_loss,
                "classification": zero_loss,
                "positive_objectness": zero_loss
            }
        
        # Average over valid samples
        avg_ciou = total_ciou / valid_samples
        avg_obj = total_obj / valid_samples  
        avg_cls = total_cls / valid_samples
        
        # Handle positive objectness loss safely
        if total_positive_obj_loss:
            avg_pos_obj = torch.stack(total_positive_obj_loss).mean()
        else:
            avg_pos_obj = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Apply loss weights
        weighted_total = (
            self.bbox_loss_weight * avg_ciou + 
            self.obj_loss_weight * avg_obj + 
            self.cls_loss_weight * avg_cls
        )
        
        return {
            "total": weighted_total,
            "ciou": avg_ciou * self.bbox_loss_weight,
            "objectness": avg_obj * self.obj_loss_weight,
            "classification": avg_cls * self.cls_loss_weight,
            "positive_objectness": avg_pos_obj
        }
    
    def repeat_for_cartesian(self, pred_list, gt_list, device):
        """Create Cartesian product repeat without breaking gradients."""
        if len(pred_list) == 0 or len(gt_list) == 0:
            return torch.empty((0, pred_list[0].shape[0]) if pred_list else (0, 0), device=device), \
                   torch.empty((0, gt_list[0].shape[0]) if gt_list else (0, 0), device=device)

        pred_t = torch.stack(pred_list)  # [P, D]
        gt_t = torch.stack(gt_list)      # [G, D]

        P, D = pred_t.size()
        G = gt_t.size(0)

        pred_repeat = pred_t.unsqueeze(1).expand(P, G, D).reshape(-1, D)
        gt_repeat = gt_t.unsqueeze(0).expand(P, G, D).reshape(-1, D)

        return pred_repeat, gt_repeat
    
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
    
    def _compute_single_image_loss(self, pred_data, gt_data, device):
        """Match predictions to ground truths in an image."""
        pred_by_grid = self._organize_by_grid(pred_data)
        gt_by_grid = self._organize_by_grid(gt_data)
        all_grids = set(pred_by_grid.keys()) | set(gt_by_grid.keys())

        ciou_losses = []
        cls_losses = []
        obj_losses = []
        pos_obj_losses = []

        for grid_coord in all_grids:
            grid_preds = pred_by_grid.get(grid_coord, [])
            grid_gts = gt_by_grid.get(grid_coord, [])

            grid_losses = self._compute_single_grid_loss(grid_preds, grid_gts, device)
            ciou_losses.append(grid_losses['ciou'])
            cls_losses.append(grid_losses['classification'])
            obj_losses.append(grid_losses['objectness'])
            
            # Safe append for positive objectness loss
            if 'positive_objectness' in grid_losses and grid_losses['positive_objectness'] is not None:
                pos_obj_losses.append(grid_losses['positive_objectness'])

        # Safe stacking with empty list checks
        ciou_loss = torch.stack(ciou_losses).mean() if ciou_losses else torch.tensor(0.0, device=device, requires_grad=True)
        obj_loss = torch.stack(obj_losses).mean() if obj_losses else torch.tensor(0.0, device=device, requires_grad=True)
        cls_loss = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device, requires_grad=True)
        pos_obj_loss = torch.stack(pos_obj_losses).mean() if pos_obj_losses else None

        return {
            "ciou": ciou_loss,
            "objectness": obj_loss, 
            "positive_objectness": pos_obj_loss,
            "classification": cls_loss,
        }
    
    def _compute_single_grid_loss(self, pred_data, gt_data, device):
        """Match predictions to ground truths per grid."""
        
        # Initialize losses as tensors
        ciou_loss = torch.tensor(0.0, device=device, requires_grad=True)
        obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
        pos_obj_loss = None  # Will be set only if there are positive matches
        cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        num_preds = len(pred_data)
        num_gts = len(gt_data)
        
        if num_preds == 0:
            return {"ciou": ciou_loss, "objectness": obj_loss, "classification": cls_loss, "positive_objectness": pos_obj_loss}
        
        if num_gts == 0:
            # No ground truths: all predictions should have objectness = 0
            negative_obj_losses = []
            for pred in pred_data:
                pred_conf = pred['conf']
                target_conf = torch.zeros_like(pred_conf)
                neg_loss = sigmoid_focal_loss(
                    pred_conf.unsqueeze(0), target_conf.unsqueeze(0), 
                    reduction='mean',
                    alpha= 0.5,
                    gamma=1.5
                )
                negative_obj_losses.append(neg_loss)
            
            if negative_obj_losses:
                obj_loss = torch.stack(negative_obj_losses).mean() * self.neg_obj_weight
            
            return {"ciou": ciou_loss, "objectness": obj_loss, "classification": cls_loss, "positive_objectness": pos_obj_loss}
        
        # Extract data for matching
        pred_bbox = [pred['bbox'] for pred in pred_data]
        gt_bbox = [gt['bbox'] for gt in gt_data]
        pred_class = [pred['class_tensor'] for pred in pred_data]
        gt_class = [gt['class_tensor'] for gt in gt_data]
        pred_conf = [pred['conf'] for pred in pred_data]
        
        # Create ground truth confidence (should be 1.0 for positive samples)
        gt_conf = [torch.ones_like(pred_conf[0]) for _ in range(num_gts)]
        
        # Compute cost matrix
        try:
            pred_bbox_repeat, gt_bbox_repeat = self.repeat_for_cartesian(pred_bbox, gt_bbox, device)
            pred_class_repeat, gt_class_repeat = self.repeat_for_cartesian(pred_class, gt_class, device)
            
            # Ensure proper format for complete_box_iou_loss (needs [N, 4] tensors)
            if pred_bbox_repeat.numel() > 0 and gt_bbox_repeat.numel() > 0:
                # Convert to xyxy format if needed
                pred_xyxy = self._convert_to_xyxy(pred_bbox_repeat)
                gt_xyxy = self._convert_to_xyxy(gt_bbox_repeat)
                
                ciou_matrix = complete_box_iou_loss(pred_xyxy, gt_xyxy, reduction='none').view(num_preds, num_gts)
                cls_matrix = sigmoid_focal_loss(pred_class_repeat, gt_class_repeat, reduction='none', alpha= 0.5, gamma=0.0).mean(dim=1).view(num_preds, num_gts)
                
                cost_matrix = (ciou_matrix + cls_matrix).detach().cpu().numpy()
                
                # Hungarian matching
                matches = self.matching_algorithm.compute(cost_matrix.tolist())
                
                # Compute losses for matched pairs
                matched_pred_idx = []
                matched_gt_idx = []

                num_pos = len(matches)
                num_neg = len(pred_data) - num_pos
                alpha_positive = num_neg / (num_pos + num_neg) if (num_pos + num_neg) > 0 else 0.5
                alpha_positive = max(min(alpha_positive, 0.85), 0.5) 
                
                ciou_losses = []
                cls_losses = []
                obj_losses = []
                pos_obj_loss_list = []
                
                for pred_idx, gt_idx in matches:
                    if pred_idx < num_preds and gt_idx < num_gts:
                        # CIOU loss
                        pred_box_xyxy = self._convert_to_xyxy(pred_bbox[pred_idx].unsqueeze(0))
                        gt_box_xyxy = self._convert_to_xyxy(gt_bbox[gt_idx].unsqueeze(0))
                        ciou = complete_box_iou_loss(pred_box_xyxy, gt_box_xyxy, reduction='mean')
                        ciou_losses.append(ciou)
                        
                        # Classification loss
                        cls = sigmoid_focal_loss(
                            pred_class[pred_idx].unsqueeze(0), 
                            gt_class[gt_idx].unsqueeze(0), 
                            alpha= 0.5,
                            gamma=0.0
                        ) 
                        cls_losses.append(cls)
                        
                        # Objectness loss (positive)
                        obj = sigmoid_focal_loss(
                            pred_conf[pred_idx].unsqueeze(0), 
                            gt_conf[gt_idx].unsqueeze(0), 
                            reduction='mean',
                            alpha= alpha_positive,
                            gamma=0.0
                        ) * self.pos_obj_weight
                        obj_losses.append(obj)
                        pos_obj_loss_list.append(obj)
                        
                        matched_pred_idx.append(pred_idx)
                        matched_gt_idx.append(gt_idx)
                
                # Compute losses for unmatched predictions (negative objectness)
                unmatched_pred_idx = [i for i in range(num_preds) if i not in matched_pred_idx]
                for pred_idx in unmatched_pred_idx:
                    target_conf = torch.zeros_like(pred_conf[pred_idx])
                    neg_obj = sigmoid_focal_loss(
                        pred_conf[pred_idx].unsqueeze(0), 
                        target_conf.unsqueeze(0), 
                        reduction ='mean',
                        alpha = 0.5,
                        gamma = 1.5
                    ) * self.neg_obj_weight
                    obj_losses.append(neg_obj)
                
                # Aggregate losses with safe stacking
                if ciou_losses:
                    ciou_loss = torch.stack(ciou_losses).mean()
                if cls_losses:
                    cls_loss = torch.stack(cls_losses).mean()
                if obj_losses:
                    obj_loss = torch.stack(obj_losses).mean()
                if pos_obj_loss_list:
                    pos_obj_loss = torch.stack(pos_obj_loss_list).mean()
                    
        except Exception as e:
            print(f"Error in loss computation: {e}")
            # Return zero losses on error
            pass
        
        return {
            "ciou": ciou_loss,
            "objectness": obj_loss, 
            "positive_objectness": pos_obj_loss,
            "classification": cls_loss,
        }
    
    def _convert_to_xyxy(self, xywh_boxes):
        """Convert from [xc, yc, w, h] to [x1, y1, x2, y2] format."""
        if xywh_boxes.numel() == 0:
            return xywh_boxes
            
        xc, yc, w, h = xywh_boxes[..., 0], xywh_boxes[..., 1], xywh_boxes[..., 2], xywh_boxes[..., 3]
        x1 = xc - w / 2
        y1 = yc - h / 2  
        x2 = xc + w / 2
        y2 = yc + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)