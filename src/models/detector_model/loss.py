# from torchvision.ops import complete_box_iou_loss, sigmoid_focal_loss
# import torch
# import torch.nn as nn
# from munkres import Munkres
# import numpy as np

# class ObjectDetectionLoss(nn.Module):
#     def __init__(self, processor, classes_alpha=None, eps=1e-7):
#         super(ObjectDetectionLoss, self).__init__()
#         self.processor = processor
#         self.matching_algorithm = Munkres()
#         self.classes_alpha = classes_alpha
    
#     def forward(self, outputs, targets):
#         """
#         outputs: Tensor [B, N, 5 + C], N = number of predictions per image (all predictions concatenated, no grid)
#         targets: Tensor [B, M, 5 + C], M = number of GT boxes per image
#         Each bbox format assumed to be [xc, yc, w, h]
#         """
#         batch_size = outputs.shape[0]
#         device = outputs.device
        
#         total_siou = torch.zeros(1, device=device)
#         total_obj = torch.zeros(1, device=device)
#         total_cls = torch.zeros(1, device=device)

#         for i in range(batch_size):
#             pred_data = self.processor.convert_yolo_output_to_bboxes(outputs[i], grid=False, class_tensor=True, conf_threshold=None)
#             gt_data = self.processor.convert_yolo_output_to_bboxes(targets[i], grid=False, class_tensor=True, conf_threshold=0.5)
            
#             loss_dict = self._compute_single_image_loss_global(pred_data, gt_data, device)
#             total_siou += loss_dict["siou"]
#             total_obj += loss_dict["objectness"]
#             total_cls += loss_dict["classification"]
        
#         batch_size_tensor = torch.tensor(batch_size, device=device, dtype=torch.float32)
#         total_loss = (total_siou + total_obj + total_cls) / batch_size_tensor
#         return {
#             "total": total_loss.squeeze(),
#             "siou": total_siou / batch_size_tensor,
#             "objectness": total_obj / batch_size_tensor,
#             "classification": total_cls / batch_size_tensor,
#         }
    
#     def repeat_for_cartesian(self, pred_list, gt_list, device):
#         """
#         Given two lists of equal-length tensors (e.g., [x, y, w, h] or class logits),
#         create a Cartesian product repeat without breaking gradients.

#         Returns:
#             pred_repeat: [P*G, D] tensor (each pred repeated G times)
#             gt_repeat:   [P*G, D] tensor (GTs repeated P times)
#         """
#         if len(pred_list) == 0 or len(gt_list) == 0:
#             return torch.empty(0), torch.empty(0)

#         # Stack into tensors
#         pred_t = torch.stack(pred_list)  # [P, D]
#         gt_t   = torch.stack(gt_list)    # [G, D]

#         P, D = pred_t.size()
#         G = gt_t.size(0)

#         # Repeat for Cartesian pairing
#         pred_repeat = pred_t.unsqueeze(1).expand(P, G, D).reshape(-1, D)
#         gt_repeat   = gt_t.unsqueeze(0).expand(P, G, D).reshape(-1, D)

#         return pred_repeat, gt_repeat
    
#     def _compute_single_image_loss_global(self, pred_data, gt_data, device):
#         """
#         Match predictions to ground truths globally (not grid-based).
#         pred_data and gt_data are lists of dicts with keys:
#             'bbox': tensor of shape [4]
#             'conf': tensor scalar
#             'class_tensor': tensor of shape [num_classes]
#             'class_id': int
#         """
#         ciou_losses = []
#         obj_losses = []
#         cls_losses = []
#         P, G = None, None
        
#         if len(gt_data) == 0:
#             # print('No Groud Truths')
#             # No GTs: all predictions should be negative (objectness=0)
#             for pred in pred_data:
#                 pred_conf = pred['conf']
#                 obj_losses.append(sigmoid_focal_loss(pred_conf, torch.zeros_like(pred_conf, device=device))*torch.tensor(1.5, device=device))
        
#         else:
#             # Compute CIoU + Classification Cost matrix between all preds and GTs
#             pred_bbox = [pred['bbox'] for pred in pred_data]
#             gt_bbox = [gt['bbox'] for gt in gt_data]
#             pred_class = [pred['class_tensor'] for pred in pred_data]
#             gt_class = [gt['class_tensor'] for gt in gt_data]
#             pred_conf = [pred['conf'] for pred in pred_data]
#             gt_conf = [gt['conf'] for gt in gt_data]

#             P = len(pred_bbox)
#             G = len(gt_bbox)

#             pred_bbox_repeat, gt_bbox_repeat = self.repeat_for_cartesian(pred_bbox, gt_bbox, device=device)
#             pred_class_repeat, gt_class_repeat = self.repeat_for_cartesian(pred_class, gt_class, device=device)

#             CIoU_matrix = complete_box_iou_loss(pred_bbox_repeat, gt_bbox_repeat).view(P, G)
#             Cls_matrix = sigmoid_focal_loss(pred_class_repeat, gt_class_repeat).mean(dim=1).view(P, G)
#             cost_matrix = CIoU_matrix + Cls_matrix
#             cost_matrix = cost_matrix.clone().detach().cpu().numpy().tolist()
            
#             matches = self.matching_algorithm.compute(cost_matrix)
#             used_pred_idx, used_gt_idx = [], []

#             for pred_idx, gt_idx in matches:
#                 if pred_idx >= P or gt_idx >= G:
#                     print(f"Invalid match: pred_idx={pred_idx}, gt_idx={gt_idx}, P={P}, G={G}")
#                     continue
#                 ciou_losses.append(complete_box_iou_loss(pred_bbox[pred_idx], gt_bbox[gt_idx].to(device)))
#                 cls_losses.append(sigmoid_focal_loss(pred_class[pred_idx], gt_class[gt_idx].to(device)).mean())
#                 obj_losses.append(sigmoid_focal_loss(pred_conf[pred_idx], gt_conf[gt_idx].to(device)))
#                 used_pred_idx.append(pred_idx)
#                 used_gt_idx.append(gt_idx)
            
#             # Process unmatched predictions
#             unmatched_pred = [idx for idx in np.arange(P) if idx not in used_pred_idx]
#             if len(unmatched_pred) > 0:
#                 for i in unmatched_pred:
#                     obj_losses.append(sigmoid_focal_loss(pred_conf[i], torch.zeros_like(pred_conf[i], device=device))*torch.tensor(1.5, device=device))

#         total_siou_loss = sum(ciou_losses) if len(ciou_losses) > 0 and G else torch.tensor(0.0, device=device, requires_grad=True)
#         total_obj_loss = sum(obj_losses) if len(obj_losses) > 0 else torch.tensor(0.0, device=device, requires_grad=True)
#         total_cls_loss = sum(cls_losses) if len(cls_losses) > 0 and G else torch.tensor(0.0, device=device, requires_grad=True)

#         total_obj_loss = total_obj_loss / torch.tensor(float(len(pred_data)), device=device) 
#         total_siou_loss = total_siou_loss / torch.tensor(float(len(used_pred_idx)), device=device) if G else torch.tensor(0.0, device=device, requires_grad=True)
#         total_cls_loss = total_cls_loss / torch.tensor(float(len(used_pred_idx)), device=device) if G else torch.tensor(0.0, device=device, requires_grad=True)

#         return {
#             "total": total_siou_loss + total_obj_loss + total_cls_loss,
#             "siou": total_siou_loss,
#             "objectness": total_obj_loss,
#             "classification": total_cls_loss,
#         }

#     def _bbox_iou_differentiable(self, box1, box2):
#         box1_xyxy = self._xywh_to_xyxy_differentiable(box1)
#         box2_xyxy = self._xywh_to_xyxy_differentiable(box2)

#         x1 = torch.max(box1_xyxy[0], box2_xyxy[0])
#         y1 = torch.max(box1_xyxy[1], box2_xyxy[1])
#         x2 = torch.min(box1_xyxy[2], box2_xyxy[2])
#         y2 = torch.min(box1_xyxy[3], box2_xyxy[3])

#         inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
#         area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
#         area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
#         union = area1 + area2 - inter + self.eps
#         return inter / union

#     def _xywh_to_xyxy_differentiable(self, box):
#         xc, yc, w, h = box[0], box[1], box[2], box[3]
#         return torch.stack([
#             xc - w / 2,
#             yc - h / 2,
#             xc + w / 2,
#             yc + h / 2
#         ])

#     def _ensure_tensor(self, data, device, dtype=torch.float32):
#         if isinstance(data, torch.Tensor):
#             return data.to(device=device, dtype=dtype)
#         else:
#             return torch.tensor(data, device=device, dtype=dtype)

from torchvision.ops import complete_box_iou_loss, sigmoid_focal_loss
import torch
import torch.nn as nn
from munkres import Munkres
import numpy as np

class ObjectDetectionLoss(nn.Module):
    def __init__(self, processor, classes_alpha=None, eps=1e-7, obj_loss_weight=2.0, 
                 cls_loss_weight=2.0, bbox_loss_weight=2.0, neg_obj_weight=1.0):
        super(ObjectDetectionLoss, self).__init__()
        self.processor = processor
        self.matching_algorithm = Munkres()
        self.classes_alpha = classes_alpha
        self.eps = eps
        self.obj_loss_weight = obj_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.neg_obj_weight = neg_obj_weight  # Weight for negative objectness loss
    
    def forward(self, outputs, targets):
        """
        outputs: Tensor [B, N, 5 + C], N = number of predictions per image
        targets: Tensor [B, M, 5 + C], M = number of GT boxes per image
        """
        batch_size = outputs.shape[0]
        device = outputs.device
        
        total_siou = 0.0
        total_obj = 0.0
        total_cls = 0.0
        valid_samples = 0

        for i in range(batch_size):
            try:
                pred_data = self.processor.convert_yolo_output_to_bboxes(
                    outputs[i], grid=False, class_tensor=True, conf_threshold=None
                )
                gt_data = self.processor.convert_yolo_output_to_bboxes(
                    targets[i], grid=False, class_tensor=True, conf_threshold=0.5
                )
                
                loss_dict = self._compute_single_image_loss_global(pred_data, gt_data, device)
                total_siou += loss_dict["siou"]
                total_obj += loss_dict["objectness"]
                total_cls += loss_dict["classification"]
                valid_samples += 1
                
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                continue
        
        if valid_samples == 0:
            # Return zero losses if no valid samples
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "total": zero_loss,
                "siou": zero_loss,
                "objectness": zero_loss,
                "classification": zero_loss,
            }
        
        # Average over valid samples
        avg_siou = total_siou / valid_samples
        avg_obj = total_obj / valid_samples  
        avg_cls = total_cls / valid_samples
        
        # Apply loss weights
        weighted_total = (
            self.bbox_loss_weight * avg_siou + 
            self.obj_loss_weight * avg_obj + 
            self.cls_loss_weight * avg_cls
        )
        
        return {
            "total": weighted_total,
            "siou": avg_siou,
            "objectness": avg_obj,
            "classification": avg_cls,
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
    
    def _compute_single_image_loss_global(self, pred_data, gt_data, device):
        """Match predictions to ground truths globally."""
        
        # Initialize losses as tensors
        siou_loss = torch.tensor(0.0, device=device, requires_grad=True)
        obj_loss = torch.tensor(0.0, device=device, requires_grad=True) 
        cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        num_preds = len(pred_data)
        num_gts = len(gt_data)
        
        if num_preds == 0:
            return {"siou": siou_loss, "objectness": obj_loss, "classification": cls_loss}
        
        if num_gts == 0:
            # No ground truths: all predictions should have objectness = 0
            negative_obj_losses = []
            for pred in pred_data:
                pred_conf = pred['conf']
                target_conf = torch.zeros_like(pred_conf)
                neg_loss = sigmoid_focal_loss(
                    pred_conf.unsqueeze(0), target_conf.unsqueeze(0), 
                    reduction='mean'
                )
                negative_obj_losses.append(neg_loss)
            
            if negative_obj_losses:
                obj_loss = torch.stack(negative_obj_losses).mean() * self.neg_obj_weight
            
            return {"siou": siou_loss, "objectness": obj_loss, "classification": cls_loss}
        
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
                cls_matrix = sigmoid_focal_loss(pred_class_repeat, gt_class_repeat, reduction='none').mean(dim=1).view(num_preds, num_gts)
                
                cost_matrix = (ciou_matrix + cls_matrix).detach().cpu().numpy()
                
                # Hungarian matching
                matches = self.matching_algorithm.compute(cost_matrix.tolist())
                
                # Compute losses for matched pairs
                matched_pred_idx = []
                matched_gt_idx = []
                
                ciou_losses = []
                cls_losses = []
                obj_losses = []
                
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
                            reduction='mean'
                        )
                        cls_losses.append(cls)
                        
                        # Objectness loss (positive)
                        obj = sigmoid_focal_loss(
                            pred_conf[pred_idx].unsqueeze(0), 
                            gt_conf[gt_idx].unsqueeze(0), 
                            reduction='mean'
                        )
                        obj_losses.append(obj)
                        
                        matched_pred_idx.append(pred_idx)
                        matched_gt_idx.append(gt_idx)
                
                # Compute losses for unmatched predictions (negative objectness)
                unmatched_pred_idx = [i for i in range(num_preds) if i not in matched_pred_idx]
                for pred_idx in unmatched_pred_idx:
                    target_conf = torch.zeros_like(pred_conf[pred_idx])
                    neg_obj = sigmoid_focal_loss(
                        pred_conf[pred_idx].unsqueeze(0), 
                        target_conf.unsqueeze(0), 
                        reduction='mean'
                    ) * self.neg_obj_weight
                    obj_losses.append(neg_obj)
                
                # Aggregate losses
                if ciou_losses:
                    siou_loss = torch.stack(ciou_losses).mean()
                if cls_losses:
                    cls_loss = torch.stack(cls_losses).mean()
                if obj_losses:
                    obj_loss = torch.stack(obj_losses).mean()
                    
        except Exception as e:
            print(f"Error in loss computation: {e}")
            # Return zero losses on error
            pass
        
        return {
            "siou": siou_loss,
            "objectness": obj_loss, 
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