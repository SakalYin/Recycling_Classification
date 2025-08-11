import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SIoU(nn.Module):
    """
    Smart Intersection over Union (SIoU) Loss
    Considers angle, distance, and shape costs for better bounding box regression
    """
    def __init__(self, x1y1x2y2=True, eps=1e-7):
        super(SIoU, self).__init__()
        self.x1y1x2y2 = x1y1x2y2
        self.eps = eps
  
    def forward(self, box1, box2):
        # Get the coordinates of bounding boxes
        if self.x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
            b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
            b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
            b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + self.eps
    
        # IoU value of the bounding boxes
        iou = inter / union
        
        # Convex (smallest enclosing box) dimensions
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        
        # Center distance components
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        
        # Angle cost calculation
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5) + self.eps
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            
        # Distance Cost
        rho_x = (s_cw / (cw + self.eps)) ** 2
        rho_y = (s_ch / (ch + self.eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            
        # Shape Cost
        omiga_w = torch.abs(w1 - w2) / (torch.max(w1, w2) + self.eps)
        omiga_h = torch.abs(h1 - h2) / (torch.max(h1, h2) + self.eps)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        
        return 1 - (iou + 0.5 * (distance_cost + shape_cost))

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Supports both binary and multi-class classification
    """
    def __init__(self, alpha=None, gamma=2, num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
        if num_classes is None:
            # Binary classification
            self.alpha_value = alpha if alpha is not None else 0.25
        else:
            # Multi-class classification
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
            alpha_t = self.alpha_value * targets + (1 - self.alpha_value) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
    
    def _multiclass_focal_loss(self, inputs, targets, device):
        # Ensure targets are in correct format
        if targets.dim() > 1:
            if targets.shape[1] > 1:  # One-hot encoded
                targets = torch.argmax(targets, dim=1)
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)

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

# Usage example and testing utilities
class LossMonitor:
    """Utility class to monitor and log loss components during training"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = {
            'total': [],
            'box': [],
            'obj': [],
            'cls': []
        }
    
    def update(self, loss_dict):
        for key, value in loss_dict.items():
            if key.endswith('_loss'):
                loss_type = key.replace('_loss', '')
                if loss_type in self.losses:
                    self.losses[loss_type].append(value.item())
    
    def get_averages(self):
        return {k: np.mean(v) if v else 0.0 for k, v in self.losses.items()}
    
    def print_summary(self):
        avgs = self.get_averages()
        print(f"Loss Summary - Total: {avgs['total']:.4f}, "
              f"Box: {avgs['box']:.4f}, Obj: {avgs['obj']:.4f}, Cls: {avgs['cls']:.4f}")


# Example usage
def create_detection_loss(processor, num_classes=80):
    """Factory function to create detection loss with reasonable defaults"""
    # Class-balanced alpha values (can be tuned based on dataset)
    classes_alpha = [1.0] * num_classes  # Equal weight for all classes
    
    # Loss component weights
    loss_weights = {
        'box': 5.0,  # Higher weight for box regression
        'obj': 1.0,  # Standard weight for objectness
        'cls': 0.5   # Lower weight for classification
    }
    
    return ObjectDetectionLoss(
        processor=processor,
        classes_alpha=classes_alpha,
        loss_weights=loss_weights
    )

import matplotlib.pyplot as plt
from collections import defaultdict

class LossDiagnostic:
    """Comprehensive loss debugging and diagnostic tool"""
    
    def __init__(self, loss_fn, processor):
        self.loss_fn = loss_fn
        self.processor = processor
        self.history = defaultdict(list)
        self.detailed_stats = defaultdict(list)
        
    def diagnose_batch(self, outputs, targets, step=None):
        """Perform detailed diagnosis of a single batch"""
        print(f"\n{'='*60}")
        print(f"LOSS DIAGNOSTIC - Step {step if step else 'N/A'}")
        print(f"{'='*60}")
        
        # Basic tensor info
        self._check_tensor_properties(outputs, targets)
        
        # Gradient flow check
        self._check_gradients(outputs, targets)
        
        # Loss component analysis
        loss_breakdown = self._analyze_loss_components(outputs, targets)
        
        # Data distribution analysis
        self._analyze_data_distribution(outputs, targets)
        
        # Learning dynamics
        self._check_learning_dynamics(outputs, targets)
        
        return loss_breakdown
    
    def _check_tensor_properties(self, outputs, targets):
        """Check basic tensor properties"""
        print("\n1. TENSOR PROPERTIES:")
        print(f"   Outputs shape: {outputs.shape}")
        print(f"   Targets shape: {targets.shape}")
        print(f"   Outputs dtype: {outputs.dtype}, device: {outputs.device}")
        print(f"   Targets dtype: {targets.dtype}, device: {targets.device}")
        
        # Check for NaN/Inf
        outputs_nan = torch.isnan(outputs).sum().item()
        outputs_inf = torch.isinf(outputs).sum().item()
        targets_nan = torch.isnan(targets).sum().item()
        targets_inf = torch.isinf(targets).sum().item()
        
        print(f"   Outputs - NaN: {outputs_nan}, Inf: {outputs_inf}")
        print(f"   Targets - NaN: {targets_nan}, Inf: {targets_inf}")
        
        if outputs_nan > 0 or outputs_inf > 0:
            print("   ⚠️  WARNING: Outputs contain NaN/Inf values!")
        if targets_nan > 0 or targets_inf > 0:
            print("   ⚠️  WARNING: Targets contain NaN/Inf values!")
    
    def _check_gradients(self, outputs, targets):
        """Check gradient flow"""
        print("\n2. GRADIENT FLOW CHECK:")
        
        # Ensure outputs require grad
        if not outputs.requires_grad:
            outputs.requires_grad_(True)
            print("   ⚠️  Enabled gradients for outputs")
        
        # Compute loss and backward
        try:
            loss_dict = self.loss_fn(outputs, targets)
            total_loss = loss_dict.get('total_loss', loss_dict)
            
            if torch.is_tensor(total_loss):
                total_loss.backward(retain_graph=True)
                
                if outputs.grad is not None:
                    grad_norm = outputs.grad.norm().item()
                    grad_max = outputs.grad.abs().max().item()
                    grad_mean = outputs.grad.mean().item()
                    
                    print(f"   Gradient norm: {grad_norm:.6f}")
                    print(f"   Gradient max: {grad_max:.6f}")
                    print(f"   Gradient mean: {grad_mean:.6f}")
                    
                    if grad_norm < 1e-7:
                        print("   ⚠️  WARNING: Very small gradients - possible vanishing gradient")
                    elif grad_norm > 100:
                        print("   ⚠️  WARNING: Very large gradients - possible exploding gradient")
                else:
                    print("   ❌ ERROR: No gradients computed!")
            else:
                print("   ❌ ERROR: Loss is not a tensor!")
                
        except Exception as e:
            print(f"   ❌ ERROR in gradient computation: {e}")
    
    def _analyze_loss_components(self, outputs, targets):
        """Detailed loss component analysis"""
        print("\n3. LOSS COMPONENT ANALYSIS:")
        
        try:
            loss_dict = self.loss_fn(outputs, targets)
            
            if isinstance(loss_dict, dict):
                for key, value in loss_dict.items():
                    if torch.is_tensor(value):
                        val = value.item()
                        print(f"   {key}: {val:.6f}")
                        self.history[key].append(val)
                    else:
                        print(f"   {key}: {value}")
            else:
                # Single loss value
                val = loss_dict.item() if torch.is_tensor(loss_dict) else loss_dict
                print(f"   Total Loss: {val:.6f}")
                self.history['total_loss'].append(val)
            
            return loss_dict
            
        except Exception as e:
            print(f"   ❌ ERROR in loss computation: {e}")
            return None
    
    def _analyze_data_distribution(self, outputs, targets):
        """Analyze data distribution"""
        print("\n4. DATA DISTRIBUTION:")
        
        # Outputs analysis
        out_min, out_max = outputs.min().item(), outputs.max().item()
        out_mean, out_std = outputs.mean().item(), outputs.std().item()
        print(f"   Outputs - Min: {out_min:.4f}, Max: {out_max:.4f}, Mean: {out_mean:.4f}, Std: {out_std:.4f}")
        
        # Targets analysis
        tgt_min, tgt_max = targets.min().item(), targets.max().item()
        tgt_mean, tgt_std = targets.mean().item(), targets.std().item()
        print(f"   Targets - Min: {tgt_min:.4f}, Max: {tgt_max:.4f}, Mean: {tgt_mean:.4f}, Std: {tgt_std:.4f}")
        
        # Check if outputs are in reasonable range
        if out_max > 1000 or out_min < -1000:
            print("   ⚠️  WARNING: Outputs have extreme values!")
        
        # Check target sparsity (for object detection)
        nonzero_targets = (targets > 0.01).float().mean().item()
        print(f"   Target sparsity: {nonzero_targets:.4f} (fraction of non-zero elements)")
        
        if nonzero_targets < 0.001:
            print("   ⚠️  WARNING: Very sparse targets - might indicate labeling issues!")
    
    def _check_learning_dynamics(self, outputs, targets):
        """Check learning dynamics"""
        print("\n5. LEARNING DYNAMICS:")
        
        # Check if loss is changing
        if len(self.history.get('total_loss', [])) > 10:
            recent_losses = self.history['total_loss'][-10:]
            loss_variance = np.var(recent_losses)
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            print(f"   Recent loss variance: {loss_variance:.8f}")
            print(f"   Loss trend (slope): {loss_trend:.8f}")
            
            if loss_variance < 1e-8:
                print("   ⚠️  WARNING: Loss not changing - possible convergence to local minimum!")
            
            if abs(loss_trend) < 1e-6:
                print("   ⚠️  WARNING: No learning trend detected!")
    
    def plot_loss_history(self, save_path=None):
        """Plot loss history"""
        if not self.history:
            print("No loss history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, (key, values) in enumerate(self.history.items()):
            if i >= 4:
                break
            axes[i].plot(values)
            axes[i].set_title(f'{key} over time')
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel('Loss')
            axes[i].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


class ObjectDetectionLoss(nn.Module):
    """Fixed version of ObjectDetectionLoss with better numerical stability"""
    
    def __init__(self, processor, classes_alpha=None, loss_weights=None, eps=1e-7):
        super(ObjectDetectionLoss, self).__init__()
        self.processor = processor
        self.eps = eps
        
        # Loss components
        self.siou_loss = SIoU(x1y1x2y2=False, eps=eps)
        self.binary_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.multi_focal_loss = FocalLoss(
            alpha=classes_alpha, 
            gamma=2.0, 
            num_classes=getattr(processor, 'num_classes', None)
        )
        
        # Improved loss weights - reduced to prevent explosion
        if loss_weights is None:
            loss_weights = {'box': 1.0, 'obj': 1.0, 'cls': 1.0}  # Balanced weights
        self.loss_weights = loss_weights
        
        # More conservative thresholds
        self.pos_iou_thresh = 0.5
        self.neg_iou_thresh = 0.3
        
        # Add loss scaling to prevent numerical issues
        self.loss_scale = 1.0
        
    def forward(self, outputs, targets):
        """Forward pass with extensive error checking"""
        # Input validation
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("WARNING: NaN/Inf detected in outputs")
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("WARNING: NaN/Inf detected in targets")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        batch_size = outputs.shape[0]
        device = outputs.device
        
        # Initialize loss accumulators
        total_siou = torch.tensor(0.0, device=device, requires_grad=True)
        total_obj = torch.tensor(0.0, device=device, requires_grad=True)
        total_cls = torch.tensor(0.0, device=device, requires_grad=True)
        
        valid_samples = 0
        
        for i in range(batch_size):
            # try:
            # Convert to interpretable format
            pred_data = self.processor.convert_yolo_output_to_bboxes(
                outputs[i], grid=True, class_tensor=True, is_training=True
            )
            gt_data = self.processor.convert_yolo_output_to_bboxes(
                targets[i], grid=True, class_tensor=True, is_training=True
            )
            
            # Skip if no valid data
            if not pred_data and not gt_data:
                continue
            
            # Compute losses
            loss_dict = self._compute_single_image_loss_safe(pred_data, gt_data, device)
            
            # Accumulate with safety checks
            if not torch.isnan(loss_dict["siou"]) and not torch.isinf(loss_dict["siou"]):
                total_siou = total_siou + loss_dict["siou"]
            if not torch.isnan(loss_dict["objectness"]) and not torch.isinf(loss_dict["objectness"]):
                total_obj = total_obj + loss_dict["objectness"]
            if not torch.isnan(loss_dict["classification"]) and not torch.isinf(loss_dict["classification"]):
                total_cls = total_cls + loss_dict["classification"]
            
            valid_samples += 1
                
            # except Exception as e:
            #     print(f"Error processing sample {i}: {e}")
            #     continue
        
        # Avoid division by zero
        if valid_samples == 0:
            valid_samples = 1
            print("WARNING: No valid samples in batch")
        
        # Average over valid samples
        avg_siou = total_siou / valid_samples
        avg_obj = total_obj / valid_samples
        avg_cls = total_cls / valid_samples
        
        # Apply conservative loss weights
        weighted_loss = (
            self.loss_weights['box'] * torch.clamp(avg_siou, max=10.0) +
            self.loss_weights['obj'] * torch.clamp(avg_obj, max=10.0) +
            self.loss_weights['cls'] * torch.clamp(avg_cls, max=10.0)
        )
        
        # Scale down if too large
        if weighted_loss > 100:
            weighted_loss = weighted_loss * 0.1
        
        return {
            'total_loss': weighted_loss,
            'box_loss': avg_siou,
            'obj_loss': avg_obj,
            'cls_loss': avg_cls,
            'valid_samples': valid_samples
        }
    
    def _compute_single_image_loss_safe(self, pred_data, gt_data, device):
        """Safe version of single image loss computation"""
        try:
            # Use grid-based matching if available, otherwise direct matching
            if hasattr(self, '_use_direct_matching') and self._use_direct_matching:
                return self._compute_direct_matching_loss(pred_data, gt_data, device)
            else:
                return self._compute_grid_based_loss_safe(pred_data, gt_data, device)
        except Exception as e:
            print(f"Grid-based matching failed: {e}, falling back to direct matching")
            self._use_direct_matching = True
            return self._compute_direct_matching_loss(pred_data, gt_data, device)
    
    def _compute_grid_based_loss_safe(self, pred_data, gt_data, device):
        """Safe grid-based loss computation"""
        try:
            # Organize by grid cells
            pred_by_grid = self._organize_by_grid_safe(pred_data)
            gt_by_grid = self._organize_by_grid_safe(gt_data)
            
            siou_losses = []
            obj_losses = []
            cls_losses = []
            
            # Process all grid cells
            all_grids = set(pred_by_grid.keys()) | set(gt_by_grid.keys())
            
            for grid_coord in all_grids:
                grid_preds = pred_by_grid.get(grid_coord, [])
                grid_gts = gt_by_grid.get(grid_coord, [])
                
                # Compute losses for this grid cell
                if len(grid_gts) == 0:
                    # No GT: all predictions should be negative
                    for pred in grid_preds:
                        try:
                            pred_conf = self._safe_tensor([pred.get('conf', 0.5)], device)
                            obj_loss = F.binary_cross_entropy_with_logits(
                                pred_conf, torch.zeros_like(pred_conf), reduction='mean'
                            )
                            obj_losses.append(obj_loss)
                        except:
                            continue
                            
                elif len(grid_preds) == 0:
                    # GT exists but no predictions: penalize
                    penalty = torch.tensor(1.0, device=device, requires_grad=True)
                    obj_losses.append(penalty)
                    
                else:
                    # Both exist: match them
                    grid_losses = self._match_grid_predictions_safe(grid_preds, grid_gts, device)
                    siou_losses.extend(grid_losses["siou"])
                    obj_losses.extend(grid_losses["objectness"])
                    cls_losses.extend(grid_losses["classification"])
            
            # Aggregate losses
            total_siou = sum(siou_losses) if siou_losses else torch.tensor(0.0, device=device, requires_grad=True)
            total_obj = sum(obj_losses) if obj_losses else torch.tensor(0.0, device=device, requires_grad=True)
            total_cls = sum(cls_losses) if cls_losses else torch.tensor(0.0, device=device, requires_grad=True)
            
            # Normalize by positive samples
            num_pos = max(len(siou_losses), 1)
            total_siou = total_siou / num_pos
            total_cls = total_cls / num_pos
            
            return {
                "siou": total_siou,
                "objectness": total_obj,
                "classification": total_cls
            }
            
        except Exception as e:
            print(f"Error in grid-based loss: {e}")
            return self._compute_direct_matching_loss(pred_data, gt_data, device)
    
    def _organize_by_grid_safe(self, data_list):
        """Safely organize data by grid coordinates"""
        grid_dict = {}
        for item in data_list:
            try:
                grid_coord = item.get('grid', (0, 0))
                if not isinstance(grid_coord, (tuple, list)) or len(grid_coord) != 2:
                    grid_coord = (0, 0)  # Default grid
                if grid_coord not in grid_dict:
                    grid_dict[grid_coord] = []
                grid_dict[grid_coord].append(item)
            except:
                continue
        return grid_dict
    
    def _match_grid_predictions_safe(self, grid_preds, grid_gts, device):
        """Safely match predictions to ground truths in a grid cell"""
        siou_losses = []
        obj_losses = []
        cls_losses = []
        
        # Simple 1:1 matching (take min of pred/gt counts)
        matches = min(len(grid_preds), len(grid_gts))
        
        # Process matched pairs
        for i in range(matches):
            try:
                pred = grid_preds[i]
                gt = grid_gts[i]
                
                # Box regression loss
                pred_bbox = self._safe_tensor(pred.get('bbox', [0,0,1,1]), device)
                gt_bbox = self._safe_tensor(gt.get('bbox', [0,0,1,1]), device)
                
                # Use simple IoU loss instead of SIoU for stability
                iou_loss = 1.0 - self._compute_simple_iou(pred_bbox, gt_bbox)
                siou_losses.append(torch.tensor(iou_loss, device=device, requires_grad=True))
                
                # Objectness loss (positive)
                pred_conf = self._safe_tensor([pred.get('conf', 0.5)], device)
                obj_loss = F.binary_cross_entropy_with_logits(
                    pred_conf, torch.ones_like(pred_conf), reduction='mean'
                )
                obj_losses.append(obj_loss)
                
                # Classification loss
                pred_class = self._safe_tensor(pred.get('class_tensor', [1.0]), device)
                target_class = torch.tensor([gt.get('class_id', 0)], device=device, dtype=torch.long)
                
                if pred_class.numel() > 1:  # Multi-class
                    cls_loss = F.cross_entropy(pred_class.unsqueeze(0), target_class, reduction='mean')
                else:  # Binary
                    cls_loss = F.binary_cross_entropy_with_logits(
                        pred_class, torch.ones_like(pred_class), reduction='mean'
                    )
                cls_losses.append(cls_loss)
                
            except Exception as e:
                print(f"Error matching prediction {i}: {e}")
                continue
        
        # Handle unmatched predictions (negative)
        for i in range(matches, len(grid_preds)):
            try:
                pred = grid_preds[i]
                pred_conf = self._safe_tensor([pred.get('conf', 0.5)], device)
                obj_loss = F.binary_cross_entropy_with_logits(
                    pred_conf, torch.zeros_like(pred_conf), reduction='mean'
                )
                obj_losses.append(obj_loss)
            except:
                continue
        
        # Handle unmatched ground truths (missed detections)
        for i in range(matches, len(grid_gts)):
            penalty = torch.tensor(1.0, device=device, requires_grad=True)
            obj_losses.append(penalty)
        
        return {
            "siou": siou_losses,
            "objectness": obj_losses,
            "classification": cls_losses
        }
    
    def _compute_direct_matching_loss(self, pred_data, gt_data, device):
        """Simplified direct matching approach"""
        siou_losses = []
        obj_losses = []
        cls_losses = []
        
        # If no predictions, penalize objectness
        if not pred_data:
            if gt_data:
                # Penalize for missed ground truths
                penalty = torch.tensor(1.0, device=device, requires_grad=True)
                obj_losses.append(penalty)
        else:
            # Simple strategy: treat first prediction as positive if GT exists
            for i, pred in enumerate(pred_data[:min(len(pred_data), len(gt_data) if gt_data else 1)]):
                if i < len(gt_data):
                    # Positive match
                    gt = gt_data[i]
                    try:
                        pred_bbox = self._safe_tensor(pred.get('bbox', [0,0,1,1]), device)
                        gt_bbox = self._safe_tensor(gt.get('bbox', [0,0,1,1]), device)
                        
                        # Simple IoU-based loss instead of SIoU
                        iou_loss = 1.0 - self._compute_simple_iou(pred_bbox, gt_bbox)
                        siou_losses.append(torch.tensor(iou_loss, device=device, requires_grad=True))
                        
                        # Objectness loss
                        pred_conf = self._safe_tensor([pred.get('conf', 0.5)], device)
                        obj_loss = F.binary_cross_entropy_with_logits(
                            pred_conf, torch.ones_like(pred_conf), reduction='mean'
                        )
                        obj_losses.append(obj_loss)
                        
                        # Classification loss
                        pred_class = self._safe_tensor(pred.get('class_tensor', [1.0]), device)
                        if pred_class.numel() > 1:  # Multi-class
                            target_class = torch.tensor([gt.get('class_id', 0)], device=device, dtype=torch.long)
                            cls_loss = F.cross_entropy(pred_class.unsqueeze(0), target_class, reduction='mean')
                        else:  # Binary
                            cls_loss = F.binary_cross_entropy_with_logits(
                                pred_class, torch.ones_like(pred_class), reduction='mean'
                            )
                        cls_losses.append(cls_loss)
                        
                    except Exception as e:
                        print(f"Error in positive match computation: {e}")
                        # Add small penalty
                        penalty = torch.tensor(0.1, device=device, requires_grad=True)
                        obj_losses.append(penalty)
                else:
                    # Negative prediction
                    try:
                        pred_conf = self._safe_tensor([pred.get('conf', 0.5)], device)
                        obj_loss = F.binary_cross_entropy_with_logits(
                            pred_conf, torch.zeros_like(pred_conf), reduction='mean'
                        )
                        obj_losses.append(obj_loss)
                    except Exception as e:
                        print(f"Error in negative prediction: {e}")
        
        # Aggregate safely
        total_siou = sum(siou_losses) if siou_losses else torch.tensor(0.0, device=device, requires_grad=True)
        total_obj = sum(obj_losses) if obj_losses else torch.tensor(0.0, device=device, requires_grad=True)
        total_cls = sum(cls_losses) if cls_losses else torch.tensor(0.0, device=device, requires_grad=True)
        
        # Normalize
        num_pos = max(len(siou_losses), 1)
        total_siou = total_siou / num_pos
        total_cls = total_cls / num_pos
        
        return {
            "siou": total_siou,
            "objectness": total_obj,
            "classification": total_cls
        }
    
    def _safe_tensor(self, data, device):
        """Safely convert data to tensor"""
        try:
            if isinstance(data, torch.Tensor):
                return data.to(device)
            else:
                tensor = torch.tensor(data, device=device, dtype=torch.float32)
                # Handle empty or malformed tensors
                if tensor.numel() == 0:
                    return torch.tensor([0.0], device=device, dtype=torch.float32)
                return tensor
        except Exception as e:
            print(f"Error creating tensor: {e}")
            return torch.tensor([0.0], device=device, dtype=torch.float32)
    
    def _compute_simple_iou(self, box1, box2):
        """Simple IoU computation with error handling"""
        try:
            # Ensure boxes are valid
            if box1.numel() < 4 or box2.numel() < 4:
                return 0.0
            
            # Convert to xyxy format
            x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
            x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
            x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
            x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
            
            # Intersection
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            
            # Union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - inter_area
            
            if union_area <= 0:
                return 0.0
            
            return float(inter_area / union_area)
        except Exception as e:
            print(f"Error computing IoU: {e}")
            return 0.0


# Quick diagnostic function
def quick_loss_diagnosis(outputs, targets, loss_fn, processor):
    """Quick diagnosis of stuck loss"""
    print("QUICK LOSS DIAGNOSIS")
    print("="*50)
    
    # Check basic properties
    print(f"Outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
    print(f"Targets range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
    print(f"Outputs mean/std: {outputs.mean().item():.4f}/{outputs.std().item():.4f}")
    print(f"Targets mean/std: {targets.mean().item():.4f}/{targets.std().item():.4f}")
    
    # Check for common issues
    if outputs.std() < 1e-6:
        print("⚠️  Issue: Outputs have very low variance - model might not be learning")
    
    if (targets == 0).float().mean() > 0.99:
        print("⚠️  Issue: Targets are mostly zeros - check data loading")
    
    # Test with simplified loss
    simple_loss_fn = ObjectDetectionLoss(processor)
    try:
        loss = simple_loss_fn(outputs, targets)
        print(f"Simplified loss: {loss['total_loss'].item():.6f}")
    except Exception as e:
        print(f"Error with simplified loss: {e}")
    
    return loss

from collections import deque, defaultdict
import time
import json

class TrainingMonitor:
    """Enhanced training monitor for object detection"""
    
    def __init__(self, window_size=100, save_path="training_log.json"):
        self.window_size = window_size
        self.save_path = save_path
        
        # Loss tracking
        self.losses = defaultdict(list)
        self.recent_losses = defaultdict(lambda: deque(maxlen=window_size))
        
        # Learning rate tracking
        self.learning_rates = []
        
        # Timing
        self.epoch_times = []
        self.batch_times = deque(maxlen=100)
        self.start_time = time.time()
        
        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        
        # Best metrics
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        # Training phases
        self.phase_history = []
        
    def log_batch(self, loss_dict, batch_idx, epoch_idx, optimizer=None):
        """Log metrics for a single batch"""
        self.current_batch = batch_idx
        self.current_epoch = epoch_idx
        
        # Log losses
        for key, value in loss_dict.items():
            if key in ['total_loss', 'cls_loss', 'obj_loss', 'box_loss']:
                if torch.is_tensor(value):
                    loss_val = value.item()
                else:
                    loss_val = float(value)
                
                self.losses[key].append(loss_val)
                self.recent_losses[key].append(loss_val)
        
        # Log learning rate
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
        
        # Update best loss
        total_loss = loss_dict.get('total_loss', loss_dict.get('loss', 0))
        if torch.is_tensor(total_loss):
            total_loss = total_loss.item()
        
        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.best_epoch = epoch_idx
    
    def log_epoch(self, avg_loss):
        """Log epoch completion"""
        self.epoch_times.append(time.time())
        
        # Detect training phases
        self._detect_training_phase(avg_loss)
        
        # Save checkpoint
        if self.current_epoch % 5 == 0:
            self.save_log()
    
    def _detect_training_phase(self, current_loss):
        """Detect current training phase"""
        if len(self.losses['total_loss']) < 20:
            phase = "warmup"
        else:
            recent_trend = self._compute_trend(self.losses['total_loss'][-20:])
            if recent_trend > 0.01:
                phase = "increasing"
            elif recent_trend < -0.01:
                phase = "decreasing"
            elif abs(recent_trend) < 0.001:
                phase = "plateau"
            else:
                phase = "stable"
        
        if not self.phase_history or self.phase_history[-1] != phase:
            self.phase_history.append(phase)
            print(f"📊 Training phase: {phase}")
    
    def _compute_trend(self, values):
        """Compute trend (slope) of recent values"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]
    
    def print_status(self, detailed=False):
        """Print current training status"""
        if not self.losses:
            print("No training data logged yet")
            return
        
        print(f"\n{'='*60}")
        print(f"TRAINING STATUS - Epoch {self.current_epoch}, Batch {self.current_batch}")
        print(f"{'='*60}")
        
        # Current losses
        print("📈 CURRENT METRICS:")
        for key in ['total_loss', 'box_loss', 'obj_loss', 'cls_loss']:
            if key in self.recent_losses and self.recent_losses[key]:
                recent_avg = np.mean(list(self.recent_losses[key]))
                overall_avg = np.mean(self.losses[key])
                trend = self._compute_trend(list(self.recent_losses[key])[-10:])
                trend_arrow = "📈" if trend > 0.01 else "📉" if trend < -0.01 else "➡️"
                print(f"   {key:12}: {recent_avg:.4f} (overall: {overall_avg:.4f}) {trend_arrow}")
        
        # Learning dynamics
        if len(self.losses['total_loss']) > 10:
            recent_losses = self.losses['total_loss'][-10:]
            variance = np.var(recent_losses)
            trend = self._compute_trend(recent_losses)
            
            print(f"\n🔍 LEARNING DYNAMICS:")
            print(f"   Variance:     {variance:.6f} {'(stable)' if variance < 0.01 else '(changing)'}")
            print(f"   Trend:        {trend:.6f} {'(improving)' if trend < -0.001 else '(worsening)' if trend > 0.001 else '(stable)'}")
            print(f"   Best loss:    {self.best_loss:.4f} (epoch {self.best_epoch})")
        
        # Training speed
        if len(self.batch_times) > 10:
            avg_batch_time = np.mean(list(self.batch_times))
            print(f"\n⏱️  TIMING:")
            print(f"   Avg batch:    {avg_batch_time:.2f}s")
            if self.total_batches > 0:
                eta = (self.total_batches - self.current_batch) * avg_batch_time
                print(f"   ETA:          {eta/60:.1f} minutes")
        
        # Current phase
        if self.phase_history:
            print(f"\n📊 CURRENT PHASE: {self.phase_history[-1].upper()}")
        
        # Recommendations
        self._print_recommendations()
        
        if detailed:
            self._print_detailed_analysis()
    
    def _print_recommendations(self):
        """Print training recommendations"""
        if len(self.losses['total_loss']) < 10:
            return
        
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = []
        
        recent_losses = self.losses['total_loss'][-20:]
        trend = self._compute_trend(recent_losses)
        variance = np.var(recent_losses)
        
        # Check for common issues
        if variance < 1e-6:
            recommendations.append("Loss not changing - consider increasing learning rate")
        
        if trend > 0.05:
            recommendations.append("Loss increasing rapidly - consider reducing learning rate")
        
        if len(recent_losses) > 10 and all(l > recent_losses[0] for l in recent_losses[-5:]):
            recommendations.append("Consistent loss increase - check for overfitting or high LR")
        
        # Loss component analysis
        if 'box_loss' in self.recent_losses and 'obj_loss' in self.recent_losses:
            box_avg = np.mean(list(self.recent_losses['box_loss']))
            obj_avg = np.mean(list(self.recent_losses['obj_loss']))
            
            if box_avg > obj_avg * 10:
                recommendations.append("Box loss dominates - consider rebalancing loss weights")
            elif obj_avg > box_avg * 10:
                recommendations.append("Objectness loss dominates - consider rebalancing loss weights")
        
        # Print recommendations
        for i, rec in enumerate(recommendations[:3], 1):  # Max 3 recommendations
            print(f"   {i}. {rec}")
        
        if not recommendations:
            if trend < -0.01:
                print("   ✅ Training progressing well!")
            else:
                print("   ⚠️  Monitor for signs of convergence")
    
    def _print_detailed_analysis(self):
        """Print detailed analysis"""
        print(f"\n🔬 DETAILED ANALYSIS:")
        
        # Loss component ratios
        if all(k in self.recent_losses for k in ['box_loss', 'obj_loss', 'cls_loss']):
            box_avg = np.mean(list(self.recent_losses['box_loss']))
            obj_avg = np.mean(list(self.recent_losses['obj_loss']))
            cls_avg = np.mean(list(self.recent_losses['cls_loss']))
            total = box_avg + obj_avg + cls_avg
            
            print(f"   Loss composition:")
            print(f"     Box:  {box_avg/total*100:.1f}% ({box_avg:.4f})")
            print(f"     Obj:  {obj_avg/total*100:.1f}% ({obj_avg:.4f})")
            print(f"     Cls:  {cls_avg/total*100:.1f}% ({cls_avg:.4f})")
        
        # Learning rate history
        if len(self.learning_rates) > 1:
            current_lr = self.learning_rates[-1]
            lr_changes = sum(1 for i in range(1, len(self.learning_rates)) 
                           if abs(self.learning_rates[i] - self.learning_rates[i-1]) > 1e-8)
            print(f"   Learning rate: {current_lr:.2e} ({lr_changes} changes)")
    
    def plot_training_curves(self, save_path=None, show=True):
        """Plot training curves"""
        if not self.losses:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot individual loss components
        loss_types = ['total_loss', 'box_loss', 'obj_loss', 'cls_loss']
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (loss_type, color) in enumerate(zip(loss_types, colors)):
            if loss_type in self.losses and self.losses[loss_type]:
                ax = axes[i]
                values = self.losses[loss_type]
                
                # Plot raw values
                ax.plot(values, alpha=0.3, color=color, linewidth=0.5)
                
                # Plot smoothed values
                if len(values) > 10:
                    window = min(50, len(values) // 10)
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(values)), smoothed, color=color, linewidth=2)
                
                ax.set_title(f'{loss_type.replace("_", " ").title()}')
                ax.set_xlabel('Batch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(values) > 20:
                    x = np.arange(len(values))
                    z = np.polyfit(x, values, 1)
                    p = np.poly1d(z)
                    ax.plot(x, p(x), '--', color='black', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        if show:
            plt.show()
    
    def save_log(self):
        """Save training log to file"""
        log_data = {
            'losses': {k: v for k, v in self.losses.items()},
            'learning_rates': self.learning_rates,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'current_epoch': self.current_epoch,
            'phase_history': self.phase_history,
            'training_time': time.time() - self.start_time
        }
        
        with open(self.save_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def load_log(self, path=None):
        """Load training log from file"""
        load_path = path or self.save_path
        try:
            with open(load_path, 'r') as f:
                log_data = json.load(f)
            
            self.losses = defaultdict(list, log_data.get('losses', {}))
            self.learning_rates = log_data.get('learning_rates', [])
            self.best_loss = log_data.get('best_loss', float('inf'))
            self.best_epoch = log_data.get('best_epoch', 0)
            self.current_epoch = log_data.get('current_epoch', 0)
            self.phase_history = log_data.get('phase_history', [])
            
            print(f"Training log loaded from {load_path}")
            
        except FileNotFoundError:
            print(f"Log file {load_path} not found")
        except Exception as e:
            print(f"Error loading log: {e}")
    
    def reset(self):
        """Reset all tracking data"""
        self.losses.clear()
        self.recent_losses.clear()
        self.learning_rates.clear()
        self.epoch_times.clear()
        self.batch_times.clear()
        self.phase_history.clear()
        self.current_epoch = 0
        self.current_batch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.start_time = time.time()


# Enhanced training loop with monitoring
def train_with_monitoring(model, dataloader, loss_fn, optimizer, num_epochs, 
                         scheduler=None, save_model_path=None, monitor_frequency=50, device='cpu'):
    """Enhanced training loop with comprehensive monitoring"""
    
    # Initialize monitor
    monitor = TrainingMonitor(save_path="detection_training.json")
    
    # Load previous training if exists
    monitor.load_log()
    
    print("Starting training with enhanced monitoring...")
    model.to(device)
    for epoch in range(monitor.current_epoch, num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_idx, (x, targets) in enumerate(dataloader):
            batch_start_time = time.time()
            x, targets = x.to(device), targets.to(device)
            # Forward pass
            outputs = model(x)
            loss_dict = loss_fn(outputs, targets)
            total_loss = loss_dict.get('total_loss', loss_dict)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Log batch metrics
            monitor.log_batch(loss_dict, batch_idx, epoch, optimizer)
            epoch_losses.append(total_loss.item())
            
            # Record batch time
            monitor.batch_times.append(time.time() - batch_start_time)
            
            # Print status periodically
            if batch_idx % monitor_frequency == 0:
                monitor.print_status()
                
                # Plot curves every 200 batches
                if batch_idx > 0 and batch_idx % 200 == 0:
                    monitor.plot_training_curves(
                        save_path=f'training_curves_epoch_{epoch}_batch_{batch_idx}.png',
                        show=False
                    )
        
        # End of epoch
        avg_epoch_loss = np.mean(epoch_losses)
        monitor.log_epoch(avg_epoch_loss)
        
        print(f"\n🎯 Epoch {epoch+1}/{num_epochs} completed:")
        print(f"   Average Loss: {avg_epoch_loss:.4f}")
        print(f"   Best Loss: {monitor.best_loss:.4f} (epoch {monitor.best_epoch+1})")
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_epoch_loss)
            else:
                scheduler.step()
        
        # Save model if best
        if avg_epoch_loss < monitor.best_loss and save_model_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, save_model_path)
            print(f"💾 Best model saved to {save_model_path}")
    
    # Final summary
    print(f"\n🏁 Training completed!")
    monitor.print_status(detailed=True)
    monitor.plot_training_curves(save_path='final_training_curves.png')
    monitor.save_log()
    
    return monitor


class AdaptiveObjectDetectionLoss(nn.Module):
    """
    Adaptive loss that automatically balances loss components based on their magnitudes
    """
    def __init__(self, processor, classes_alpha=None, 
                 initial_weights=None, adaptive_balancing=True, eps=1e-7):
        super(AdaptiveObjectDetectionLoss, self).__init__()
        self.processor = processor
        self.eps = eps
        self.adaptive_balancing = adaptive_balancing
        
        # Loss components
        self.siou_loss = SIoU(x1y1x2y2=False, eps=eps)
        self.binary_focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.multi_focal_loss = FocalLoss(
            alpha=classes_alpha, 
            gamma=2.0, 
            num_classes=getattr(processor, 'num_classes', None)
        )
        
        # Adaptive loss weights - start with balanced approach
        if initial_weights is None:
            initial_weights = {'box': 2.0, 'obj': 0.5, 'cls': 1.0}  # Reduce obj weight!
        
        self.loss_weights = nn.Parameter(
            torch.tensor([initial_weights['box'], initial_weights['obj'], initial_weights['cls']],
                        dtype=torch.float32),
            requires_grad=False  # We'll update these manually
        )
        
        # Track loss component history for adaptive balancing
        self.loss_history = {'box': [], 'obj': [], 'cls': []}
        self.update_frequency = 50  # Rebalance every N batches
        self.batch_count = 0
        
        # Conservative thresholds
        self.pos_iou_thresh = 0.5
        self.neg_iou_thresh = 0.3
        
    def forward(self, outputs, targets):
        """Forward pass with adaptive loss balancing"""
        # Input validation
        outputs = self._sanitize_inputs(outputs)
        targets = self._sanitize_inputs(targets)
        
        batch_size = outputs.shape[0]
        device = outputs.device
        
        # Initialize loss accumulators
        total_siou = torch.tensor(0.0, device=device, requires_grad=True)
        total_obj = torch.tensor(0.0, device=device, requires_grad=True)
        total_cls = torch.tensor(0.0, device=device, requires_grad=True)
        
        valid_samples = 0
        
        # Process each sample in batch
        for i in range(batch_size):
            try:
                pred_data = self.processor.convert_yolo_output_to_bboxes(
                    outputs[i], grid=True, class_tensor=True, is_training=True
                )
                gt_data = self.processor.convert_yolo_output_to_bboxes(
                    targets[i], grid=True, class_tensor=True, is_training=True
                )
                
                loss_dict = self._compute_single_image_loss(pred_data, gt_data, device)
                
                if self._is_valid_loss(loss_dict):
                    total_siou = total_siou + loss_dict["siou"]
                    total_obj = total_obj + loss_dict["objectness"]
                    total_cls = total_cls + loss_dict["classification"]
                    valid_samples += 1
                
            except Exception as e:
                print(f"Warning: Error processing sample {i}: {e}")
                continue
        
        # Avoid division by zero
        valid_samples = max(valid_samples, 1)
        
        # Average over valid samples
        avg_siou = total_siou / valid_samples
        avg_obj = total_obj / valid_samples
        avg_cls = total_cls / valid_samples
        
        # Clamp extreme values
        avg_siou = torch.clamp(avg_siou, max=5.0)
        avg_obj = torch.clamp(avg_obj, max=5.0)
        avg_cls = torch.clamp(avg_cls, max=5.0)
        
        # Update loss weights adaptively
        if self.adaptive_balancing:
            self._update_adaptive_weights(avg_siou, avg_obj, avg_cls)
        
        # Apply current weights
        current_weights = self.loss_weights.to(device)
        weighted_loss = (
            current_weights[0] * avg_siou +    # box
            current_weights[1] * avg_obj +     # obj
            current_weights[2] * avg_cls       # cls
        )
        
        # Store for adaptive balancing
        self.loss_history['box'].append(avg_siou.item())
        self.loss_history['obj'].append(avg_obj.item())
        self.loss_history['cls'].append(avg_cls.item())
        
        # Keep only recent history
        max_history = 200
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history:]
        
        return {
            'total_loss': weighted_loss,
            'box_loss': avg_siou,
            'obj_loss': avg_obj,
            'cls_loss': avg_cls,
            'valid_samples': valid_samples,
            'loss_weights': {
                'box': current_weights[0].item(),
                'obj': current_weights[1].item(), 
                'cls': current_weights[2].item()
            }
        }
    
    def _update_adaptive_weights(self, box_loss, obj_loss, cls_loss):
        """Update loss weights based on component magnitudes"""
        self.batch_count += 1
        
        if self.batch_count % self.update_frequency == 0 and len(self.loss_history['box']) > 10:
            # Calculate recent averages
            recent_box = sum(self.loss_history['box'][-10:]) / 10
            recent_obj = sum(self.loss_history['obj'][-10:]) / 10
            recent_cls = sum(self.loss_history['cls'][-10:]) / 10
            
            # Target ratios (what we want the losses to be relative to each other)
            target_ratios = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)  # Equal contribution
            
            # Current ratios
            total_loss = recent_box + recent_obj + recent_cls
            if total_loss > 0:
                current_ratios = torch.tensor([recent_box, recent_obj, recent_cls]) / total_loss
                
                # Compute adjustment factors (inverse of current ratios to balance)
                adjustments = target_ratios / (current_ratios + 1e-6)
                
                # Smooth the adjustments to avoid sudden changes
                smoothing = 0.1
                new_weights = self.loss_weights * (1 - smoothing) + adjustments * smoothing
                
                # Clamp weights to reasonable ranges
                new_weights = torch.clamp(new_weights, min=0.1, max=5.0)
                
                self.loss_weights.data = new_weights
                
                print(f"🔄 Adaptive weights updated:")
                print(f"   Box: {new_weights[0]:.3f}, Obj: {new_weights[1]:.3f}, Cls: {new_weights[2]:.3f}")
    
    def _sanitize_inputs(self, tensor):
        """Clean inputs of NaN/Inf values"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        return tensor
    
    def _is_valid_loss(self, loss_dict):
        """Check if computed losses are valid"""
        for key, value in loss_dict.items():
            if torch.isnan(value) or torch.isinf(value) or value < 0:
                return False
        return True
    
    def _compute_single_image_loss(self, pred_data, gt_data, device):
        """Simplified and robust single image loss computation"""
        siou_losses = []
        obj_losses = []
        cls_losses = []
        
        # Handle empty cases
        if not pred_data and not gt_data:
            return self._zero_losses(device)
        
        if not pred_data and gt_data:
            # Missing detections - penalize objectness
            penalty = torch.tensor(len(gt_data) * 0.5, device=device, requires_grad=True)
            obj_losses.append(penalty)
        
        elif pred_data and not gt_data:
            # False positives - penalize all predictions
            for pred in pred_data:
                pred_conf = self._safe_tensor([pred.get('conf', 0.5)], device)
                neg_loss = F.binary_cross_entropy_with_logits(
                    pred_conf, torch.zeros_like(pred_conf), reduction='mean'
                )
                obj_losses.append(neg_loss)
        
        else:
            # Both exist - match and compute losses
            matches = min(len(pred_data), len(gt_data))
            
            # Process matched pairs
            for i in range(matches):
                try:
                    pred = pred_data[i]
                    gt = gt_data[i]
                    
                    # Box regression loss (simplified)
                    pred_bbox = self._safe_tensor(pred.get('bbox', [0,0,0.1,0.1]), device)
                    gt_bbox = self._safe_tensor(gt.get('bbox', [0,0,0.1,0.1]), device)
                    
                    # Use L1 loss for stability instead of SIoU initially
                    box_loss = F.l1_loss(pred_bbox, gt_bbox, reduction='mean')
                    siou_losses.append(box_loss)
                    
                    # Objectness loss (positive)
                    pred_conf = self._safe_tensor([pred.get('conf', 0.5)], device)
                    
                    # Use IoU as soft target for objectness
                    iou = self._compute_simple_iou(pred_bbox, gt_bbox)
                    soft_target = torch.clamp(torch.tensor([iou], device=device), min=0.3, max=0.9)
                    
                    obj_loss = F.binary_cross_entropy_with_logits(
                        pred_conf, soft_target, reduction='mean'
                    )
                    obj_losses.append(obj_loss)
                    
                    # Classification loss
                    pred_class = self._safe_tensor(pred.get('class_tensor', [1.0]), device)
                    target_class_id = max(0, min(gt.get('class_id', 0), pred_class.numel()-1))
                    target_class = torch.tensor([target_class_id], device=device, dtype=torch.long)
                    
                    if pred_class.numel() > 1:  # Multi-class
                        cls_loss = F.cross_entropy(pred_class.unsqueeze(0), target_class, reduction='mean')
                    else:  # Binary
                        cls_loss = F.binary_cross_entropy_with_logits(
                            pred_class, torch.ones_like(pred_class), reduction='mean'
                        )
                    cls_losses.append(cls_loss)
                    
                except Exception as e:
                    print(f"Warning: Error in matched pair {i}: {e}")
                    continue
            
            # Handle unmatched predictions (false positives)
            for i in range(matches, len(pred_data)):
                try:
                    pred = pred_data[i]
                    pred_conf = self._safe_tensor([pred.get('conf', 0.5)], device)
                    neg_loss = F.binary_cross_entropy_with_logits(
                        pred_conf, torch.zeros_like(pred_conf), reduction='mean'
                    )
                    obj_losses.append(neg_loss)
                except:
                    continue
            
            # Handle unmatched ground truths (missed detections)
            if matches < len(gt_data):
                missed_penalty = torch.tensor((len(gt_data) - matches) * 0.3, 
                                            device=device, requires_grad=True)
                obj_losses.append(missed_penalty)
        
        # Aggregate losses
        total_siou = sum(siou_losses) if siou_losses else torch.tensor(0.0, device=device, requires_grad=True)
        total_obj = sum(obj_losses) if obj_losses else torch.tensor(0.0, device=device, requires_grad=True)
        total_cls = sum(cls_losses) if cls_losses else torch.tensor(0.0, device=device, requires_grad=True)
        
        # Normalize by positive samples
        num_pos = max(len(siou_losses), 1)
        total_siou = total_siou / num_pos
        total_cls = total_cls / num_pos
        
        return {
            "siou": total_siou,
            "objectness": total_obj,
            "classification": total_cls
        }
    
    def _zero_losses(self, device):
        """Return zero losses for empty cases"""
        return {
            "siou": torch.tensor(0.0, device=device, requires_grad=True),
            "objectness": torch.tensor(0.0, device=device, requires_grad=True),
            "classification": torch.tensor(0.0, device=device, requires_grad=True)
        }
    
    def _safe_tensor(self, data, device):
        """Safely convert to tensor with error handling"""
        try:
            if isinstance(data, torch.Tensor):
                return data.to(device)
            tensor = torch.tensor(data, device=device, dtype=torch.float32)
            if tensor.numel() == 0:
                return torch.tensor([0.0], device=device, dtype=torch.float32)
            return tensor
        except:
            return torch.tensor([0.0], device=device, dtype=torch.float32)
    
    def _compute_simple_iou(self, box1, box2):
        """Simple IoU computation"""
        try:
            if box1.numel() < 4 or box2.numel() < 4:
                return 0.0
            
            # Convert to xyxy
            x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
            x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
            x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
            x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
            
            # Intersection
            xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
            xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
            
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - inter + 1e-6
            
            return float(inter / union)
        except:
            return 0.0
    
    def get_loss_weights(self):
        """Get current loss weights"""
        return {
            'box': self.loss_weights[0].item(),
            'obj': self.loss_weights[1].item(),
            'cls': self.loss_weights[2].item()
        }
    
    def set_loss_weights(self, box_weight=None, obj_weight=None, cls_weight=None):
        """Manually set loss weights"""
        with torch.no_grad():
            if box_weight is not None:
                self.loss_weights[0] = box_weight
            if obj_weight is not None:
                self.loss_weights[1] = obj_weight  
            if cls_weight is not None:
                self.loss_weights[2] = cls_weight

def create_balanced_loss(processor):
    """Create a balanced loss function for your current training"""
    
    # Reduce objectness weight significantly since it's dominating
    balanced_weights = {
        'box': 2.0,   # Increase box loss importance
        'obj': 0.3,   # Reduce objectness weight (was causing 94% of loss)
        'cls': 1.5    # Moderate classification weight
    }
    
    return AdaptiveObjectDetectionLoss(
        processor=processor,
        classes_alpha=None,  # Will use default balanced alpha
        initial_weights=balanced_weights,
        adaptive_balancing=True  # Will auto-adjust over time
    )