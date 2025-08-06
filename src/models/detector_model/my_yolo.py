import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class YOLO(nn.Module):
    def __init__(self, num_classes=20, num_anchors=3, grid_size=7):
        super(YOLO, self).__init__()
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
            nn.Flatten(),
            nn.Linear(256 * (grid_size // 4)**2, grid_size * grid_size * (num_anchors * 5 + num_classes)),
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.detector(features)
        return predictions.view(-1, self.grid_size, self.grid_size, self.num_anchors * 5 + self.num_classes * self.num_anchors)
    

class YOLOTrainingProcessor:
    def __init__(self, classes, input_size=448, grid_size=7, num_anchors=1):
        """
        Initialize YOLO training data processor
        
        Args:
            input_size: Model input image size
            grid_size: Grid size (7x7)
            num_classes: Number of object classes
            num_anchors: Number of anchors per grid cell
        """
        self.input_size = input_size
        self.grid_size = grid_size
        self.num_classes = len(classes)
        self.num_anchors = num_anchors
        self.cell_size = input_size / grid_size
        self.class_names = classes
        
        # Training transforms with augmentation
        self.train_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        self.val_transforms = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_json_object(self, json):
        """
        Extrach Json image path and annotation
        
        Args:
            json_obj: Json object
            
        Returns:
            List of bounding boxes: [[x1, y1, x2, y2, class_id], ...]
        """
        image_path = json['Path']
        objects = json['Class']
        bboxes = json['Bbox']
        
        boxes = []
        for i in range(len(objects)):
            class_name = objects[i]
            class_id = self._get_class_id(class_name)
            
            bbox = bboxes[i]
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            
            boxes.append([x1, y1, x2, y2, class_id])
        
        return boxes, image_path
    
    def _get_class_id(self, class_name):
        """Map class name to class ID"""
        return self.class_names.index(class_name) if class_name in self.class_names else 0
    
    def apply_horizontal_flip(self, image, boxes, p=0.5):
        """Apply horizontal flip augmentation to image and bboxes."""
        if random.random() < p:
            image = transforms.functional.hflip(image)

            # Get image width (works for both PIL and tensor)
            width = image.width if hasattr(image, 'width') else image.shape[-1]

            flipped_boxes = []
            for box in boxes:
                x1, y1, x2, y2, class_id = box

                # Flip x-coordinates
                new_x1 = width - x2
                new_x2 = width - x1

                # Sort just in case
                new_x1, new_x2 = sorted([new_x1, new_x2])

                flipped_boxes.append([new_x1, y1, new_x2, y2, class_id])

            return image, flipped_boxes

        return image, boxes

    
    def apply_random_crop(self, image, boxes, p=0.3):
        """Apply random crop augmentation (simplified version)"""
        if random.random() < p and len(boxes) > 0:
            width, height = image.size
            
            # Simple crop - take 80-100% of image
            crop_factor = random.uniform(0.8, 1.0)
            new_width = int(width * crop_factor)
            new_height = int(height * crop_factor)
            
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            
            image = image.crop((left, top, left + new_width, top + new_height))
            
            # Adjust bounding boxes
            adjusted_boxes = []
            for box in boxes:
                x1, y1, x2, y2, class_id = box
                
                # Adjust coordinates
                x1 -= left
                y1 -= top
                x2 -= left
                y2 -= top
                
                # Check if box is still valid after crop
                if x2 > 0 and y2 > 0 and x1 < new_width and y1 < new_height:
                    # Clamp to image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(new_width, x2)
                    y2 = min(new_height, y2)
                    
                    # Only keep if box is still reasonable size
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        adjusted_boxes.append([x1, y1, x2, y2, class_id])
            
            return image, adjusted_boxes
        
        return image, boxes
    
    def convert_to_yolo_target(self, boxes, original_size, get_anchors=False):
        """
        Convert bounding boxes to YOLO training target format
        
        Args:
            boxes: List of [x1, y1, x2, y2, class_id]
            original_size: (width, height) of original image
            
        Returns:
            target: Tensor of shape [grid_size, grid_size, num_anchors * 5 + num_classes]
        """
        target = torch.zeros(self.grid_size, self.grid_size, self.num_anchors * 5 + self.num_classes * self.num_anchors)
        anchor_counter = torch.zeros(self.grid_size, self.grid_size, dtype=torch.int)
        
        orig_width, orig_height = original_size
        
        for box in boxes:
            x1, y1, x2, y2, class_id = box
            
            # Convert to normalized coordinates (0-1)
            x1 /= orig_width
            y1 /= orig_height
            x2 /= orig_width
            y2 /= orig_height
            
            # Calculate center and dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Find which grid cell contains the center
            grid_x = int(center_x * self.grid_size)
            grid_y = int(center_y * self.grid_size)
            
            # Clamp to grid boundaries
            grid_x = min(grid_x, self.grid_size - 1)
            grid_y = min(grid_y, self.grid_size - 1)
            
            # Calculate relative position within the cell (0-1)
            cell_x = (center_x * self.grid_size) - grid_x
            cell_y = (center_y * self.grid_size) - grid_y
            
            # anchor indexer
            base_idx = anchor_counter[grid_y, grid_x] * 5
            
            # Set bounding box coordinates (relative to cell and image)
            target[grid_y, grid_x, base_idx] = cell_x      # x relative to cell
            target[grid_y, grid_x, base_idx + 1] = cell_y  # y relative to cell
            target[grid_y, grid_x, base_idx + 2] = width   # width relative to image
            target[grid_y, grid_x, base_idx + 3] = height  # height relative to image
            target[grid_y, grid_x, base_idx + 4] = 1.0     # confidence (objectness)
            
            # Set class probabilities (one-hot encoding)
            class_start_idx = self.num_anchors * 5 + anchor_counter[grid_y, grid_x] * self.num_classes
            target[grid_y, grid_x, class_start_idx + class_id] = 1.0
            
            anchor_counter[grid_y, grid_x] += 1
        
        if get_anchors:
            return target, anchor_counter
        
        return target
    
    def process_training_sample(self, json_obj, apply_augmentation=True, get_anchors=False):
        """
        Process a single training sample
        
        Args:
            json_obj: JSON object with annotations
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            image_tensor: Processed image tensor
            target_tensor: Target tensor for training
        """
        
        bboxes, image_path = self.extract_json_object(json_obj)
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Apply augmentations if training
        if apply_augmentation:
            image, boxes = self.apply_horizontal_flip(image, bboxes)
            image, boxes = self.apply_random_crop(image, bboxes)
        
        # Apply transforms
        if apply_augmentation:
            image_tensor = self.train_transforms(image)
        else:
            image_tensor = self.val_transforms(image)
        
        # Convert to YOLO target format
        if get_anchors:
            target_tensor, anchor_pose = self.convert_to_yolo_target(bboxes, original_size, get_anchors=get_anchors)
        else:
            target_tensor = self.convert_to_yolo_target(bboxes, original_size, get_anchors=get_anchors)
        
        if get_anchors:
            return image_tensor, target_tensor, anchor_pose
        return image_tensor, target_tensor
    
    def visualize_training_sample(self, image_tensor, target_tensor, anchors_pose, save_path=None):
        """Visualize processed training sample"""
        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image_tensor * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to PIL
        image_pil = transforms.ToPILImage()(image)
        
        # Create visualization
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_pil)
        
        # Draw grid
        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
            ax.axhline(x, color='gray', alpha=0.3, linewidth=0.5)
        
        # Draw bounding boxes from target
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for anchors_idx in range(anchors_pose[i, j]):
                # Check if cell has an object
                    start_idx = anchors_idx * 5
                    cell_x = target_tensor[i, j, start_idx].item()
                    cell_y = target_tensor[i, j, start_idx+1].item()
                    width = target_tensor[i, j, start_idx+2].item()
                    height = target_tensor[i, j, start_idx+3].item()
                    
                    # Convert to absolute coordinates
                    center_x = (j + cell_x) * self.cell_size
                    center_y = (i + cell_y) * self.cell_size
                    abs_width = width * self.input_size
                    abs_height = height * self.input_size
                    
                    x1 = center_x - abs_width / 2
                    y1 = center_y - abs_height / 2
                    
                    # Draw bounding box
                    rect = patches.Rectangle(
                        (x1, y1), abs_width, abs_height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Find class
                    class_prob_start = self.num_anchors * 5 + (anchors_idx * self.num_classes)
                    class_probs = target_tensor[i, j, class_prob_start:class_prob_start + self.num_classes]
                    class_id = torch.argmax(class_probs).item()
                    
                    # Add label
                    ax.text(x1, y1 - 5, f'Class {self.class_names[class_id]}', 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.8),
                        fontsize=8)
        
        ax.set_xlim(0, self.input_size)
        ax.set_ylim(self.input_size, 0)
        ax.set_title('Training Sample with Ground Truth')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()


class YOLODataset(Dataset):
    """PyTorch Dataset for YOLO training"""
    def __init__(self, data_json, processor, is_training=True):
        self.data = data_json   
        self.processor = processor
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]['Path']
        
        try:
            image_tensor, target_tensor = self.processor.process_training_sample(
                self.data[idx],
                apply_augmentation=self.is_training)
            return image_tensor, target_tensor
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return empty tensors as fallback
            return torch.zeros(3, self.processor.input_size, self.processor.input_size), \
                   torch.zeros(self.processor.grid_size, self.processor.grid_size, 
                              self.processor.num_anchors * 5 + self.processor.num_classes)
