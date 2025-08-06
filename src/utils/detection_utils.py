import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
from PIL import Image

train_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

def load_image(self, image_path):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size  # (width, height)
    
    # Preprocess for model
    processed_image = self.transform(original_image).unsqueeze(0)  # Add batch dim
    
    return original_image, processed_image, original_size
    
def process_image(self, image_path, apply_nms=True, visualize=True, save_path=None):
    """
    Complete processing pipeline for a single image
    
    Args:
        image_path: Path to input image
        apply_nms: Whether to apply non-maximum suppression
        visualize: Whether to show/save visualization
        save_path: Path to save visualization (optional)
        
    Returns:
        detections: List of final detections
    """
    print(f"Processing: {image_path}")
    
    # Load and preprocess image
    original_image, processed_image, original_size = self.load_image(image_path)
    print(f"Original image size: {original_size}")
    
    # Run inference
    self.model.eval()
    with torch.no_grad():
        predictions = self.model(processed_image)
    
    print(f"Model output shape: {predictions.shape}")
    
    # Decode predictions
    detections = self.decode_predictions(predictions, original_size)
    print(f"Raw detections: {len(detections)}")
    
    # Apply NMS if requested
    if apply_nms:
        detections = self.apply_nms(detections)
        print(f"Final detections after NMS: {len(detections)}")
    
    # Print detection results
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, conf, class_id, class_name = detection
        print(f"Detection {i+1}: {class_name} ({conf:.3f}) at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Visualize if requested
    if visualize and len(detections) > 0:
        self.visualize_predictions(image_path, detections, save_path)
    elif len(detections) == 0:
        print("No detections found above confidence threshold")
    
    return detections