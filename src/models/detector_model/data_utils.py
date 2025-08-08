from torch.utils.data import Dataset
import json, os, torch

class TrainingDataset(Dataset):
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
                              self.processor.num_anchors * (5 + self.processor.num_classes))

class COCOProcessor:
    def __init__(self, classes):
        self.group_classes = classes

    def get_grouped_class(self, class_name):
        class_name = class_name.strip()

        for group, items in self.group_classes.items():
            if class_name in items:
                return group
        print(f"Class '{class_name}' not found in grouped classes.")
        return "Other"        

    def extract_annotations(self, json_path, image_dir, image_id=None, convert=False):
        with open(json_path, 'r') as f:
            data = json.load(f)
        labels = []
        # Select image
        images = data['images']
        annotations = data['annotations']
        categories = {cat['id']: cat['name'] for cat in data['categories']}

        for img in images:
            if image_id and img['id'] != image_id:
                continue

            image_path = os.path.join(image_dir, img['file_name'])
            size = (img['width'], img['height'])
            bboxes = []
            classes = []
            for ann in annotations:
                if ann['image_id'] != img['id']:
                    continue
                bboxes.append(ann['bbox'])
                if convert:
                    class_name = categories[ann['category_id']]
                    grouped_class = self.get_grouped_class(class_name)
                    classes.append(grouped_class)
                else:
                    classes.append(categories[ann['category_id']])
                    
            label = {"Path": image_path,
                    "Size": size,
                    "Bbox": bboxes,
                    "Class": classes}
                                    
            labels.append(label)
        return labels