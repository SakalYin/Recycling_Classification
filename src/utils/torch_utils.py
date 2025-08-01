import torch
from models.CustomNet import MediumCustomNet
from torchvision import transforms
from PIL import Image

train_transforms = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.ToTensor(),
])

def transforms(image):
    image = train_transforms(image)
    return image.unsqueeze(0)


class MediumCustomNetWrapper(MediumCustomNet):
    def __init__(self, num_classes=2, dropout_rate=0.2):
        super(MediumCustomNetWrapper, self).__init__(num_classes)
        self.transforms = transforms
        self.labels = {0: 'glass',
                       1: 'metal',
                       2: 'organic',
                       3: 'paper',
                       4: 'plastic',
                       5: 'textile',
                    }
        
    def detect(self, image):
        image_tensor = self.transforms(image)
        with torch.no_grad():
            output = self.forward(image_tensor)
            _, preds = torch.max(output, 1)
        return self.labels[preds.item()]