import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Normalize
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = os.listdir(img_dir)
        self.classes.sort()
        self.img_labels = []
        self.img_paths = []
        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.img_labels.append(class_index)
                self.img_paths.append(os.path.join(class_dir, img_name))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]
        image = read_image(img_path).float()
        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label