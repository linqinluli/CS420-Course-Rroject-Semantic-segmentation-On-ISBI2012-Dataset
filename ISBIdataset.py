from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import glob


class ISBIDataset(Dataset):
    
    def __init__(self, img_path, label_path, transforms=None):
        img_list = sorted(glob.glob(img_path+'*.png'))
        label_list = sorted(glob.glob(label_path+'*.png'))
        self.img = img_list
        self.label = label_list
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):
        img = Image.open(self.img[index]).convert("L")
        label = Image.open(self.label[index]).convert("L")
        
        if self.transforms:
            img = self.transforms(img)
            label = self.transforms(label)
            
        return img, label