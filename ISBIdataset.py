from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import glob
import cv2

class ISBIDataset(Dataset):
    
    def __init__(self, img_path, label_path, transforms=None, aug=False):
        img_list = sorted(glob.glob(img_path+'*.png'))
        label_list = sorted(glob.glob(label_path+'*.png'))
        self.img = img_list
        self.label = label_list
        self.transforms = transforms
        self.aug = aug
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):
        # read images
        img = Image.open(self.img[index]).convert("L")
        label = Image.open(self.label[index]).convert("L")
        
        img = np.asarray(img)
        label = np.asarray(label)
        # data augmentation
        if(self.aug==True):
            img = cv2.copyMakeBorder(img,32,32,32,32,cv2.BORDER_REFLECT)
            label = cv2.copyMakeBorder(label,32,32,32,32,cv2.BORDER_REFLECT)

        if self.transforms:
            img = self.transforms(img)
            label = self.transforms(label)
            
        return img, label