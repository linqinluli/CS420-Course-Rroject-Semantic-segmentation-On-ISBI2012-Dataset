#%%
import os
from numpy.lib.function_base import append
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

from torchvision import transforms

from torch.utils.data import DataLoader
from model import Unet
from ISBIdataset import ISBIDataset

import warnings
import numpy as np
import imagej
import re

from torch.utils.data import DataLoader
from PIL import Image

warnings.filterwarnings("ignore")
transform = transforms.Compose([transforms.ToTensor()])

# define the path of data set
predict_test_dir = 'dataset/predict_test/'
train_img_dir = 'dataset/train_img/'
test__img_dir = 'dataset/test_img/'
train_label_dir = 'dataset/train_label/'
test_label_dir = 'dataset/test_label/'

# use ij to evaluate the performance by Vrand and Vinfo. These are the scripts
ij = imagej.init('Fiji.app')

Language_extension = "BeanShell"

macroVRand = """
import trainableSegmentation.metrics.*;
#@output String VRand
import ij.IJ;
originalLabels=IJ.openImage("AAAAA");
proposedLabels=IJ.openImage("BBBBB");
metric = new RandError( originalLabels, proposedLabels );
maxThres = 1.0;
maxScore = metric.getMaximalVRandAfterThinning( 0.0, maxThres, 0.1, true );  
VRand = maxScore;
"""

macroVInfo = """
import trainableSegmentation.metrics.*;
#@output String VInfo
import ij.IJ;
originalLabels=IJ.openImage("AAAAA");
proposedLabels=IJ.openImage("BBBBB");
metric = new VariationOfInformation( originalLabels, proposedLabels );
maxThres =1.0;
maxScore = metric.getMaximalVInfoAfterThinning( 0.0, maxThres, 0.1 );  
VInfo = maxScore;
"""


# evaluate function, input: image path and label path, output: two scores
def evl(image_path, label_path):
    reg1 = re.compile('AAAAA')
    macror = reg1.sub(label_path, macroVRand)
    macroi = reg1.sub(label_path, macroVInfo)

    reg2 = re.compile('BBBBB')
    macror = reg2.sub(image_path, macror)
    macroi = reg2.sub(image_path, macroi)

    VRand = float(
        str(ij.py.run_script(Language_extension, macror).getOutput('VRand')))
    VInfo = float(
        str(ij.py.run_script(Language_extension, macroi).getOutput('VInfo')))

    return VRand, VInfo


# evaluate all test images
def evl_all():
    # define test dataset
    test_set = ISBIDataset(test__img_dir, test_label_dir, transforms=transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    vrand_list = []
    vinfo_list = []
    #predict
    for index, (img, label) in enumerate(loader):
        predict_img_name = predict_test_dir + str(index) + '.png'
        label_img_name = test_label_dir + str(index) + '.png'

        vrand, vinfo = evl(predict_img_name, label_img_name)
        vrand_list.append(vrand)
        vinfo_list.append(vinfo)
    # return the mean scores of all test images
    return np.mean(vrand_list), np.mean(vinfo_list)
#%%
vrand, vinfo = evl_all()
print('vrand = ', vrand, 'vinfo = ', vinfo)
#%%