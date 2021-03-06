import os
from numpy.lib.function_base import append

# allow scripts running
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from torch.utils.data import DataLoader
from model import Unet
from ISBIdataset import ISBIDataset

import warnings
import numpy as np
import imagej
import re
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import csv
warnings.filterwarnings("ignore")
transform = transforms.Compose([transforms.ToTensor()])

# define the path of data set
predict_test_dir = 'dataset/predict_test/'
train_img_dir = 'dataset/train_img/'
test_img_dir = 'dataset/test_img/'
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
def evl_all(umodel, aug=False):
    # define test dataset
    test_set = ISBIDataset(test_img_dir,
                           test_label_dir,
                           transforms=transform,
                           aug=aug)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    vrand_list = []
    vinfo_list = []
    #predict

    for index, (img, label) in enumerate(loader):
        predict_img = umodel.predict(img[0])
        # print(predict_img.shape)
        # print(label.shape)1

        # prepare for the evalution data
        predict_img_name = predict_test_dir + str(index) + '.png'

        label_img_name = test_label_dir + str(index) + '.png'

        predict_img = predict_img * 255
        outputImg = predict_img
        # ignore edge
        if (aug == True):
            outputImg = predict_img[32:predict_img.shape[0] - 32,
                                    32:predict_img.shape[0] - 32]
        outputImg = Image.fromarray(outputImg)

        outputImg = outputImg.convert('L')
        outputImg.save(predict_img_name)

        vrand, vinfo = evl(predict_img_name, label_img_name)
        vrand_list.append(vrand)
        vinfo_list.append(vinfo)
    # return the mean scores of all test images
    return np.mean(vrand_list), np.mean(vinfo_list)


# train function, batch_size, # epoch, learning rate can be adjusted.
def train(batch_size, n_epochs, learning_rate, aug):
    #define train dataset
    train_set = ISBIDataset(train_img_dir,
                            train_label_dir,
                            transforms=transform,
                            aug=aug)
    #define model
    umodel = Unet()
    umodel.cuda()
    #define loss function
    criterion = nn.BCEWithLogitsLoss()
    #define optimizer
    optimizer = optim.Adam(umodel.parameters(), lr=learning_rate)

    loss_history = []

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    #begin to train
    for epoch in range(n_epochs):
        running_loss = 0.0

        print('Epoch : {}/{}'.format(epoch + 1, n_epochs))
        print('-' * 10)

        for batch, (img, label) in enumerate(loader):
            img, label = Variable(img.cuda()), Variable(label.cuda())

            optimizer.zero_grad()
            output = umodel(img)
            loss = criterion(output, label)
            running_loss += loss.item()

            # print log every 5 batch
            if (batch % 5) == 4:
                print('\tBatch : {}/{}\tLoss : {:.4f}'.format(
                    batch + 1, len(loader), loss.item()))

            loss.backward()
            optimizer.step()
        vrand, vinfo = evl_all(umodel, aug)
        # save best model
        # the best scores in our prj is 0.98, so we set it as 0.98
        if (vrand > 0.98):
            torch.save(umodel.state_dict(), 'model/model.pkl')
        #save the logs
        with open("fn_log.csv", "a+", newline='') as file:
            csv_file = csv.writer(file)
            datas = [[
                'epoch:', epoch, 'vrand:', vrand, 'vinfo:', vinfo, 'loss',
                running_loss / len(loader)
            ]]
            print(datas)
            csv_file.writerows(datas)
        loss_history.append(running_loss / len(loader))


if __name__ == '__main__':
    #parameters initialization
    batch_size = 2
    n_epochs = 60
    learning_rate = 2e-3
    aug = True
    with open("log.csv", "a+", newline='') as file:
        csv_file = csv.writer(file)
        datas = [[
            'data augmentation (edges)', 'batch_size', batch_size, 'n_epoch',
            n_epochs, 'learning rate', learning_rate
        ]]
        print(datas)
        csv_file.writerows(datas)
    train(batch_size=batch_size,
          n_epochs=n_epochs,
          learning_rate=learning_rate,
          aug=aug)
