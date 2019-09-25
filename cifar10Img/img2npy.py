import numpy as np 
import os
from PIL import Image

imgHeight = 32
imgWidth = 32
imgPath = 'cifar10Img/img/'
imgList = os.listdir(imgPath)
trainData = np.zeros((len(imgList),imgHeight,imgWidth,3)).astype('float32')

for i in range(len(imgList)):
    img = Image.open(imgPath+imgList[i])
    imageData = np.array(img)
    trainData[i] = imageData

trainData = (255-trainData)/255
np.save('trainData32x32.npy',trainData)