import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import scipy.misc as sm

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(dir_name = 'faces_imgs'):
    '''
    Load images from the "faces_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    imgs = []
    cnt = 1
    name_list = os.listdir(dir_name)
    for filename in os.listdir(dir_name):
        print(filename)
        if os.path.isfile(dir_name + '/' + filename):
            print(cnt)
            # print(filename)
            cnt += 1
            img = mpimg.imread(dir_name + '/' + filename)
            # img = rgb2gray(img)
            imgs.append(img)
    print('load finished')
    return imgs, name_list
