import matplotlib.pyplot as plt
import os
import cv2
import matplotlib.pyplot as plt
import os, time
import itertools
import pickle
import argparse
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
from torch.autograd import Variable
from torch import autograd
from skimage.color import rgb2gray
from utils import canny_edge_dector, canny_utils


# canny extract edge
def get_edge(img_path):
    imgs, name_list = canny_utils.load_data(img_path)
    ced = canny_edge_dector.cannyEdgeDetector(imgs, sigma=2, kernel_size=5, lowthreshold=0.1, highthreshold=0.2,
                                              weak_pixel=100)
    edge_map = np.array(ced.detect(save_dir='../visualization/fashion', name_list=name_list))



def show_edge(edge_img):
    plt.figure(figsize=(8, 8))
    plt.imshow(edge_img, cmap=plt.cm.gray)
    plt.show()


def test_code(path='../src/FaceSample'):
    '''
        img = cv2.imread(path)
    img = np.transpose(img, (2, 0, 1))
    print(img.shape)
    img_tensor = torch.tensor(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    edges = forward_canny.get_edge(img_tensor).cpu().squeeze()
    show_edge(edges)
    '''
    get_edge(img_path='../visualization/fashion/fashion1')


if __name__ == '__main__':
    test_code()
