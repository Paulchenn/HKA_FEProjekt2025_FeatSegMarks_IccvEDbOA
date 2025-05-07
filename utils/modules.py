import json
import numpy as np
import canny
import itertools

from skimage.color import rgb2gray
from tps_grid_gen import TPSGridGen

import torch
from torch.autograd import Variable
import torch.nn.functional as F

# --- Load config ---
with open('config/config.json', 'r') as f:
    config = json.load(f)

class EMSE:
    """
    Edge map-based shape encoding (EMSE)
    """

    def __init__(self):
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    def get_edge(self, images, sigma=1.0, high_threshold=0.3, low_threshold=0.2):
        # median = kornia.filters.MedianBlur((3,3))
        # for i in range(3):
        #     images = median(images)

        images = images.cpu().numpy()
        edges = []
        for i in range(images.shape[0]):
            img = images[i]

            img = img * 0.5 + 0.5
            img_gray = rgb2gray(np.transpose(img, (1, 2, 0)))
            edge = canny(np.array(img_gray), sigma=sigma, high_threshold=high_threshold,
                        low_threshold=low_threshold).astype(float)
            # edge = Image.fromarray((edge * 255.).astype(np.int8), mode='L')
            edge = (edge - 0.5) / 0.5
            edges.append([edge])
        edges = np.array(edges).astype('float32')
        edges = torch.from_numpy(edges).to(self.device)
        return edges
    
    def rgb2Gray_batch(self, input):
        R = input[:, 0]
        G = input[:, 1]
        B = input[:, 2]
        input[:, 0] = 0.299 * R + 0.587 * G + 0.114 * B
        input = input[:, 0]

        input = input.view(input.shape[0], 1, 32, 32)
        return input
    
    def get_info(self, input, batch_size):
        gray = self.rgb2Gray_batch(input)
        # gray = input
        mat1 = torch.cat([gray[:, :, 0, :].unsqueeze(2), gray], 2)[:, :, :gray.shape[2], :]
        mat2 = torch.cat([gray, gray[:, :, gray.shape[2] - 1, :].unsqueeze(2)], 2)[:, :, 1:, :]
        mat3 = torch.cat([gray[:, :, :, 0].unsqueeze(3), gray], 3)[:, :, :, :gray.shape[3]]
        mat4 = torch.cat([gray, gray[:, :, :, gray.shape[3] - 1].unsqueeze(3)], 3)[:, :, :, 1:]
        info_rec = (gray - mat1) ** 2 + (gray - mat2) ** 2 + (gray - mat3) ** 2 + (gray - mat4) ** 2
        info_rec_ave = info_rec.view(batch_size, -1)
        ave = torch.mean(info_rec_ave, dim=1)
        # info = torch.zeros(gray.shape, dtype=torch.float32)
        tmp = torch.zeros(gray.shape).to(self.device)
        for b in range(input.shape[0]):
            tmp[b] = ave[b]
        info = torch.where(info_rec > tmp, 1.0, 0.0)

        return info
    

class TSD:
    """
    TPS-based shape deformation (TSD)
    TPS - Thin-plate spline
    """

    def __init__(self):
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    def grid_sample(self, input, grid, canvas=None):
        output = F.grid_sample(input, grid, align_corners=True).to(self.device)
        if canvas is None:
            return output
        else:
            input_mask = Variable(input.data.new(input.size()).fill_(1).to(self.device))
            output_mask = F.grid_sample(input_mask, grid, align_corners=True)
            padded_output = output * output_mask + canvas * (1 - output_mask)
            return padded_output

    def TPS_Batch(self, imgs):
        height, width = imgs.shape[3], imgs.shape[2]
        tps_img = []
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :]
            img = img.unsqueeze(0)
            target_control_points = torch.Tensor(list(itertools.product(
                torch.arange(-1.0, 1.00001, 2.0 / 4),
                torch.arange(-1.0, 1.00001, 2.0 / 4),
            ))).to(self.device)
            source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1).to(self.device)
            # source_control_points = target_control_points + 0.01*torch.ones(target_control_points.size()).to(device)
            tps = TPSGridGen(height, width, target_control_points)
            source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))

            grid = source_coordinate.view(1, height, width, 2)
            canvas = Variable(torch.Tensor(1, 3, height, width).fill_(1.0)).to(self.device)
            target_image = self.grid_sample(img, grid, canvas)
            tps_img.append(target_image)
        tps_img = torch.cat(tps_img, dim=0)
        return tps_img