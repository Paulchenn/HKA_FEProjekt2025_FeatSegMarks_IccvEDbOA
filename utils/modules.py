import json
import numpy as np
import itertools
import pdb
import os
import time

from skimage.color import rgb2gray
from skimage.feature import canny
from utils.tps_grid_gen import TPSGridGen
from matplotlib import pyplot as plt
from types import SimpleNamespace

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms


class EMSE:
    """
    Edge map-based shape encoding (EMSE)
    """

    def __init__(self, config):
        self.config = config

    def get_edge(
        self,
        images,
        sigma=1.0,
        high_threshold=0.3,
        low_threshold=0.2
    ):
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
        edges = torch.from_numpy(edges).to(self.config.DEVICE)
        return edges
    
    def rgb2Gray_batch(
        self,
        input
    ):
        R = input[:, 0]
        G = input[:, 1]
        B = input[:, 2]

        # Convert RGB to grayscale using the luminosity method
        # Using the formula: gray = 0.299 * R + 0.587 * G + 0.114 * B
        # Factors come from ITU-R BT.601 standard for converting RGB to Y (luminance)
        # Note: In-place operations are not recommended for gradients, so we avoid them here
        gray = 0.299 * R + 0.587 * G + 0.114 * B  # No in-place operation

        # Dynamically calculate the height and width
        height = int((gray.numel() // gray.shape[0]) ** 0.5)
        width = height

        gray = gray.view(gray.shape[0], 1, height, width)
        return gray
        
    def get_info(
        self,
        input
    ):
        batch_size = input.shape[0]
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
        tmp = torch.zeros(gray.shape).to(self.config.DEVICE)
        for b in range(input.shape[0]):
            tmp[b] = ave[b]
        info = torch.where(info_rec > tmp, 1.0, 0.0)

        return info
    
    def doEMSE(
        self,
        img,
        sigma=1.0,
        high_threshold=0.3,
        low_threshold=0.2
    ):
        # get the edge map created by Canny
        robustCanny = self.get_edge(
            img,
            sigma,
            high_threshold,
            low_threshold
        )

        # get the self-information guided map
        selfInfoMap = self.get_info(img)

        # normalize the edge map created by Canny
        robustCanny = torch.where(robustCanny < 0, 0., 1.)

        # inverse and normalize the Self-Information guided map
        selfInfoMap = selfInfoMap * -1
        selfInfoMap = torch.where(selfInfoMap < 0, 0., 1.)

        # get the final edge map
        edgeMap = robustCanny + selfInfoMap
        edgeMap = torch.cat([edgeMap, edgeMap, edgeMap], 1).detach().to(self.config.DEVICE)

        return edgeMap
    

class TSD:
    """
    TPS-based shape deformation (TSD)
    TPS - Thin-plate spline
    """

    def __init__(self, config):
        self.config = config

    def grid_sample(
        self,
        input,
        grid,
        canvas=None
    ):
        output = F.grid_sample(input, grid, align_corners=True).to(self.config.DEVICE)
        if canvas is None:
            return output
        else:
            input_mask = Variable(input.data.new(input.size()).fill_(1).to(self.config.DEVICE))
            output_mask = F.grid_sample(input_mask, grid, align_corners=True)
            padded_output = output * output_mask + canvas * (1 - output_mask)
            return padded_output

    def TPS_Batch(
        self,
        imgs
    ):
        height, width = imgs.shape[2], imgs.shape[3]
        tps_img = []
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :]
            img = img.unsqueeze(0)
            target_control_points = torch.Tensor(list(itertools.product(
                torch.arange(-1.0, 1.00001, 2.0 / 4),
                torch.arange(-1.0, 1.00001, 2.0 / 4),
            ))).to(self.config.DEVICE)
            source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1).to(self.config.DEVICE)
            # source_control_points = target_control_points + 0.01*torch.ones(target_control_points.size()).to(device)
            tps = TPSGridGen(self.config, height, width, target_control_points)
            source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))

            grid = source_coordinate.view(1, height, width, 2)
            canvas = Variable(torch.Tensor(1, 3, height, width).fill_(1.0)).to(self.config.DEVICE)
            target_image = self.grid_sample(img, grid, canvas)
            tps_img.append(target_image)
        tps_img = torch.cat(tps_img, dim=0)
        return tps_img
    
    def doTSD(
        self,
        img
    ):
        # get the TPS-based shape deformation
        tpsImg = self.TPS_Batch(img)
        return tpsImg
    

class TSG:
    """
    Texture and shape-based generation (TSG)
    """

    def __init__(self, config):
        self.config = config

    def myNormalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def blur_image(
        self,
        img,
        downSize=12
    ):
        # blur image by downsampling and interpolation / upsampling

        # original image height and width
        height, width = img.shape[2], img.shape[3]
        if height < width:
            print("Denoising: Input Height smaler than input width!")
            print("Denoising: Taking smaler input height.")
            upSize = height
        elif width < height:
            print("Denoising: Input Height bigger than input width!")
            print("Denoising: Taking smaler input width.")
            upSize = width
        else:
            upSize = height


        # downsampling
        downsampler = transforms.Resize((downSize, downSize))
        downsampled = downsampler(img)

        # upsampling
        upsampler   = transforms.Resize((upSize, upSize))
        upsampled   = upsampler(downsampled)

        return upsampled
    
    def generateImg(
        self,
        mn_batch,
        netG,
        deformedImg,
        blurImg
    ):
        z_ = Variable(torch.randn((mn_batch, 100)).view(-1, 100, 1, 1).to(self.config.DEVICE))

        G_result = netG(z_, deformedImg, blurImg)

        return G_result
    
    def getDResult(
        self,
        img,
        netD
    ):
        D_result, aux_output = netD(img)
        D_result = D_result.squeeze()

        D_result_1 = D_result.mean()

        return D_result_1, aux_output
    
    def doTSG_training(
        self,
        config,
        emse,
        tsd,
        img,
        label,
        tpsImg,
        netD,
        netG,
        cls,
        optimD,
        optimG,
        optimC,
        CE_loss,
        L1_loss,
        scaler,
        time_TSG,
        downSize=12
    ):
        # Initialize loss variables
        loss = SimpleNamespace()

        # pdb.set_trace()
        mn_batch = img.shape[0]
        
        # blur image to minimize impact of texture
        img_blur = self.blur_image(img, downSize)

        # Zero the gradients
        netG.zero_grad()
        netD.zero_grad()
        
        # classifier in evaluation mode
        cls.eval()

        # Discriminator result of real image without blur
        D_result_realImg, aux_output_realImg = self.getDResult(img, netD)

        # Generate image and get Discriminator result
        G_result = self.generateImg(mn_batch, netG, tpsImg, img_blur)
        D_result_genImg, aux_output_genImg = self.getDResult(G_result, netD)

        # === Discriminator training ===
        time_startTrainD = time.time()
        # Compute the classification loss of discriminator
        D_celoss = CE_loss(aux_output_realImg, label)
        # Compute the whole loss of discriminator
        loss.D_loss = (-D_result_realImg + D_result_genImg) + 0.5 * D_celoss
        # Calculate gradient of discriminator
        scaler.scale(loss.D_loss).backward(retain_graph=True) #D_loss.backward()
        # Update the discriminator by calling optimizer
        scaler.step(optimD) #optimD.step()
        time_TSG.time_trainD.append(time.time() - time_startTrainD)

        # === Classifier loss ===
        time_startTrainCls = time.time()
        if config.TRAIN_CLS:
            # Resize to 299x299
            G_result_resized = F.interpolate(G_result, size=(299, 299), mode='bilinear', align_corners=False)
            # Normalize (if needed)
            normalize = self.myNormalize()
            # If Tensor [B, 1, H, W], duplicate up to 3-Channel
            if G_result_resized.shape[1] == 1:
                G_result_resized = G_result_resized.repeat(1, 3, 1, 1)
            # Apply normalization
            G_result_norm = normalize(G_result_resized) # torch.stack([normalize(img) for img in G_result_resized])
            # Compute the classification loss of classifier
            cls_output = cls(G_result_norm.detach())  # nur logits
            loss.cls_loss = CE_loss(cls_output, label)
        else:
            # Classifier only in Evaluation
            loss.cls_loss = torch.tensor(0.0).to(self.config.DEVICE)
        time_TSG.time_trainCls.append(time.time() - time_startTrainCls)

        # === Generator training ===
        time_startTrainG = time.time()
        # If Image is grayscale (only 1 channel), repeat / duplicate the channel to 3
        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss = L1_loss(G_result, img_for_loss)
        # Compute the classification loss of generator and sum them up
        aux_output_genImg_ = aux_output_genImg.detach()
        G_celoss = CE_loss(aux_output_genImg_, label).sum()
        # Loss of edge information
        combined_gray = tpsImg[:, 0:1, : , :]
        edge_loss = L1_loss(
            emse.get_info(G_result),
            combined_gray
        )
        # Compute the whole loss of generator
        G_loss_tot = G_L1_loss - D_result_genImg.detach() + 0.5 * G_celoss + edge_loss + loss.cls_loss
        # Calculate gradient of generator
        scaler.scale(G_loss_tot).backward() #G_loss_tot.backward()
        loss.G_loss_tot = G_loss_tot
        # Update the generator by calling optimizer
        scaler.step(optimG) #optimG.step()
        time_TSG.time_trainG2.append(time.time() - time_startTrainG)
        
        time_startScaler = time.time()
        scaler.update()
        time_TSG.time_Scaler.append(time.time() - time_startScaler)

        return netD, netG, cls, optimD, optimG, optimC, CE_loss, L1_loss, loss, scaler, time_TSG
    
    def doTSG_testing(
        self,
        config,
        emse,
        tsd,
        img,
        label,
        tpsImg,
        netD,
        netG,
        cls,
        CE_loss,
        L1_loss,
        downSize=12
    ):
        # Initialize loss variables
        loss = SimpleNamespace()

        # pdb.set_trace()
        mn_batch = img.shape[0]
        
        # blur image to minimize impact of texture
        img_blur = self.blur_image(img, downSize)

        # set netG, netD, cls to evaluation mode
        netG.eval()
        netD.eval()
        cls.eval()

        # Discriminator result of real image without blur
        D_result_realImg, aux_output_realImg = self.getDResult(img, netD)

        # Generate image and get Discriminator result
        G_result = self.generateImg(mn_batch, netG, tpsImg, img_blur)
        D_result_genImg, aux_output_genImg = self.getDResult(G_result, netD)

        # === Discriminator loss ===
        # Compute the classification loss of discriminator
        D_celoss = CE_loss(aux_output_realImg, label)
        # Compute the whole loss of discriminator
        loss.D_loss = (-D_result_realImg + D_result_genImg) + 0.5 * D_celoss

        # === Classifier loss ===
        if config.TRAIN_CLS:
            # Resize to 299x299
            G_result_resized = F.interpolate(G_result, size=(299, 299), mode='bilinear', align_corners=False)
            # Normalize (if needed)
            normalize = self.myNormalize()
            # If Tensor [B, 1, H, W], duplicate up to 3-Channel
            if G_result_resized.shape[1] == 1:
                G_result_resized = G_result_resized.repeat(1, 3, 1, 1)
            # Apply normalization
            G_result_norm = torch.stack([normalize(img) for img in G_result_resized])
            # Compute the classification loss of classifier
            cls_output = cls(G_result_norm)  # nur logits
            loss.cls_loss = CE_loss(cls_output, label)
            cls_prediction = torch.argmax(cls_output, dim=1)
        else:
            # Classifier only in Evaluation
            loss.cls_loss = torch.tensor(0.0).to(self.config.DEVICE)

        # === Generator loss ===
        # If Image is grayscale (only 1 channel), repeat / duplicate the channel to 3
        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss = L1_loss(G_result, img_for_loss)
        # Compute the classification loss of generator and sum them up
        G_celoss = CE_loss(aux_output_genImg, label).sum()
        # Loss of edge information
        combined_gray = tpsImg[:, 0:1, : , :]
        edge_loss = L1_loss(
            emse.get_info(G_result),
            combined_gray
        )
        # Compute the whole loss of generator
        loss.G_loss_tot = G_L1_loss - D_result_genImg + 0.5 * G_celoss + edge_loss + loss.cls_loss

        return loss, cls_prediction