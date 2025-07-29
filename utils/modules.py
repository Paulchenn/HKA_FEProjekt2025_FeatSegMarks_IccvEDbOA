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

        x = torch.linspace(-1.0, 1.0, steps=5)
        y = torch.linspace(-1.0, 1.0, steps=5)
        self.target_control_points = torch.tensor(list(itertools.product(x, y)), device=self.config.DEVICE)

        # We need height and width later — we initialize tps at the first doTSD call or specify height/width during construction with
        self.tps = None

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

        # Create TPSGridGen only once
        if self.tps is None:
            self.tps = TPSGridGen(self.config, height, width, self.target_control_points)

        tps_img = []
        for i in range(imgs.shape[0]):
            img = imgs[i, :, :, :].unsqueeze(0)
            
            source_control_points = self.target_control_points + torch.Tensor(self.target_control_points.size()).uniform_(-0.1, 0.1).to(self.config.DEVICE)

            source_coordinate = self.tps(torch.unsqueeze(source_control_points, 0))

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
        print("img contiguous in getDResult:", img.is_contiguous())
        D_result, aux_output = netD(img)
        D_result = D_result.squeeze()

        D_result_1 = D_result.mean()

        return D_result_1, aux_output
    
    def doTSG_training(
        self,
        config,
        emse,
        img,
        label,
        e_extend,
        e_deformed,
        netD,
        netG,
        cls,
        optimD,
        optimG,
        optimC,
        CE_loss,
        L1_loss,
        time_TSG,
        scaler,
        downSize=12
    ):
        # Initialize loss variables
        loss = SimpleNamespace()

        # pdb.set_trace()
        mn_batch = img.shape[0]

        # === Step 0: Precompute blurred texture map Itxt ===
        img_blur = self.blur_image(img, downSize)  # corresponds to Itxt in the paper

        # === Step 1: Phase 1 – Rough Generation (Eextend + Itxt) ===
        # Generate rough image (Stage 1)
        G_rough = self.generateImg(mn_batch, netG, e_extend, img_blur)  # Input: edge + blurred image

        # Discriminator output on real and fake
        D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
        D_result_roughImg, aux_output_roughImg = self.getDResult(G_rough, netD)

        # === Train Discriminator (Stage 1) ===
        netG.zero_grad()
        netD.zero_grad()
        time_startTrainD = time.time()
        D_celoss = CE_loss(aux_output_realImg, label)
        loss.stage1_D_loss = (-D_result_realImg + D_result_roughImg) + 0.5 * D_celoss
        #scaler.scale(loss.stage1_D_loss).backward(retain_graph=True)
        #scaler.step(optimD)
        #scaler.update()
        loss.stage1_D_loss.backward(retain_graph=True)
        optimD.step()
        time_TSG.time_trainD.append(time.time() - time_startTrainD)

        # === Train Generator (Stage 1) ===
        netG.zero_grad()
        netD.zero_grad()
        time_startTrainG1 = time.time()
        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss_rough = L1_loss(G_rough, img_for_loss)
        G_celoss_rough = CE_loss(aux_output_roughImg.detach(), label).sum()

        # Total generator loss: L1 + GAN + classification (no edge loss in phase 1)
        loss.stage1_G_loss = G_L1_loss_rough - D_result_roughImg.detach() + 0.5 * G_celoss_rough
        #scaler.scale(loss.stage1_G_loss).backward()
        #scaler.step(optimG)
        #scaler.update()
        loss.stage1_G_loss.backward()
        optimG.step()
        time_TSG.time_trainG1.append(time.time() - time_startTrainG1)

        # === Step 2: Phase 2 – Fine Tuning (Edeform + Itxt) ===
        netG.zero_grad()
        netD.zero_grad()
        cls.eval()

        # Generate fine image from deformed edge map
        G_fine = self.generateImg(mn_batch, netG, e_deformed, img_blur)
        D_result_fineImg, aux_output_fineImg = self.getDResult(G_fine, netD)

        # === Classifier Loss ===
        time_startTrainCls = time.time()
        if config.TRAIN_CLS:
            G_fine_resized = F.interpolate(G_fine, size=(299, 299), mode='bilinear', align_corners=False)
            if G_fine_resized.shape[1] == 1:
                G_fine_resized = G_fine_resized.repeat(1, 3, 1, 1)
            G_fine_norm = self.myNormalize()(G_fine_resized)
            cls_output = cls(G_fine_norm.detach())
            #G_fine_norm = self.myNormalize()(G_fine_resized)
            #with torch.cuda.amp.autocast(enabled=False):
            #    cls_output = cls(G_fine_norm.detach().float())
            loss.cls_loss = CE_loss(cls_output, label)
        else:
            loss.cls_loss = torch.tensor(0.0).to(self.config.DEVICE)
        time_TSG.time_trainCls.append(time.time() - time_startTrainCls)

        # === Edge preservation loss (Ledge) ===
        edge_map_from_syn = emse.get_info(G_fine)
        edge_loss = L1_loss(edge_map_from_syn, e_deformed[:, 0:1, :, :])  # compare gray channels

        # === Generator loss (Phase 2) ===
        netG.zero_grad()
        netD.zero_grad()
        time_startTrainG2 = time.time()
        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss_fine = L1_loss(G_fine, img_for_loss)
        G_celoss_fine = CE_loss(aux_output_fineImg.detach(), label).sum()

        # Total loss = GAN + Classification + Shape-Preservation
        loss.stage2_G_loss = G_L1_loss_fine - D_result_fineImg.detach() + G_celoss_fine + edge_loss + loss.cls_loss
        #scaler.scale(loss.stage2_G_loss).backward()
        #scaler.step(optimG)
        #scaler.update()
        loss.stage2_G_loss.backward()
        optimG.step()
        time_TSG.time_trainG2.append(time.time() - time_startTrainG2)

        return netD, netG, cls, optimD, optimG, optimC, CE_loss, L1_loss, loss, time_TSG, scaler
    
    def doTSG_testing(
        self,
        config,
        emse,
        img,
        label,
        e_extend,
        e_deformed,
        netD,
        netG,
        cls,
        CE_loss,
        L1_loss,
        downSize=12
    ):
        print("G_rough contiguous after generateImg:", e_extend.is_contiguous())
        print("G_rough contiguous after generateImg:", e_deformed.is_contiguous())

        # Initialize loss variables
        loss = SimpleNamespace()

        # pdb.set_trace()
        mn_batch = img.shape[0]

        # === Step 0: Precompute blurred texture map Itxt ===
        img_blur = self.blur_image(img, downSize)  # corresponds to Itxt in the paper

        # === Step 1: Phase 1 – Rough Generation (Eextend + Itxt) ===
        # Generate rough image (Stage 1)
        G_rough = self.generateImg(mn_batch, netG, e_extend, img_blur)  # Input: edge + blurred image
        print("G_rough contiguous after generateImg:", G_rough.is_contiguous())

        # Discriminator output on real and fake
        print(f"img-shape: {img.shape}")
        print(f"G-rough-shape: {G_rough.shape}")
        D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
        D_result_roughImg, aux_output_roughImg = self.getDResult(G_rough, netD)

        # === Train Discriminator (Stage 1) ===
        netG.eval()
        netD.eval()
        D_celoss = CE_loss(aux_output_realImg, label)
        loss.stage1_D_loss = (-D_result_realImg + D_result_roughImg) + 0.5 * D_celoss

        # === Train Generator (Stage 1) ===
        netG.eval()
        netD.eval()
        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss_rough = L1_loss(G_rough, img_for_loss)
        G_celoss_rough = CE_loss(aux_output_roughImg.detach(), label).sum()

        # Total generator loss: L1 + GAN + classification (no edge loss in phase 1)
        loss.stage1_G_loss = G_L1_loss_rough - D_result_roughImg.detach() + 0.5 * G_celoss_rough

        # === Step 2: Phase 2 – Fine Tuning (Edeform + Itxt) ===
        netG.eval()
        netD.eval()
        cls.eval()

        # Generate fine image from deformed edge map
        G_fine = self.generateImg(mn_batch, netG, e_deformed, img_blur)
        D_result_fineImg, aux_output_fineImg = self.getDResult(G_fine, netD)

        # === Classifier Loss ===
        G_fine_resized = F.interpolate(G_fine, size=(299, 299), mode='bilinear', align_corners=False)
        if G_fine_resized.shape[1] == 1:
            G_fine_resized = G_fine_resized.repeat(1, 3, 1, 1)
        G_fine_norm = self.myNormalize()(G_fine_resized)
        cls_output = cls(G_fine_norm.detach())
        loss.cls_loss = CE_loss(cls_output, label)
        cls_prediction = torch.argmax(cls_output, dim=1)

        # === Edge preservation loss (Ledge) ===
        edge_map_from_syn = emse.get_info(G_fine)
        edge_loss = L1_loss(edge_map_from_syn, e_deformed[:, 0:1, :, :])  # compare gray channels

        # === Generator loss (Phase 2) ===
        netG.eval()
        netD.eval()
        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss_fine = L1_loss(G_fine, img_for_loss)
        G_celoss_fine = CE_loss(aux_output_fineImg.detach(), label).sum()

        # Total loss = GAN + Classification + Shape-Preservation
        loss.stage2_G_loss = G_L1_loss_fine - D_result_fineImg.detach() + G_celoss_fine + edge_loss + loss.cls_loss

        return loss, cls_prediction