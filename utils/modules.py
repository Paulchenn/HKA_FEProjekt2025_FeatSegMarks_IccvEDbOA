import json
from signal import pause
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
from contextlib import nullcontext
from utils.helpers import *

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch import amp


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
        robustCanny_norm = torch.where(robustCanny < 0, 0., 1.)

        # inverse and normalize the Self-Information guided map
        selfInfoMap_inv = selfInfoMap * -1
        selfInfoMap_inv_norm = torch.where(selfInfoMap_inv < 0, 0., 1.)

        # get the final edge map
        edgeMap_1 = robustCanny_norm + selfInfoMap_inv_norm
        edgeMap_2 = torch.cat([edgeMap_1, edgeMap_1, edgeMap_1], 1).detach().to(self.config.DEVICE)

        #show_tsg(selfInfoMap, selfInfoMap_inv, selfInfoMap_inv_norm, robustCanny, robustCanny_norm, edgeMap_1, edgeMap_2, )

        #pdb.set_trace()

        return edgeMap_2
    

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
    
    def generateImg(
        self,
        mn_batch,
        netG,
        deformedImg,
        blurImg
    ):
        z_ = Variable(torch.randn((mn_batch, 100)).view(-1, 100, 1, 1).to(self.config.DEVICE))

        G_result = netG(z_, deformedImg, blurImg)

        # G_result = (G_result + 1) / 2   # scaling from [-1, 1] to [0, 1] due to tanh as last layer in netG

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
    

    def doTSG_stage1_training(
        self,
        iteration,
        config,
        img,
        label,
        e_extend,
        netD,
        netG,
        optimD,
        optimG,
        CE_loss,
        L1_loss,
        time_TSG,
        scaler
    ):
        '''
        Function for stage 1 training of GAN with rough image (extented edge map and blured image).
        '''

        # === Stage 1: Step 0: Preparations ===
        #pdb.set_trace()
        autocast_ctx = nullcontext() #amp.autocast(device_type="cuda") if config.DEVICE.type != "cpu" else nullcontext()
        loss = SimpleNamespace() # Initialize loss variables
        mn_batch = img.shape[0] # get the batch size


        # === Stage 1: Step 1: Precompute blurred texture map Itxt ===
        img_blur = blur_image(img, config.DOWN_SIZE)  # corresponds to Itxt in the paper


        # === Stage 1: Step 2: Discriminator training with rough image ===
        time_startTrainD = time.time()

        with nullcontext():
            # Generate rough image
            #print(f"Got input: {mn_batch}, {e_extend.shape} and {img_blur.shape}")

            G_rough = self.generateImg(mn_batch, netG, e_extend, img_blur)  # Input: edge + blurred image

            # Discriminator output on real and fake
            D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
            D_result_realImg = D_result_realImg.mean()
            D_result_roughImg, aux_output_roughImg = self.getDResult(G_rough, netD)
            D_result_roughImg = D_result_roughImg.mean()
            D_celoss = CE_loss(aux_output_realImg, label)
            loss.D_loss = -D_result_realImg + D_result_roughImg + 0.5 * D_celoss

        # === Train Discriminator ===
        netD.zero_grad()
        optimD.zero_grad()
        scaler.scale(loss.D_loss).backward(retain_graph=True)
        scaler.step(optimD)
        scaler.update()

        # get time needed for disciminator training
        time_TSG.time_trainD.append(time.time() - time_startTrainD)


        # === Stage 1: Step 3: Train Generator with rough image ===
        time_startTrainG1 = time.time()

        with autocast_ctx:
            # call discriminator again due to inplace-error
            D_result_roughImg, aux_output_roughImg = self.getDResult(G_rough, netD)
            D_result_roughImg = D_result_roughImg.mean()
            
            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            G_L1_loss_rough = L1_loss(G_rough, img_for_loss)
            G_celoss_rough = CE_loss(aux_output_roughImg, label).sum()

            # Total generator loss: L1 + GAN + classification (no edge loss in phase 1)
            lambda_L1 = config.LAMBDA_S1_L1 # original: 1.0
            lamda_GAN = config.LAMBDA_S1_GAN # original: 1.0
            lambda_CE = config.LAMBDA_S1_CE # original: 0.5
            loss.G_loss = (
                lambda_L1 * G_L1_loss_rough
                - lamda_GAN * D_result_roughImg
                + lambda_CE * G_celoss_rough
            )

        netG.zero_grad()
        optimG.zero_grad()
        scaler.scale(loss.G_loss).backward()
        scaler.step(optimG)
        scaler.update()
        
        # get time needed for generator training in stage 1
        time_TSG.time_trainG1.append(time.time() - time_startTrainG1)

        if config.SHOW_TSG and iteration % config.SHOW_IMAGES_INTERVAL == 0:
            #pdb.set_trace()
            show_tsg(
                img=img,
                e_extend=e_extend,
                img_blur=img_blur,
                img_for_loss=img_for_loss,
                G_rough=G_rough,
            )

        return netD, netG, optimD, optimG, loss, time_TSG
    

    def doTSG_stage2_training(
        self,
        iteration,
        config,
        emse,
        img,
        label,
        e_deformed,
        netD,
        netG,
        cls,
        transform_cls,
        optimD,
        optimG,
        CE_loss,
        L1_loss,
        time_TSG,
        scaler
    ):
        '''
        Function for stage 2 training of GAN with fine image (extented and deformed edge map and blured image).
        '''
        
        # === Stage 2: Step 0: Preparations ===
        #pdb.set_trace()
        autocast_ctx = nullcontext() #amp.autocast(device_type="cuda") if config.DEVICE.type != "cpu" else nullcontext()
        loss = SimpleNamespace() # Initialize loss variables
        mn_batch = img.shape[0] # get the batch size


        # === Stage 2: Step 1: Precompute blurred texture map Itxt ===
        img_blur = blur_image(img, config.DOWN_SIZE)  # corresponds to Itxt in the paper


        # === Stage 2: Step 2: Discriminator training with fine image ===
        time_startTrainD = time.time()

        with nullcontext():
            # Generate fine image
            G_fine = self.generateImg(mn_batch, netG, e_deformed, img_blur)  # Input: edge deformed + blurred image

            # Discriminator output on real and fake
            D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
            D_result_realImg = D_result_realImg.mean()
            D_result_fineImg, aux_output_fineImg = self.getDResult(G_fine, netD)
            D_result_fineImg = D_result_fineImg.mean()
            D_celoss = CE_loss(aux_output_realImg, label)
            loss.D_loss = -D_result_realImg + D_result_fineImg + 0.5 * D_celoss

        # === Train Discriminator ===
        netD.zero_grad()
        optimD.zero_grad()
        scaler.scale(loss.D_loss).backward(retain_graph=True)
        scaler.step(optimD)
        scaler.update()

        # get time needed for disciminator training
        time_TSG.time_trainD.append(time.time() - time_startTrainD)


        # === Stage 2: Step 3: Generator training with fine image ===
        time_startTrainG2 = time.time()

        with autocast_ctx:
            # === Classifier Loss ===
            time_startTrainCls = time.time()
            if config.TRAIN_WITH_CLS:
                G_fine_resized = F.interpolate(G_fine, size=(299, 299), mode='bilinear', align_corners=False)
                if G_fine_resized.shape[1] == 1:
                    G_fine_resized = G_fine_resized.repeat(1, 3, 1, 1)
                G_fine_norm = transform_cls(G_fine_resized)
                cls_output = cls(G_fine_norm)
                loss.cls_loss = CE_loss(cls_output, label)
            else:
                loss.cls_loss = torch.tensor(0.0).to(config.DEVICE)
            time_TSG.time_trainCls.append(time.time() - time_startTrainCls)

            # === Edge preservation loss (Ledge) ===
            e_extend_G_fine = emse.doEMSE(G_fine.detach())
            edge_loss_2 = L1_loss(e_extend_G_fine, e_deformed)

            # === Generator loss ===
            # call discriminator again due to inplace-error
            D_result_fineImg, aux_output_fineImg = self.getDResult(G_fine, netD)
            D_result_fineImg = D_result_fineImg.mean()
            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            G_L1_loss_fine = L1_loss(G_fine, img_for_loss)
            G_celoss_fine = CE_loss(aux_output_fineImg, label).sum()

            # Total loss = GAN + Classification + Shape-Preservation
            lambda_L1 = config.LAMBDA_S2_L1 # original: 1.0
            lamda_GAN = config.LAMBDA_S2_GAN # original: 1.0
            lambda_CE = config.LAMBDA_S2_CE # original: 0.5
            lambda_cls = config.LAMBDA_S2_CLS # original: 1.0
            lambda_edge = config.LAMBDA_S2_EDGE # original: 1.0
            loss.G_loss = (
                lambda_L1 * G_L1_loss_fine
                - lamda_GAN * D_result_fineImg
                + lambda_CE * G_celoss_fine
                + lambda_cls * loss.cls_loss
                + lambda_edge * edge_loss_2
            )

        netG.zero_grad()
        optimG.zero_grad()
        scaler.scale(loss.G_loss).backward()
        scaler.step(optimG)
        scaler.update()
        
        time_TSG.time_trainG2.append(time.time() - time_startTrainG2)

        if config.SHOW_TSG and iteration % config.SHOW_IMAGES_INTERVAL == 0:
            show_tsg(
                img=img,
                e_deformed=e_deformed,
                img_blur=img_blur,
                img_for_loss=img_for_loss,
                G_fine=G_fine,
                G_fine_resized=G_fine_resized,
                G_fine_norm=G_fine_norm,
                e_extend_G_fine=e_extend_G_fine
            )
            #pdb.set_trace()

        return netD, netG, optimD, optimG, loss, time_TSG
    

    def doTSG_stage1_testing(
        self,
        config,
        img,
        label,
        e_extend,
        netD,
        netG,
        CE_loss,
        L1_loss
    ):
        '''
        Function for stage 1 validation of GAN with rough image (extented edge map and blured image).
        '''
        with torch.no_grad():
            netG.eval()
            netD.eval()

            # === Stage 1: Step 0: Preparations
            # pdb.set_trace()
            loss = SimpleNamespace()
            mn_batch = img.shape[0]


            # === Stage 1: Step 1: Precompute blurred texture map Itxt ===
            img_blur = blur_image(img, config.DOWN_SIZE)  # corresponds to Itxt in the paper


            # === Stage 1: Step 2: Discriminator validation ===
            # Generate rough image from extended edge map and blurred image
            #print(f"Got input: {mn_batch}, {e_extend.shape} and {img_blur.shape}")
            G_rough = self.generateImg(mn_batch, netG, e_extend, img_blur)  # Input: edge + blurred image
            G_rough = G_rough.contiguous()

            # Discriminator output on real and fake
            D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
            D_result_realImg = D_result_realImg.mean()
            D_result_roughImg, aux_output_roughImg = self.getDResult(G_rough, netD)
            D_result_roughImg = D_result_roughImg.mean()

            # === Validate Discriminator ===
            D_celoss = CE_loss(aux_output_realImg, label)
            loss.D_loss = -D_result_realImg + D_result_roughImg + 0.5 * D_celoss


            # === Stage 2: Step 3: Generator validation ===
            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            G_L1_loss_rough = L1_loss(G_rough, img_for_loss)
            G_celoss_rough = CE_loss(aux_output_roughImg, label).sum()

            # Total generator loss: L1 + GAN + classification (no edge loss in phase 1)
            lambda_L1 = config.LAMBDA_S1_L1 # original: 1.0
            lamda_GAN = config.LAMBDA_S1_GAN # original: 1.0
            lambda_CE = config.LAMBDA_S1_CE # original: 0.5
            loss.G_loss = (
                lambda_L1 * G_L1_loss_rough
                - lamda_GAN * D_result_roughImg
                + lambda_CE * G_celoss_rough
            )

            return loss
    

    def doTSG_stage2_testing(
        self,
        config,
        emse,
        img,
        label,
        e_deformed,
        netD,
        netG,
        cls,
        transform_cls,
        CE_loss,
        L1_loss
    ):
        '''
        Function for stage 2 validation of GAN with fine image (extented and deformed edge map and blured image).
        '''
        with torch.no_grad():
            netG.eval()
            netD.eval()
            cls.eval()

            # === Stage 2: Step 0: Preparations ===
            # pdb.set_trace()
            loss = SimpleNamespace()
            mn_batch = img.shape[0]


            # === Stage 2: Step 1: Precompute blurred texture map Itxt ===
            img_blur = blur_image(img, config.DOWN_SIZE)  # corresponds to Itxt in the paper


            # === Stage 2: Step 2: Discriminator validation with fine image ===
            # Generate fine image from deformed edge map
            G_fine = self.generateImg(mn_batch, netG, e_deformed, img_blur)
            G_fine = G_fine.contiguous()

            # Discriminator output on real and fake
            D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
            D_result_realImg = D_result_realImg.mean()
            D_result_fineImg, aux_output_fineImg = self.getDResult(G_fine, netD)
            D_result_fineImg = D_result_fineImg.mean()

            # === Validate Discriminator ===
            D_celoss = CE_loss(aux_output_realImg, label)
            loss.D_loss = -D_result_realImg + D_result_fineImg + 0.5*D_celoss


            # === Stage 2: Step 3: Generator validation with fine image ===
            # === Classifier Loss ===
            G_fine_resized = F.interpolate(G_fine, size=(299, 299), mode='bilinear', align_corners=False)
            if G_fine_resized.shape[1] == 1:
                G_fine_resized = G_fine_resized.repeat(1, 3, 1, 1)
            G_fine_norm = transform_cls(G_fine_resized)
            cls_output = cls(G_fine_norm)
            loss.cls_loss = CE_loss(cls_output, label)
            cls_prediction = torch.argmax(cls_output, dim=1)

            # === Edge preservation loss (Ledge) ===
            emse_G_fine = emse.doEMSE(G_fine.detach())
            edge_loss_2 = L1_loss(emse_G_fine, e_deformed)

            # === Generator loss ===
            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            G_L1_loss_fine = L1_loss(G_fine, img_for_loss)
            G_celoss_fine = CE_loss(aux_output_fineImg, label).sum()

            # Total loss = GAN + Classification + Shape-Preservation
            lambda_L1 = config.LAMBDA_S2_L1 # original: 1.0
            lamda_GAN = config.LAMBDA_S2_GAN # original: 1.0
            lambda_CE = config.LAMBDA_S2_CE # original: 0.5
            lambda_cls = config.LAMBDA_S2_CLS # original: 1.0
            lambda_edge = config.LAMBDA_S2_EDGE # original: 1.0
            loss.G_loss = (
                lambda_L1 * G_L1_loss_fine
                - lamda_GAN * D_result_fineImg
                + lambda_CE * G_celoss_fine
                + lambda_cls * loss.cls_loss
                + lambda_edge * edge_loss_2
            )

            return loss, cls_prediction
