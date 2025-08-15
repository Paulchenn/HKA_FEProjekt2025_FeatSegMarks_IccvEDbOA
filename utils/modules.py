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
        images = images.cpu().numpy()
        edges = []
        for i in range(images.shape[0]):
            img = images[i]

            img = img * 0.5 + 0.5
            img_gray = rgb2gray(np.transpose(img, (1, 2, 0)))
            edge = canny(np.array(img_gray), sigma=sigma, high_threshold=high_threshold,
                        low_threshold=low_threshold).astype(float)
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

        gray = 0.299 * R + 0.587 * G + 0.114 * B

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
        mat1 = torch.cat([gray[:, :, 0, :].unsqueeze(2), gray], 2)[:, :, :gray.shape[2], :]
        mat2 = torch.cat([gray, gray[:, :, gray.shape[2] - 1, :].unsqueeze(2)], 2)[:, :, 1:, :]
        mat3 = torch.cat([gray[:, :, :, 0].unsqueeze(3), gray], 3)[:, :, :, :gray.shape[3]]
        mat4 = torch.cat([gray, gray[:, :, :, gray.shape[3] - 1].unsqueeze(3)], 3)[:, :, :, 1:]
        info_rec = (gray - mat1) ** 2 + (gray - mat2) ** 2 + (gray - mat3) ** 2 + (gray - mat4) ** 2
        info_rec_ave = info_rec.view(batch_size, -1)
        ave = torch.mean(info_rec_ave, dim=1)
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
        robustCanny = self.get_edge(
            img,
            sigma,
            high_threshold,
            low_threshold
        )

        selfInfoMap = self.get_info(img)

        robustCanny_norm = torch.where(robustCanny < 0, 0., 1.)

        selfInfoMap_inv = selfInfoMap * -1
        selfInfoMap_inv_norm = torch.where(selfInfoMap_inv < 0, 0., 1.)

        edgeMap_1 = robustCanny_norm + selfInfoMap_inv_norm
        edgeMap_2 = torch.cat([edgeMap_1, edgeMap_1, edgeMap_1], 1).detach().to(self.config.DEVICE)

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
        # z: [B, 100] – passt zur neuen Generator-Signatur Generator(z, label, bc)
        z_ = torch.randn(mn_batch, 100, 1, 1, device=self.config.DEVICE)
        G_result = netG(z_, deformedImg, blurImg)
        return G_result
    
    def getDResult(
        self,
        img,
        netD
    ):
        # Patch-Logits (B,1,H',W') und optional Aux-Logits
        patch_logits, aux_output = netD(img)
        # stabil: Mittel über alle Patches/Batch
        D_mean = patch_logits.mean()
        return D_mean, aux_output
    

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

        autocast_ctx = nullcontext()
        loss = SimpleNamespace()
        mn_batch = img.shape[0]

        # === Step 1: blurred texture map ===
        img_blur = blur_image(img, config.DOWN_SIZE)

        # === Step 2: Train D (Hinge) ===
        time_startTrainD = time.time()
        with nullcontext():
            G_rough = self.generateImg(mn_batch, netG, e_extend, img_blur)

            D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
            D_result_roughImg, aux_output_roughImg = self.getDResult(G_rough.detach(), netD)

            # Aux‑Loss nur, wenn Aux‑Kopf aktiv
            D_celoss = CE_loss(aux_output_realImg, label) if aux_output_realImg is not None else torch.tensor(0.0, device=self.config.DEVICE)

            # Hinge: (-real + fake)
            loss.D_loss = -D_result_realImg + D_result_roughImg + 0.5 * D_celoss

        netD.zero_grad(set_to_none=True)
        optimD.zero_grad(set_to_none=True)
        scaler.scale(loss.D_loss).backward()
        scaler.step(optimD)
        scaler.update()
        time_TSG.time_trainD.append(time.time() - time_startTrainD)

        # === Step 3: Train G (Hinge) ===
        time_startTrainG1 = time.time()
        with autocast_ctx:
            # erneute D-Auswertung auf G_rough (ohne detach)
            D_result_roughImg, aux_output_roughImg = self.getDResult(G_rough, netD)

            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            G_L1_loss_rough = L1_loss(G_rough, img_for_loss)

            # Aux‑Loss nur, falls Aux‑Kopf existiert
            G_celoss_rough = CE_loss(aux_output_roughImg, label) if aux_output_roughImg is not None else torch.tensor(0.0, device=self.config.DEVICE)

            lambda_L1 = config.LAMBDA_S1_L1
            lamda_GAN = config.LAMBDA_S1_GAN
            lambda_CE = config.LAMBDA_S1_CE
            loss.G_loss = (
                lambda_L1 * G_L1_loss_rough
                - lamda_GAN * D_result_roughImg
                + lambda_CE * G_celoss_rough
            )

        netG.zero_grad(set_to_none=True)
        optimG.zero_grad(set_to_none=True)
        scaler.scale(loss.G_loss).backward()
        scaler.step(optimG)
        scaler.update()
        time_TSG.time_trainG1.append(time.time() - time_startTrainG1)

        if config.SHOW_TSG and iteration % config.SHOW_IMAGES_INTERVAL == 0:
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
        
        autocast_ctx = nullcontext()
        loss = SimpleNamespace()
        mn_batch = img.shape[0]

        img_blur = blur_image(img, config.DOWN_SIZE)

        # === Train D (Hinge) ===
        time_startTrainD = time.time()
        with nullcontext():
            G_fine = self.generateImg(mn_batch, netG, e_deformed, img_blur)

            D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
            D_result_fineImg, aux_output_fineImg = self.getDResult(G_fine.detach(), netD)

            D_celoss = CE_loss(aux_output_realImg, label) if aux_output_realImg is not None else torch.tensor(0.0, device=self.config.DEVICE)

            loss.D_loss = -D_result_realImg + D_result_fineImg + 0.5 * D_celoss

        netD.zero_grad(set_to_none=True)
        optimD.zero_grad(set_to_none=True)
        scaler.scale(loss.D_loss).backward()
        scaler.step(optimD)
        scaler.update()
        time_TSG.time_trainD.append(time.time() - time_startTrainD)

        # === Train G ===
        time_startTrainG2 = time.time()
        with autocast_ctx:
            # Classifier Loss (optional)
            time_startTrainCls = time.time()
            if config.TRAIN_WITH_CLS:
                G_fine_resized = F.interpolate(G_fine, size=(299, 299), mode='bilinear', align_corners=False)
                if G_fine_resized.shape[1] == 1:
                    G_fine_resized = G_fine_resized.repeat(1, 3, 1, 1)
                G_fine_norm = transform_cls(G_fine_resized)
                cls_output = cls(G_fine_norm)
                loss.cls_loss = CE_loss(cls_output, label)
            else:
                loss.cls_loss = torch.tensor(0.0, device=self.config.DEVICE)
            time_TSG.time_trainCls.append(time.time() - time_startTrainCls)

            # Edge preservation
            e_extend_G_fine = emse.doEMSE(G_fine.detach())
            edge_loss_2 = L1_loss(e_extend_G_fine, e_deformed)

            # GAN + Aux
            D_result_fineImg, aux_output_fineImg = self.getDResult(G_fine, netD)
            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            G_L1_loss_fine = L1_loss(G_fine, img_for_loss)
            G_celoss_fine = CE_loss(aux_output_fineImg, label) if aux_output_fineImg is not None else torch.tensor(0.0, device=self.config.DEVICE)

            lambda_L1 = config.LAMBDA_S2_L1
            lamda_GAN = config.LAMBDA_S2_GAN
            lambda_CE = config.LAMBDA_S2_CE
            lambda_cls = config.LAMBDA_S2_CLS
            lambda_edge = config.LAMBDA_S2_EDGE
            loss.G_loss = (
                lambda_L1 * G_L1_loss_fine
                - lamda_GAN * D_result_fineImg
                + lambda_CE * G_celoss_fine
                + lambda_cls * loss.cls_loss
                + lambda_edge * edge_loss_2
            )

        netG.zero_grad(set_to_none=True)
        optimG.zero_grad(set_to_none=True)
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

        loss = SimpleNamespace()
        mn_batch = img.shape[0]

        img_blur = blur_image(img, config.DOWN_SIZE)

        # Generate rough image
        G_rough = self.generateImg(mn_batch, netG, e_extend, img_blur).contiguous()

        # D on real/fake
        D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
        D_result_roughImg, aux_output_roughImg = self.getDResult(G_rough, netD)

        D_celoss = CE_loss(aux_output_realImg, label) if aux_output_realImg is not None else torch.tensor(0.0, device=self.config.DEVICE)
        loss.D_loss = -D_result_realImg + D_result_roughImg + 0.5 * D_celoss

        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss_rough = L1_loss(G_rough, img_for_loss)
        G_celoss_rough = CE_loss(aux_output_roughImg, label) if aux_output_roughImg is not None else torch.tensor(0.0, device=self.config.DEVICE)

        lambda_L1 = config.LAMBDA_S1_L1
        lamda_GAN = config.LAMBDA_S1_GAN
        lambda_CE = config.LAMBDA_S1_CE
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

        loss = SimpleNamespace()
        mn_batch = img.shape[0]

        img_blur = blur_image(img, config.DOWN_SIZE)

        G_fine = self.generateImg(mn_batch, netG, e_deformed, img_blur).contiguous()

        D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
        D_result_fineImg, aux_output_fineImg = self.getDResult(G_fine, netD)

        D_celoss = CE_loss(aux_output_realImg, label) if aux_output_realImg is not None else torch.tensor(0.0, device=self.config.DEVICE)
        loss.D_loss = -D_result_realImg + D_result_fineImg + 0.5*D_celoss

        # Classifier
        G_fine_resized = F.interpolate(G_fine, size=(299, 299), mode='bilinear', align_corners=False)
        if G_fine_resized.shape[1] == 1:
            G_fine_resized = G_fine_resized.repeat(1, 3, 1, 1)
        G_fine_norm = transform_cls(G_fine_resized)
        cls_output = cls(G_fine_norm)
        loss.cls_loss = CE_loss(cls_output, label)
        cls_prediction = torch.argmax(cls_output, dim=1)

        # Edge preservation
        emse_G_fine = emse.doEMSE(G_fine.detach())
        edge_loss_2 = L1_loss(emse_G_fine, e_deformed)

        # Generator loss
        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss_fine = L1_loss(G_fine, img_for_loss)
        G_celoss_fine = CE_loss(aux_output_fineImg, label) if aux_output_fineImg is not None else torch.tensor(0.0, device=self.config.DEVICE)

        lambda_L1 = config.LAMBDA_S2_L1
        lamda_GAN = config.LAMBDA_S2_GAN
        lambda_CE = config.LAMBDA_S2_CE
        lambda_cls = config.LAMBDA_S2_CLS
        lambda_edge = config.LAMBDA_S2_EDGE
        loss.G_loss = (
            lambda_L1 * G_L1_loss_fine
            - lamda_GAN * D_result_fineImg
            + lambda_CE * G_celoss_fine
            + lambda_cls * loss.cls_loss
            + lambda_edge * edge_loss_2
        )

        return loss, cls_prediction
