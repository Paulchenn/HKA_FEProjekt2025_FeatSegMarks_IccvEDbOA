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
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torchvision import transforms
from torch import amp


class DS_EMSE:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _gauss_kernel(k=5, sigma=1.0, device="cpu", dtype=torch.float32):
        # separabler 1D-Kern, daraus 2D
        x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
        g1 = torch.exp(-0.5 * (x / sigma) ** 2)
        g1 = g1 / g1.sum()
        g2 = g1[:, None] @ g1[None, :]
        return g2

    @staticmethod
    def _morph_open(x, ks):
        # Opening auf "hellen" Strukturen: Erosion (min) -> Dilation (max)
        pad = ks // 2
        # Erosion ~ minpool = -maxpool(-x)
        eroded = -F.max_pool2d(-x, kernel_size=ks, stride=1, padding=pad)
        opened = F.max_pool2d(eroded, kernel_size=ks, stride=1, padding=pad)
        return opened

    @staticmethod
    def _morph_close(x, ks):
        # optional: Closing, falls du Lücken füllen willst (dicke Kanten verbinden)
        pad = ks // 2
        dilated = F.max_pool2d(x, kernel_size=ks, stride=1, padding=pad)
        closed = -F.max_pool2d(-dilated, kernel_size=ks, stride=1, padding=pad)
        return closed
    
    @staticmethod
    def _odd_or_zero(k: int) -> int:
        """0 bleibt 0 (deaktiviert). k>=1 wird auf ungerade korrigiert."""
        k = int(max(0, k))
        if k == 0:
            return 0
        return k if (k % 2 == 1) else (k + 1)

    @staticmethod
    def _box_blur(gray: torch.Tensor, k: int) -> torch.Tensor:
        """Einfacher Box-Blur auf [B,1,H,W]; k=0/1 → passthrough."""
        k = DS_EMSE._odd_or_zero(k)
        if k <= 1:
            return gray
        w = torch.ones((1, 1, k, k), dtype=gray.dtype, device=gray.device) / float(k * k)
        return F.conv2d(gray, w, padding=k // 2)

    def diff_edge_map(self, img, eps=1e-6):
        """
        Differenzierbare Kantenkarte via Sobel.
        Erwartet img in [-1,1], Shape [B,C,H,W], gibt [B,3,H,W] in [0,1] zurück (weiß=Hintergrund).
        """
        B, C, H, W = img.shape

        # Grauwert
        if C == 3:
            r = img[:, 0:1]; g = img[:, 1:2]; b = img[:, 2:3]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray = img

        # (Optional) Vorglätten vor Sobel — identisch zum Edge-Tuner (Box-Blur)
        pre_blur_k = int(getattr(self.config, "DS_PRE_BLUR_K", 0))  # 0=aus, sonst beliebig
        pre_blur_k = self._odd_or_zero(pre_blur_k)                  # gerade → ungerade +1
        if pre_blur_k > 1:
            gray = self._box_blur(gray, pre_blur_k)

        # Sobel
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + eps)

        # Normalisieren
        mag = mag / (mag.amax(dim=(2, 3), keepdim=True) + eps)

        # (Optional) Morphologisches Opening auf der HELLEN Magnitude, um feine Kanten zu killen
        open_ks = int(getattr(self.config, "DS_OPEN_KS", 0))  # 0=aus, sonst 3/5/7...
        if open_ks and open_ks >= 3 and open_ks % 2 == 1:
            mag = self._morph_open(mag, open_ks)

        # (Optional) Soft-Threshold (Sigmoid-Gating) für „nur starke“ Kanten
        # tau ~ Schwellwert in [0..1], beta ~ Steilheit (10..30 ist oft gut)
        tau = getattr(self.config, "DS_TAU", None)   # z.B. 0.3
        beta = float(getattr(self.config, "DS_BETA", 20.0))
        if tau is not None:
            mag = torch.sigmoid((mag - float(tau)) * beta)

        # (Optional) Closing, falls du nach dem Opening Lücken schließen willst
        close_ks = int(getattr(self.config, "DS_CLOSE_KS", 0))
        if close_ks and close_ks >= 3 and close_ks % 2 == 1:
            mag = self._morph_close(mag, close_ks)

        # invertieren: weißer Hintergrund (1), Kanten dunkel
        inv = 1.0 - mag

        # auf 3 Kanäle bringen
        edge3 = inv.repeat(1, 3, 1, 1)
        return edge3




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
    
    def rgb2Gray_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Erwartet x in [B, C, H, W]; gibt [B, 1, H, W] zurück.
        Kein Reshape über sqrt – unterstützt rechteckige Bilder.
        """
        # Falls bereits 1 Kanal: einfach zurückgeben
        if x.dim() == 4 and x.size(1) == 1:
            return x

        # Standard-RGB → Grauwert, Kanal-Dim beibehalten
        # (Slice mit :1 erhält die Kanalachse)
        r = x[:, 0:1]
        g = x[:, 1:1+1]
        b = x[:, 2:2+1]
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # [B,1,H,W]
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
        self.debug_prints = False
        self.config = config

        try:
            device=self.config.get("device", "cpu")
        except:
            device=getattr(self.config, "device", "cpu")
        print(device)

        # Ziel-Gitter (fix, normalisierte Koords [-1,1])
        x = torch.linspace(-1.0, 1.0, steps=5)
        y = torch.linspace(-1.0, 1.0, steps=5)
        self.target_control_points = torch.tensor(
            list(itertools.product(x, y)),
            device=device
        )
        
        # Stärke & Verteilung aus der Config
        self.lam = getattr(self.config, "TSD_LAM", getattr(self.config, "tsd_lam", 0.1))
        self.dist = getattr(self.config, "TSD_DIST", getattr(self.config, "tsd_dist", "uniform"))

        self.tps = None  # lazily init mit erstem Call

    def _sample_noise(self):
        """Rauschen für die Kontrollpunkte gemäß lam & dist."""
        if self.dist.lower() == "normal":
            return torch.randn_like(self.target_control_points) * self.lam
        else:
            # default: gleichverteilt
            return torch.empty_like(self.target_control_points).uniform_(-self.lam, self.lam)

    def _ensure_tps(self, height, width):
        if self.tps is None:
            self.tps = TPSGridGen(self.config, height, width, self.target_control_points)

    def _build_grid(self, batch_size, height, width):
        self._ensure_tps(height, width)
        grids = []
        for _ in range(batch_size):
            src_cps = self.target_control_points + self._sample_noise()
            src_coord = self.tps(src_cps.unsqueeze(0))          # aktuell: [1, H*W, 2]
            src_coord = src_coord.view(1, height, width, 2)     # FIX: [1, H, W, 2]
            grids.append(src_coord)
        return torch.cat(grids, dim=0)                          # [B, H, W, 2]

    def apply_grid(self, imgs, grid, canvas_val=1.0):
        """Wendet ein gegebenes Grid auf imgs an (gleiche H/W)."""
        B, C, H, W = imgs.shape
        assert grid.shape == (B, H, W, 2), "Grid shape must match batch and spatial size"
        out = F.grid_sample(imgs, grid, align_corners=True)
        # optionales Padding (wie vorher), falls Samples rausfallen
        inp_mask = imgs.new_ones(imgs.size())
        out_mask = F.grid_sample(inp_mask, grid, align_corners=True)
        canvas = imgs.new_full(imgs.size(), fill_value=canvas_val)
        return out * out_mask + canvas * (1 - out_mask)

    def doTSD(self, img, grid=None, return_grid=False, lam=None):
        """
        Verformt img.
        - Wenn grid=None: zieht ein neues Random-Grid (mit lam falls angegeben).
        - Wenn grid!=None: verwendet exakt dieses Grid (Identische Deformation).
        - return_grid=True: gibt (verformtes_img, grid) zurück.
        """
        B, C, H, W = img.shape

        # ggf. lam temporär überschreiben
        if lam is not None:
            old_lam = self.lam
            self.lam = lam
        else:
            old_lam = None

        if grid is None:
            grid = self._build_grid(B, H, W)

        out = self.apply_grid(img, grid)

        if old_lam is not None:
            self.lam = old_lam

        return (out, grid) if return_grid else out
    

class TSG:
    """
    Texture and shape-based generation (TSG)
    """

    def __init__(self, config):
        self.config = config
        self.debug_prints = False
        self.bce_logits = BCEWithLogitsLoss()   # stabiler GAN-Loss auf Logits

    def myNormalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def generateImg(
        self,
        mn_batch,
        netG,
        deformedImg,
        blurImg
    ):
        zdim = getattr(self.config, "noise_size", 100)
        z_ = Variable(torch.randn((mn_batch, zdim)).view(-1, 100, 1, 1).to(self.config.DEVICE))

        G_result = netG(z_, deformedImg, blurImg)

        # G_result = (G_result + 1) / 2   # scaling from [-1, 1] to [0, 1] due to tanh as last layer in netG

        return G_result
    
    def getDResult(
        self,
        img,
        netD
    ):
        D_result, aux_output = netD(img)
        # D_result = D_result.squeeze()

        # D_result_1 = D_result.mean()

        return D_result, aux_output
    

    def doTSG_stage1_training(
        self,
        iteration,
        config,
        ds_emse,
        emse,
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
        autocast_ctx = amp.autocast(device_type="cuda") if config.DEVICE.type != "cpu" else nullcontext()
        loss = SimpleNamespace() # Initialize loss variables
        mn_batch = img.shape[0] # get the batch size


        # === Stage 1: Step 1: Precompute blurred texture map Itxt ===
        #img_blur = blur_image(img, config.DOWN_SIZE)  # corresponds to Itxt in the paper
        #replaced with new dynamic blur:

        if getattr(config, "tex_dropout_s1", False):
            img_blur = blur_image_dynamic(
                img,
                down_min=getattr(config, "texD_down_min", 64),
                down_max=getattr(config, "texD_down_max", 128),
                p_heavy=getattr(config, "texD_p_heavy", 0.35),
                heavy_down=getattr(config, "texD_heavy_down", 12),
                p_zero=getattr(config, "texD_p_zero", 0.15),
                noise_std=getattr(config, "texD_noise_std", 0.0),
            )
        else:
            img_blur = blur_image(img, config.DOWN_SIZE)



        # === Stage 1: Step 2: Discriminator training with rough image ===
        time_startTrainD = time.time()

        with autocast_ctx:
            # Generate rough image
            #print(f"Got input: {mn_batch}, {e_extend.shape} and {img_blur.shape}")
            with torch.no_grad():
                G_rough_1 = self.generateImg(mn_batch, netG, e_extend, img_blur)  # Input: edge + blurred image

            # Discriminator output on real
            D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
            # D_result_realImg = D_result_realImg.mean()
            valid = torch.ones_like(D_result_realImg)

            # Discriminator output on fake
            D_result_roughImg_1, _ = self.getDResult(G_rough_1, netD)
            # D_result_roughImg = D_result_roughImg.mean()
            fake_1 = torch.zeros_like(D_result_roughImg_1)

            loss.D_loss = self.bce_logits(D_result_realImg, valid) + self.bce_logits(D_result_roughImg_1, fake_1)

            if aux_output_realImg is not None:
                D_celoss = CE_loss(aux_output_realImg, label)
                if self.debug_prints:
                    print(f"[TSG] D_celoss S1 training: {D_celoss}")
                loss.D_loss = loss.D_loss + 0.5 * D_celoss

        # === Train Discriminator ===
        try:
            netD.zero_grad(); optimD.zero_grad()
            scaler.scale(loss.D_loss).backward()
            scaler.unscale_(optimD)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), 5.0)
            scaler.step(optimD); scaler.update()
        except:
            print("[WARN] D_loss not finite → skip D step")

        # get time needed for disciminator training
        time_TSG.time_trainD.append(time.time() - time_startTrainD)


        # === Stage 1: Step 3: Train Generator with rough image ===
        time_startTrainG1 = time.time()

        with autocast_ctx:
            # Generate rough image
            G_rough_2 = self.generateImg(mn_batch, netG, e_extend, img_blur)

            # Discriminator output on fake
            D_result_roughImg_2, aux_output_roughImg_2 = self.getDResult(G_rough_2, netD)
            fake_2 = torch.ones_like(D_result_roughImg_2)

            # stabiler GAN-Loss
            D_result_roughImg_2 = self.bce_logits(D_result_roughImg_2, fake_2)

            # === Edge preservation loss (Ledge) ===
            e_extend_G_rough = ds_emse.diff_edge_map(G_rough_2)                          # [B,1,H,W]
            # e_extend_G_rough = emse.doEMSE(G_rough.detach())
            edge_loss = L1_loss(e_extend_G_rough, e_extend)

            # e_extend_G_rough = emse.doEMSE(G_rough.detach())
            # edge_loss = L1_loss(e_extend_G_rough, e_extend)
            
            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            G_L1_loss_rough = L1_loss(G_rough_2, img_for_loss)
            
            if self.debug_prints:
                print(f"[TSG] aux_output_roughImg training: {aux_output_roughImg_2}")
            if aux_output_roughImg_2 is not None:
                G_celoss_rough = CE_loss(aux_output_roughImg_2, label)
            else:
                G_celoss_rough = torch.tensor(0.0).to(config.DEVICE)

            lambda_L1 = config.LAMBDA_S1_L1 # original: 1.0
            lamda_GAN = config.LAMBDA_S1_GAN # original: 1.0
            lambda_CE = config.LAMBDA_S1_CE # original: 0.5
            lambda_edge = config.LAMBDA_S1_EDGE
            loss.G_loss = (
                lambda_L1 * G_L1_loss_rough
                + lamda_GAN * D_result_roughImg_2
                + lambda_CE * G_celoss_rough
                + lambda_edge * edge_loss
                )

        try:
            netG.zero_grad(); optimG.zero_grad()
            scaler.scale(loss.G_loss).backward()
            scaler.unscale_(optimG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.0)
            scaler.step(optimG); scaler.update()
        except:
            print("[WARN] G_loss not finite → skip G step")
        
        # get time needed for generator training in stage 1
        time_TSG.time_trainG1.append(time.time() - time_startTrainG1)

        if config.SHOW_TSG and iteration % config.SHOW_IMAGES_INTERVAL == 0:
            #pdb.set_trace()
            show_tsg(
                img=img,
                e_extend=e_extend,
                img_blur=img_blur,
                img_for_loss=img_for_loss,
                G_rough=G_rough_2,
            )

        return netD, netG, optimD, optimG, loss, time_TSG
    

    def doTSG_stage2_training(
        self,
        iteration,
        config,
        ds_emse,
        emse,
        tsd,
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
        scaler,
        tsd_grid=None         # <— NEU
    ):
        '''
        Function for stage 2 training of GAN with fine image (extented and deformed edge map and blured image).
        '''
        
        # === Stage 2: Step 0: Preparations ===
        #pdb.set_trace()
        autocast_ctx = amp.autocast(device_type="cuda") if config.DEVICE.type != "cpu" else nullcontext()
        loss = SimpleNamespace() # Initialize loss variables
        mn_batch = img.shape[0] # get the batch size


        # === Stage 2: Step 1: Precompute blurred texture map Itxt ===
        img_blur = blur_image(img, config.DOWN_SIZE2)  # corresponds to Itxt in the paper


        # === Stage 2: Step 2: Discriminator training with fine image ===
        time_startTrainD = time.time()

        with autocast_ctx:
            # Generate fine image from deformed edge map
            with torch.no_grad():
                G_fine_1 = self.generateImg(mn_batch, netG, e_deformed, img_blur)  # Input: edge deformed + blurred image

            # Discriminator output on real and fake
            D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
            # D_result_realImg = D_result_realImg.mean()
            valid = torch.ones_like(D_result_realImg)

            D_result_fineImg_1, _ = self.getDResult(G_fine_1, netD)
            # D_result_fineImg = D_result_fineImg.mean()
            fake_1 = torch.zeros_like(D_result_fineImg_1)

            loss.D_loss = self.bce_logits(D_result_realImg, valid) + self.bce_logits(D_result_fineImg_1, fake_1)

            if aux_output_realImg is not None:
                D_celoss = CE_loss(aux_output_realImg, label)
                if self.debug_prints:
                    print(f"[TSG] D_celoss S2 training: {D_celoss}")
                loss.D_loss = loss.D_loss + 0.5 * D_celoss

        # === Train Discriminator ===
        try:
            netD.zero_grad(); optimD.zero_grad()
            scaler.scale(loss.D_loss).backward()
            scaler.unscale_(optimD)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), 5.0)
            scaler.step(optimD); scaler.update()
        except:
            print("[WARN] D_loss not finite → skip D step")

        # get time needed for disciminator training
        time_TSG.time_trainD.append(time.time() - time_startTrainD)


        # === Stage 2: Step 3: Generator training with fine image ===
        time_startTrainG2 = time.time()

        with autocast_ctx:
            # Generate fine image from deformed edge map
            G_fine_2 = self.generateImg(mn_batch, netG, e_deformed, img_blur)

            # === Classifier Loss ===
            time_startTrainCls = time.time()
            if config.TRAIN_WITH_CLS:
                G_fine_resized = F.interpolate(G_fine_2, size=(299, 299), mode='bilinear', align_corners=False)
                if G_fine_resized.shape[1] == 1:
                    G_fine_resized = G_fine_resized.repeat(1, 3, 1, 1)
                G_fine_norm = transform_cls(G_fine_resized)
                cls_output = cls(G_fine_norm)
                loss.cls_loss = CE_loss(cls_output, label)
            else:
                loss.cls_loss = torch.tensor(0.0).to(config.DEVICE)
                G_fine_resized = None
                G_fine_norm = None
            time_TSG.time_trainCls.append(time.time() - time_startTrainCls)

            # Discriminator output on fake
            D_result_fineImg_2, aux_output_fineImg_2 = self.getDResult(G_fine_2, netD)
            fake_2 = torch.ones_like(D_result_fineImg_2)

            # stabiler GAN-Loss
            D_result_fineImg_2 = self.bce_logits(D_result_fineImg_2, fake_2)

            # === Edge preservation loss (Ledge) ===
            e_extend_G_fine = ds_emse.diff_edge_map(G_fine_2)                          # [B,1,H,W]
            # e_extend_G_rough = emse.doEMSE(G_rough.detach())
            edge_loss = L1_loss(e_extend_G_fine, e_deformed)

            # e_extend_G_fine = emse.doEMSE(G_fine.detach())
            # edge_loss = L1_loss(e_extend_G_fine, e_deformed)

            # === Generator loss ===
            # call discriminator again due to inplace-error
            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            # NEU: nur deformieren, wenn wir gegen das deformierte GT vergleichen wollen
            if not getattr(config, "L1_COMPARE_TO_ORIG", False):
                if tsd_grid is not None:
                    img_for_loss = tsd.apply_grid(img_for_loss, tsd_grid)
                else:
                    img_for_loss = tsd.doTSD(img_for_loss)

            G_L1_loss_fine = L1_loss(G_fine_2, img_for_loss)
            
            if self.debug_prints:
                print(f"[TSG] aux_output_fineImg training: {aux_output_fineImg_2}")
            if aux_output_fineImg_2 is not None:
                G_celoss_fine = CE_loss(aux_output_fineImg_2, label)
            else:
                G_celoss_fine = torch.tensor(0.0).to(config.DEVICE)

            # Total loss = GAN + Classification + Shape-Preservation
            lambda_L1 = config.LAMBDA_S2_L1 # original: 1.0
            lamda_GAN = config.LAMBDA_S2_GAN # original: 1.0
            lambda_CE = config.LAMBDA_S2_CE # original: 0.5
            lambda_cls = config.LAMBDA_S2_CLS # original: 1.0
            lambda_edge = config.LAMBDA_S2_EDGE # original: 1.0
            loss.G_loss = (
                lambda_L1 * G_L1_loss_fine
                + lamda_GAN * D_result_fineImg_2
                + lambda_CE * G_celoss_fine
                + lambda_cls * loss.cls_loss
                + lambda_edge * edge_loss
            )

        try:
            netG.zero_grad(); optimG.zero_grad()
            scaler.scale(loss.G_loss).backward()
            scaler.unscale_(optimG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.0)
            scaler.step(optimG); scaler.update()
        except:
            print("[WARN] G_loss not finite → skip G step")
        
        time_TSG.time_trainG2.append(time.time() - time_startTrainG2)

        if config.SHOW_TSG and iteration % config.SHOW_IMAGES_INTERVAL == 0:
            show_tsg(
                img=img,
                e_deformed=e_deformed,
                img_blur=img_blur,
                img_for_loss=img_for_loss,
                G_fine=G_fine_2,
                G_fine_resized=G_fine_resized,
                G_fine_norm=G_fine_norm,
                e_extend_G_fine=e_extend_G_fine
            )
            #pdb.set_trace()

        return netD, netG, optimD, optimG, loss, time_TSG
    

    def doTSG_stage1_testing(
        self,
        config,
        ds_emse,
        emse,
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
            autocast_ctx = amp.autocast(device_type="cuda") if config.DEVICE.type != "cpu" else nullcontext()
            loss = SimpleNamespace()
            mn_batch = img.shape[0]


            # === Stage 1: Step 1: Precompute blurred texture map Itxt ===
            img_blur = blur_image(img, config.DOWN_SIZE)  # corresponds to Itxt in the paper


            # === Stage 1: Step 2: Discriminator validation ===
            with autocast_ctx:
                # Generate rough image from extended edge map and blurred image
                #print(f"Got input: {mn_batch}, {e_extend.shape} and {img_blur.shape}")
                with torch.no_grad():
                    G_rough_1 = self.generateImg(mn_batch, netG, e_extend, img_blur)  # Input: edge + blurred image
                # G_rough_1 = G_rough_1.contiguous()

                # Discriminator output on real
                D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
                # D_result_realImg = D_result_realImg.mean()
                valid = torch.ones_like(D_result_realImg)

                # Discriminator output on fake
                D_result_roughImg_1, _ = self.getDResult(G_rough_1, netD)
                # D_result_roughImg = D_result_roughImg.mean()
                fake_1 = torch.zeros_like(D_result_roughImg_1)

                loss.D_loss = self.bce_logits(D_result_realImg, valid) + self.bce_logits(D_result_roughImg_1, fake_1)

                if aux_output_realImg is not None:
                    D_celoss = CE_loss(aux_output_realImg, label)
                    if self.debug_prints:
                        print(f"[TSG] D_celoss S1 training: {D_celoss}")
                    loss.D_loss = loss.D_loss + 0.5 * D_celoss


            # === Stage 1: Step 3: Generator validation ===
            with autocast_ctx:
                # Generate rough image
                G_rough_2 = self.generateImg(mn_batch, netG, e_extend, img_blur)

                # Discriminator output on fake
                D_result_roughImg_2, aux_output_roughImg_2 = self.getDResult(G_rough_2, netD)
                fake_2 = torch.ones_like(D_result_roughImg_2)

                # stabiler GAN-Loss
                D_result_roughImg_2 = self.bce_logits(D_result_roughImg_2, fake_2)
                
                # === Edge preservation loss (Ledge) ===
                e_extend_G_rough = ds_emse.diff_edge_map(G_rough_2)                          # [B,1,H,W]
                # e_extend_G_rough = emse.doEMSE(G_rough.detach())
                edge_loss = L1_loss(e_extend_G_rough, e_extend)

                # e_extend_G_rough = emse.doEMSE(G_rough.detach())
                # edge_loss = L1_loss(e_extend_G_rough, e_extend)
                
                img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
                G_L1_loss_rough = L1_loss(G_rough_2, img_for_loss)

                if self.debug_prints:
                    print(f"[TSG] aux_output_roughImg validating: {aux_output_roughImg_2}")
                if aux_output_roughImg_2 is not None:
                    G_celoss_rough = CE_loss(aux_output_roughImg_2, label)
                else:
                    G_celoss_rough = torch.tensor(0.0).to(config.DEVICE)

                lambda_L1 = config.LAMBDA_S1_L1 # original: 1.0
                lamda_GAN = config.LAMBDA_S1_GAN # original: 1.0
                lambda_CE = config.LAMBDA_S1_CE # original: 0.5
                lambda_edge = config.LAMBDA_S1_EDGE
                loss.G_loss = (
                    lambda_L1 * G_L1_loss_rough
                    + lamda_GAN * D_result_roughImg_2
                    + lambda_CE * G_celoss_rough
                    + lambda_edge * edge_loss
                    )

            return loss
    

    def doTSG_stage2_testing(
        self,
        config,
        ds_emse,
        emse,
        tsd,
        img,
        label,
        e_deformed,
        netD,
        netG,
        cls,
        transform_cls,
        CE_loss,
        L1_loss,
        tsd_grid=None         # <— NEU
    ):
        '''
        Function for stage 2 validation of GAN with fine image (extented and deformed edge map and blured image).
        '''
        with torch.no_grad():
            netG.eval()
            netD.eval()
            if cls:
                cls.eval()

            # === Stage 2: Step 0: Preparations ===
            # pdb.set_trace()
            autocast_ctx = amp.autocast(device_type="cuda") if config.DEVICE.type != "cpu" else nullcontext()
            loss = SimpleNamespace()
            mn_batch = img.shape[0]


            # === Stage 2: Step 1: Precompute blurred texture map Itxt ===
            img_blur = blur_image(img, config.DOWN_SIZE2)  # corresponds to Itxt in the paper


            # === Stage 2: Step 2: Discriminator validation with fine image ===
            with autocast_ctx:
                # Generate fine image from deformed edge map
                with torch.no_grad():
                    G_fine_1 = self.generateImg(mn_batch, netG, e_deformed, img_blur)  # Input: edge deformed + blurred image
                # G_fine_1 = G_fine_1.contiguous()

                # Discriminator output on real and fake
                D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
                # D_result_realImg = D_result_realImg.mean()
                valid = torch.ones_like(D_result_realImg)

                D_result_fineImg_1, _ = self.getDResult(G_fine_1, netD)
                # D_result_fineImg = D_result_fineImg.mean()
                fake_1 = torch.zeros_like(D_result_fineImg_1)

                loss.D_loss = self.bce_logits(D_result_realImg, valid) + self.bce_logits(D_result_fineImg_1, fake_1)

                if aux_output_realImg is not None:
                    D_celoss = CE_loss(aux_output_realImg, label)
                    if self.debug_prints:
                        print(f"[TSG] D_celoss S2 training: {D_celoss}")
                    loss.D_loss = loss.D_loss + 0.5 * D_celoss


            # === Stage 2: Step 3: Generator validation with fine image ===
            with autocast_ctx:
                # Generate fine image from deformed edge map
                G_fine_2 = self.generateImg(mn_batch, netG, e_deformed, img_blur)

                # === Classifier Loss ===
                if config.TRAIN_WITH_CLS:
                    G_fine_resized = F.interpolate(G_fine_2, size=(299, 299), mode='bilinear', align_corners=False)
                    if G_fine_resized.shape[1] == 1:
                        G_fine_resized = G_fine_resized.repeat(1, 3, 1, 1)
                    G_fine_norm = transform_cls(G_fine_resized)
                    cls_output = cls(G_fine_norm)
                    loss.cls_loss = CE_loss(cls_output, label)
                    cls_prediction = torch.argmax(cls_output, dim=1)
                else:
                    loss.cls_loss = torch.tensor(0.0).to(config.DEVICE)
                    cls_prediction = None

                # Discriminator output on fake
                D_result_fineImg_2, aux_output_fineImg_2 = self.getDResult(G_fine_2, netD)
                fake_2 = torch.ones_like(D_result_fineImg_2)

                # stabiler GAN-Loss
                D_result_fineImg_2 = self.bce_logits(D_result_fineImg_2, fake_2)

                # === Edge preservation loss (Ledge) ===
                e_extend_G_fine = ds_emse.diff_edge_map(G_fine_2)                          # [B,1,H,W]
                # e_extend_G_rough = emse.doEMSE(G_rough.detach())
                edge_loss = L1_loss(e_extend_G_fine, e_deformed)

                # emse_G_fine = emse.doEMSE(G_fine.detach())
                # edge_loss = L1_loss(emse_G_fine, e_deformed)

                # === Generator loss ===
                img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
                # *** WICHTIG: dieselbe Deformation wie für e_deformed ***
                if not getattr(config, "L1_COMPARE_TO_ORIG", False):
                    if tsd_grid is not None:
                        img_for_loss = tsd.apply_grid(img_for_loss, tsd_grid)
                    else:
                        img_for_loss = tsd.doTSD(img_for_loss)

                G_L1_loss_fine = L1_loss(G_fine_2, img_for_loss)

                if self.debug_prints:
                    print(f"[TSG] aux_output_fineImg validating: {aux_output_fineImg_2}")
                if aux_output_fineImg_2 is not None:
                    G_celoss_fine = CE_loss(aux_output_fineImg_2, label)
                else:
                    G_celoss_fine = torch.tensor(0.0).to(config.DEVICE)

                # Total loss = GAN + Classification + Shape-Preservation
                lambda_L1 = config.LAMBDA_S2_L1 # original: 1.0
                lamda_GAN = config.LAMBDA_S2_GAN # original: 1.0
                lambda_CE = config.LAMBDA_S2_CE # original: 0.5
                lambda_cls = config.LAMBDA_S2_CLS # original: 1.0
                lambda_edge = config.LAMBDA_S2_EDGE # original: 1.0
                loss.G_loss = (
                    lambda_L1 * G_L1_loss_fine
                    + lamda_GAN * D_result_fineImg_2
                    + lambda_CE * G_celoss_fine
                    + lambda_cls * loss.cls_loss
                    + lambda_edge * edge_loss
                )

            return loss, cls_prediction
