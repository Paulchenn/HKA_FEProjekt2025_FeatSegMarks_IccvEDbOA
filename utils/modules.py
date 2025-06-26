import json
import numpy as np
import itertools
import pdb

from skimage.color import rgb2gray
from skimage.feature import canny
from utils.tps_grid_gen import TPSGridGen
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms

# --- Load config ---
with open('config/config.json', 'r') as f:
    config = json.load(f)
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")


def show_images(
    img,
    deformedImg,
    edgeMap
):
    """
    Show the original image, deformed image, and edge map.
    :param img: Original image tensor
    :param deformedImg: Deformed image tensor
    :param edgeMap: Edge map tensor
    """
    
    # === Show the original and deformed image ===
    plt.figure(figsize=(10, 5))  # Adjusted figure size
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Original Image')
    plt.imshow(img[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Deformed Image')
    deformedImg_clipped = deformedImg[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    deformedImg_clipped = np.clip(deformedImg_clipped, 0, 1)
    plt.imshow(deformedImg_clipped)

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('Edge Map')
    edgeMap_clipped = edgeMap[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    edgeMap_clipped = np.clip(edgeMap_clipped, 0, 1)
    plt.imshow(edgeMap_clipped)

    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds (adjust as needed)
    plt.close()


def show_result(
    num_epoch,
    edgeMap,
    deformedImg,
    show=False,
    save=False,
    path='result.png',
    *,
    device=device,
    netG
):
    zz = torch.randn(64, 100, 1, 1).to(device)
    netG.eval()
    test_images = netG(zz, edgeMap, deformedImg)
    netG.train()

    size_figure_grid = 8
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(64):
        i = k // 8
        j = k % 8
        ax[i, j].cla()
        ax[i, j].imshow(np.transpose(test_images[k].cpu().data.numpy(), (1, 2, 0)))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


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
        downSize=12
    ):
        mn_batch = img.shape[0]
        
        # blur image to minimize impact of texture
        img_blur = self.blur_image(img, downSize)

        # Discriminator result of real image without blur
        D_result_realImg, aux_output_realImg = self.getDResult(img, netD)
        D_result_realImg = -D_result_realImg

        # Generate image and get Discriminator result
        G_result_1 = self.generateImg(mn_batch, netG, tpsImg, img_blur)
        D_result_genImg, _ = self.getDResult(G_result_1, netD)

        # === Discriminator training ===
        # Zero the gradients of discriminator
        netD.zero_grad()
        # Compute the classification loss of discriminator
        D_celoss = CE_loss(aux_output_realImg, label)
        # Compute the whole loss of discriminator
        D_loss = D_result_realImg + D_result_genImg + 0.5 * D_celoss
        # Calculate gradient of discriminator
        D_loss.backward()
        # Update the discriminator by calling optimizer
        optimD.step()

        # === Generator training (part 1)===
        # Zero the gradients of generator
        netG.zero_grad()
        # Generate new image and get discriminator result
        G_result_2 = self.generateImg(mn_batch, netG, tpsImg, img_blur)
        D_result_genImg, aux_output_genImg = self.getDResult(G_result_2, netD)
        # If Image is grayscale (only 1 channel), repeat / duplicate the channel to 3
        img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
        G_L1_loss = L1_loss(G_result_2, img_for_loss)
        # Compute the classification loss of generator and sum them up
        G_celoss = CE_loss(aux_output_genImg, label).sum()

        # === Classifier training ===
        # Zero the gradients of classifier
        cls.zero_grad()
        # Compute the classification loss of classifier
        cls_output = cls(G_result_2)
        cls_loss = CE_loss(cls_output, label)
        ### ????? >>>>> Classifier optimation missing? <<<<< ????? #####

        # === Generator training (part 2) ===
        # Loss of edge information
        combined_gray = tpsImg[:, 0:1, : , :]
        edge_loss = L1_loss(
            emse.get_info(G_result_2),
            combined_gray
        )
        
        # Compute the whole loss of generator
        G_loss_tot = G_L1_loss - D_result_genImg + 0.5 * G_celoss + edge_loss + cls_loss
        # Calculate gradient of generator
        G_loss_tot.backward()
        # Update the generator by calling optimizer
        optimG.step()

        return netD, netG, cls, optimD, optimG, optimC, CE_loss, L1_loss, G_loss_tot
        
    def doTSG_testing(
        self,
        img,
        tpsImg,
        netG,
        cls
    ):
        '''
        Generate image and classify it
        :param img: input image
        :param tpsImg: deformed (after TSD) image
        :param netG: generator network
        :param cls: classifier network
        :return: prediction of classifier
        '''
        mn_batch = img.shape[0]
        
        # blur image
        img_blur = self.blur_image(img)

        # Generate image and get Discriminator result
        G_result = self.generateImg(
            mn_batch,
            netG,
            tpsImg,
            img_blur
        )
        
        # Classification of generated image
        cls_output = cls(G_result)
        prediction = torch.argmax(cls_output, 1)

        return prediction