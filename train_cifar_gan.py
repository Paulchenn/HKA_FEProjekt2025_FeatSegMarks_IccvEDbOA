import numpy as np
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from models import resnet, generation
from utils.canny import canny
from skimage.color import rgb2gray
from tps_grid_gen import TPSGridGen
from torch.autograd import Variable
import itertools


def get_edge(images, sigma=1.0, high_threshold=0.3, low_threshold=0.2):
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
                     low_threshold=low_threshold).astype(np.float)
        # edge = Image.fromarray((edge * 255.).astype(np.int8), mode='L')
        edge = (edge - 0.5) / 0.5
        edges.append([edge])
    edges = np.array(edges).astype('float32')
    edges = torch.from_numpy(edges).cuda()
    return edges


def rgb2Gray_batch(input):
    R = input[:, 0]
    G = input[:, 1]
    B = input[:, 2]
    input[:, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    input = input[:, 0]

    input = input.view(input.shape[0], 1, 32, 32)
    return input


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid).cuda()
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1).cuda())
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


def TPS_Batch(imgs):
    height, width = imgs.shape[3], imgs.shape[2]
    tps_img = []
    for i in range(imgs.shape[0]):
        img = imgs[i, :, :, :]
        img = img.unsqueeze(0)
        target_control_points = torch.Tensor(list(itertools.product(
            torch.arange(-1.0, 1.00001, 2.0 / 4),
            torch.arange(-1.0, 1.00001, 2.0 / 4),
        ))).cuda()
        source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1,
                                                                                                            0.1).cuda()
        # source_control_points = target_control_points + 0.01*torch.ones(target_control_points.size()).cuda()
        tps = TPSGridGen(height, width, target_control_points)
        source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))

        grid = source_coordinate.view(1, height, width, 2)
        canvas = Variable(torch.Tensor(1, 3, height, width).fill_(1.0)).cuda()
        target_image = grid_sample(img, grid, canvas)
        tps_img.append(target_image)
    tps_img = torch.cat(tps_img, dim=0)
    return tps_img


def get_info(input, batch_size):
    gray = rgb2Gray_batch(input)
    # gray = input
    mat1 = torch.cat([gray[:, :, 0, :].unsqueeze(2), gray], 2)[:, :, :gray.shape[2], :]
    mat2 = torch.cat([gray, gray[:, :, gray.shape[2] - 1, :].unsqueeze(2)], 2)[:, :, 1:, :]
    mat3 = torch.cat([gray[:, :, :, 0].unsqueeze(3), gray], 3)[:, :, :, :gray.shape[3]]
    mat4 = torch.cat([gray, gray[:, :, :, gray.shape[3] - 1].unsqueeze(3)], 3)[:, :, :, 1:]
    info_rec = (gray - mat1) ** 2 + (gray - mat2) ** 2 + (gray - mat3) ** 2 + (gray - mat4) ** 2
    info_rec_ave = info_rec.view(batch_size, -1)
    ave = torch.mean(info_rec_ave, dim=1)
    # info = torch.zeros(gray.shape, dtype=torch.float32)
    tmp = torch.zeros(gray.shape).cuda()
    for b in range(input.shape[0]):
        tmp[b] = ave[b]
    info = torch.where(info_rec > tmp, 1.0, 0.0)

    return info


def show_result(num_epoch, show=False, save=False, path='result.png'):
    zz = torch.randn(64, 100, 1, 1).cuda()
    netG.eval()
    test_images = netG(zz, show_x, bx)
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


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
cifar10_train = datasets.CIFAR10(root='./src/cifar10', train=True, download=True, transform=transform_train)
cifar10_test = datasets.CIFAR10(root='./src/cifar10', train=False, download=True, transform=transform_test)

batch_size = 128
lr = 1e-4
epochs = 60
device = torch.device('cuda')

re12 = transforms.Resize((12, 12))
re32 = transforms.Resize((32, 32))

netG = generation.generator(128)
netD = generation.Discriminator()
netG.cuda()
netD.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0., 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0., 0.99))

train_loader = DataLoader(dataset=cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=cifar10_test, batch_size=64)

L1_loss = nn.L1Loss()
MSE_loss = nn.MSELoss()
CE_loss = nn.CrossEntropyLoss()

for i, (img, label) in enumerate(test_loader):
    img = img.cuda()
    label = label.cuda()

    netD.zero_grad()

    mn_batch = img.shape[0]

    generated1 = get_edge(img, sigma=1.0, high_threshold=0.3, low_threshold=0.2)
    generated2 = get_info(img)
    generated1 = torch.where(generated1 < 0, 0., 1.)
    generated2 *= -1
    generated2 = torch.where(generated2 < 0, 0., 1.)
    combined = generated2 + generated1
    combined = torch.cat([combined, combined, combined], 1)

    blur_img = re12(img)
    blur_img = re32(blur_img)

    if i > 0:
        break
show_x = combined
bx = blur_img

for epoch in range(epochs):
    print(epoch)
    netG.train()
    netD.train()

    for i, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()

        netD.zero_grad()

        mn_batch = img.shape[0]

        generated1 = get_edge(img, sigma=1.0, high_threshold=0.3, low_threshold=0.2)
        generated2 = get_info(img)

        generated1 = torch.where(generated1 < 0, 0., 1.)
        generated2 *= -1
        generated2 = torch.where(generated2 < 0, 0., 1.)
        combined = generated2 + generated1
        combined = torch.cat([combined, combined, combined], 1).detach().cuda()

        blur_img = re12(img)
        blur_img = re32(blur_img)

        D_result, aux_output = netD(img)
        D_result = D_result.squeeze()

        D_result_1 = -D_result.mean()
        # D_result_1.backward()

        z_ = Variable(torch.randn((mn_batch, 100)).view(-1, 100, 1, 1).cuda())

        G_result = netG(z_, combined, blur_img)

        D_result, _ = netD(G_result)
        D_result = D_result.squeeze()
        D_result_2 = D_result.mean()

        D_celoss = CE_loss(aux_output, label)

        D_loss = D_result_1 + D_result_2 + 0.5 * D_celoss
        D_loss.backward()
        optimizerD.step()

        netG.zero_grad()
        z_ = Variable(torch.randn((mn_batch, 100)).view(-1, 100, 1, 1).cuda())

        G_result = netG(z_, combined, blur_img)
        D_result, aux_output = netD(G_result)
        D_result = D_result.squeeze()
        G_L1_loss = L1_loss(G_result, img)

        D_result = D_result.mean()

        G_celoss = CE_loss(aux_output, label)
        G_celoss = G_celoss.sum()

        total_loss = G_L1_loss - D_result + 0.5 * G_celoss
        total_loss.backward()
        optimizerG.step()

        if i % 200 == 0:
            print('Epoch[', epoch + 1, '/', 40, '][', i, '/', len(train_loader), ']: TOTAL_LOSS', total_loss.item())

    fixed_p = './Result/cifar_gan/visualization' + str(epoch) + '.png'
    show_result(epoch, path=fixed_p)
    torch.save(netG.state_dict(), './Result/cifar_gan/G_'+str(epoch)+'.pth')
    torch.save(netD.state_dict(), './Result/cifar_gan/D_'+str(epoch)+'.pth')
