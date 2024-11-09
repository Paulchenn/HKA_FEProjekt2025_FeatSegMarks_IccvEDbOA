import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 8, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*6, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d*2, 3, 4, 2, 1)


        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

        self.conv1_1col = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2col = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1col = nn.MaxPool2d(kernel_size=2)
        self.conv2_1col = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2col = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool2col = nn.MaxPool2d(kernel_size=2)
        self.conv3_1col = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label, bc):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))

        #########################################################
        label = F.relu(self.conv1_1(label))
        label = F.relu(self.conv1_2(label))

        label = self.maxpool1(label)

        label = F.relu(self.conv2_1(label))
        label = F.relu(self.conv2_2(label))

        label = self.maxpool2(label)

        y = F.relu(self.conv3_1(label))

        bc = F.relu(self.conv1_1col(bc))
        bc = F.relu(self.conv1_2col(bc))
        bc = self.maxpool1col(bc)
        bc = F.relu(self.conv2_1col(bc))
        bc = F.relu(self.conv2_2col(bc))
        bc = self.maxpool2col(bc)

        yc = F.relu(self.conv3_1col(bc))

        #########################################################
        x = torch.cat([x, y, yc], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        # print(x.shape) [bs, 256, 16, 16]
        x = self.self_att(x)
        x = F.tanh(self.deconv4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 10), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()