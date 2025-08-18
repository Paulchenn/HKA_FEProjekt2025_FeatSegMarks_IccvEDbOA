import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

class BlurPool2x(nn.Module):
    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32)
        k = (k / k.sum()).view(1,1,3,3)
        self.register_buffer("k", k)
        self.channels = channels
    def forward(self, x):
        w = self.k.expand(self.channels, 1, 3, 3)
        return F.conv2d(x, w, stride=2, padding=1, groups=self.channels)


class generator(nn.Module):
    # initializers
    def __init__(self, d=128, img_size=256):
        super(generator, self).__init__()
        # z->Featuremap (passt für 256x256: Kernel = img_size/4 = 64 -> 64x64)
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, int(img_size/4), 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        # (unverändert; wird unten nicht genutzt – belassen für Minimaländerung)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)

        # NEU (minimal): zwei kleine Refine-Blöcke auf 64x64 nach dem Concatenate
        self.refine1 = nn.Sequential(
            nn.Conv2d(d*6, d*6, kernel_size=3, padding=1),
            nn.BatchNorm2d(d*6),
            nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(d*6, d*6, kernel_size=3, padding=1),
            nn.BatchNorm2d(d*6),
            nn.ReLU(inplace=True)
        )

        # Upsampling-Stufen wie gehabt
        self.deconv2 = nn.ConvTranspose2d(d*6, d*2, 4, 2, 1)  # 64 -> 128
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)    # (bleibt definiert, wird nicht benutzt)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d*2, 3, 4, 2, 1)    # 128 -> 256

        # edgeMap-Encoder (bis 64x64)--------------------------------------------------------------
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # 256×256 Block (zusätzlich):
        self.conv1_3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_4 = nn.Conv2d(32, 32, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        #self.blurpool1 = BlurPool2x(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 128×128 Block (zusätzlich):
        self.conv2_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        #self.blurpool2 = BlurPool2x(64)

        # 64×64 Head (ersetzen):
        # vorher: self.conv3_1 = nn.Conv2d(64, d*2, 3, padding=1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_pj = nn.Conv2d(128, d * 2, 1)
        #------------------------------------------------------------------------------------------

        # blurImg-Encoder (spiegelgleich)----------------------------------------------------------
        self.conv1_1col = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2col = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # 256×256 Block (zusätzlich):
        self.conv1_3col = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_4col = nn.Conv2d(32, 32, 3, padding=1)
        self.maxpool1col = nn.MaxPool2d(kernel_size=2)
        # self.blurpool1col = BlurPool2x(32)

        self.conv2_1col = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2col = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 128×128 Block (zusätzlich):
        self.conv2_3col = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_4col = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool2col = nn.MaxPool2d(kernel_size=2)
        # self.blurpool2col = BlurPool2x(64)

        # 64×64 Head (ersetzen):
        # vorher: self.conv3_1col = nn.Conv2d(64, d*2, kernel_size=3, stride=1, padding=1)
        self.conv3_1col = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3_pjcol = nn.Conv2d(128, d * 2, 1)
        # ------------------------------------------------------------------------------------------

        # Self-Attention auf 128x128 (unverändert)
        self.self_att = MultiheadAttention(embed_dim=d*2, num_heads=4, batch_first=True)

        #neu: zusätzliche Convs auf 128×128 (sichtbare Mehrschärfe/Details, weniger Checkerboard)
        self.refine128 = nn.Sequential(
            nn.Conv2d(d * 2, d * 2, 3, padding=1), nn.BatchNorm2d(d * 2), nn.ReLU(True),
            nn.Conv2d(d * 2, d * 2, 3, padding=1), nn.BatchNorm2d(d * 2), nn.ReLU(True),
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, noiseVector, edgeMap, blurImg):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(noiseVector)))

        #########################################################
        # edgeMap-encoder -> 64x64
        edgeMap = F.relu(self.conv1_1(edgeMap))
        edgeMap = F.relu(self.conv1_2(edgeMap))
        edgeMap = F.relu(self.conv1_3(edgeMap))  # neu
        edgeMap = F.relu(self.conv1_4(edgeMap))  # neu
        edgeMap = self.maxpool1(edgeMap)  # 256 -> 128
        #edgeMap = self.blurpool1(edgeMap)

        edgeMap = F.relu(self.conv2_1(edgeMap))
        edgeMap = F.relu(self.conv2_2(edgeMap))
        edgeMap = F.relu(self.conv2_3(edgeMap))  # neu
        edgeMap = F.relu(self.conv2_4(edgeMap))  # neu
        edgeMap = self.maxpool2(edgeMap)  # 128 -> 64
        #edgeMap = self.blurpool2(edgeMap)

        edgeMap = F.relu(self.conv3_1(edgeMap))  # 64 -> 128
        y = F.relu(self.conv3_pj(edgeMap))  # 128 -> 256

        # blurImg-encoder -> 64x64
        blurImg = F.relu(self.conv1_1col(blurImg))
        blurImg = F.relu(self.conv1_2col(blurImg))
        blurImg = F.relu(self.conv1_3col(blurImg))  # neu
        blurImg = F.relu(self.conv1_4col(blurImg))  # neu
        blurImg = self.maxpool1col(blurImg)
        #blurImg = self.blurpool1col(blurImg)

        blurImg = F.relu(self.conv2_1col(blurImg))
        blurImg = F.relu(self.conv2_2col(blurImg))
        blurImg = F.relu(self.conv2_3col(blurImg))  # neu
        blurImg = F.relu(self.conv2_4col(blurImg))  # neu
        blurImg = self.maxpool2col(blurImg)
        #blurImg = self.blurpool2col(blurImg)

        blurImg = F.relu(self.conv3_1col(blurImg))  # 64 -> 128
        yc = F.relu(self.conv3_pjcol(blurImg))  # 128 -> 256

        #########################################################
        try:
            x = torch.cat([x, y, yc], 1)  # [B, d*6, 64, 64]
        except RuntimeError as e:
            print(f"RuntimeError: {e}")

        # *** MINIMALE MEHR-TIEFE auf 64x64 (ohne Shape-Änderung) ***
        x = self.refine1(x)
        # x = self.refine2(x)   # --> (maybe) not needed anymore because of new layers before concatentate

        # 64 -> 128
        x = F.relu(self.deconv2_bn(self.deconv2(x)))   # [B, d*2, 128, 128]

        #neu
        #x = self.refine128(x)  # mehr Tiefe auf 128

        # Self-Attention auf 128x128
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)      # [B, H*W, C]
        x_attended, _ = self.self_att(x_flat, x_flat, x_flat)
        x = x_attended.permute(0, 2, 1).view(B, C, H, W)

        # 128 -> 256
        x = torch.tanh(self.deconv4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, input_size=256):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 16, bn=False),  # 256 -> 128
            *discriminator_block(16, 32),          # 128 -> 64
            *discriminator_block(32, 64),          # 64  -> 32
            *discriminator_block(64, 128),         # 32  -> 16
        )

        # Dynamically compute the size after conv_blocks
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            out = self.conv_blocks(dummy)
            self.flat_features = out.view(1, -1).shape[1]

        self.adv_layer = nn.Sequential(nn.Linear(self.flat_features, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(
            nn.Linear(self.flat_features, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
