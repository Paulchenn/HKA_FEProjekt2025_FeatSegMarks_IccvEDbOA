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
    
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='in', mode='bilinear'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False if mode=='bilinear' else None)
        self.aa = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)  # kleiner FIR-Blur
        with torch.no_grad():
            k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32)
            k = (k / k.sum()).view(1,1,3,3)
            self.aa.weight.copy_(k.repeat(in_ch,1,1,1))
        self.aa.weight.requires_grad_(False)  # fixierter Blur
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True) if norm=='in' else nn.GroupNorm(32, out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.up(x)
        x = self.aa(x)           # Anti-Aliasing
        x = self.pad(x)
        x = self.conv(x)
        return self.act(self.norm(x))



class generator(nn.Module):
    # initializers
    def __init__(self, d=128, img_size=256, debug=False):
        super(generator, self).__init__()
        self.myDebug=False
        # z->Featuremap (passt für 256x256: Kernel = img_size/4 = 64 -> 64x64)
        self.deconv1_1 = nn.ConvTranspose2d(100, 256, 16, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(256)


        # edgeMap-Encoder -------------------------------------------------------------------------
        self.conv1_1edge = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # channels: 3 --> 32    | pixels: 256 --> 256
        self.conv1_2edge = nn.Conv2d(32, 32, kernel_size=3, padding=1)   # channels: 3 --> 32    | pixels: 256 --> 256
        self.maxpool1edge = nn.MaxPool2d(kernel_size=2)                  # channels: 32 --> 32   | pixels: 256 --> 128
        
        self.conv2_1edge = nn.Conv2d(32, 64, 3, padding=1)               # channels: 64 --> 64   | pixels: 128 --> 128
        self.conv2_2edge = nn.Conv2d(64, 64, 3, padding=1)               # channels: 64 --> 64   | pixels: 128 --> 128
        self.maxpool2edge = nn.MaxPool2d(kernel_size=2)                  # channels: 64 --> 64   | pixels: 128 --> 64
        # self.blurpool1edge = BlurPool2x(32)

        self.conv3_1edge = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # channels: 64 --> 128  | pixels: 64 --> 64
        self.conv3_2edge = nn.Conv2d(128, 128, kernel_size=3, padding=1) # channels: 128 --> 128 | pixels: 64 --> 64
        self.conv3_3edge = nn.Conv2d(128, 128, kernel_size=3, padding=1) # channels: 128 --> 128 | pixels: 64 --> 64
        self.maxpool3edge = nn.MaxPool2d(kernel_size=2)                  # channels: 128 --> 128 | pixels: 64 --> 32
        # self.blurpool2edge = BlurPool2x(64)

        self.conv4_1edge = nn.Conv2d(128, 256, kernel_size=3, padding=1) # channels: 128 --> 256 | pixels: 32 --> 32
        self.conv4_2edge = nn.Conv2d(256, 256, kernel_size=3, padding=1) # channels: 256 --> 256 | pixels: 32 --> 32
        self.conv4_3edge = nn.Conv2d(256, 256, kernel_size=3, padding=1) # channels: 256 --> 256 | pixels: 32 --> 32
        self.maxpool4edge = nn.MaxPool2d(kernel_size=2)                  # channels: 256 --> 256 | pixels: 32 --> 16
        # self.blurpool2edge = BlurPool2x(64)
        # ------------------------------------------------------------------------------------------


        # blurImg-Encoder -------------------------------------------------------------------------
        self.conv1_1blur = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # channels: 3 --> 32    | pixels: 256 --> 256
        self.conv1_2blur = nn.Conv2d(32, 32, kernel_size=3, padding=1)   # channels: 3 --> 32    | pixels: 256 --> 256
        self.maxpool1blur = nn.MaxPool2d(kernel_size=2)                  # channels: 32 --> 32   | pixels: 256 --> 128
        
        self.conv2_1blur = nn.Conv2d(32, 64, 3, padding=1)               # channels: 64 --> 64   | pixels: 128 --> 128
        self.conv2_2blur = nn.Conv2d(64, 64, 3, padding=1)               # channels: 64 --> 64   | pixels: 128 --> 128
        self.maxpool2blur = nn.MaxPool2d(kernel_size=2)                  # channels: 64 --> 64   | pixels: 128 --> 64
        # self.blurpool1blur = BlurPool2x(32)

        self.conv3_1blur = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # channels: 64 --> 128  | pixels: 64 --> 64
        self.conv3_2blur = nn.Conv2d(128, 128, kernel_size=3, padding=1) # channels: 128 --> 128 | pixels: 64 --> 64
        self.conv3_3blur = nn.Conv2d(128, 128, kernel_size=3, padding=1) # channels: 128 --> 128 | pixels: 64 --> 64
        self.maxpool3blur = nn.MaxPool2d(kernel_size=2)                  # channels: 128 --> 128 | pixels: 64 --> 32
        # self.blurpool2blur = BlurPool2x(64)

        self.conv4_1blur = nn.Conv2d(128, 256, kernel_size=3, padding=1) # channels: 128 --> 256 | pixels: 32 --> 32
        self.conv4_2blur = nn.Conv2d(256, 256, kernel_size=3, padding=1) # channels: 256 --> 256 | pixels: 32 --> 32
        self.conv4_3blur = nn.Conv2d(256, 256, kernel_size=3, padding=1) # channels: 256 --> 256 | pixels: 32 --> 32
        self.maxpool4blur = nn.MaxPool2d(kernel_size=2)                  # channels: 256 --> 256 | pixels: 32 --> 16
        # self.blurpool2blur = BlurPool2x(64)
        # ------------------------------------------------------------------------------------------


        # Upsampling -------------------------------------------------------------------------------
        self.up4 = UpBlock(768, 384)   # 16 -> 32
        self.up3 = UpBlock(384, 128)   # 32 -> 64
        self.up2 = UpBlock(128,  32)   # 64 -> 128
        self.up1 = UpBlock( 32,  32)   # 128 -> 256
        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 3, 3, padding=0),
            nn.Tanh()
        )
        # self.deconv4 = nn.ConvTranspose2d(768, 384, 4, 2, 1)  # channels: 768 --> 512 | pixels: 16 --> 32
        # self.deconv4_bn = nn.BatchNorm2d(384)
        # self.deconv3 = nn.ConvTranspose2d(384, 128, 4, 2, 1)    # channels: 512 --> 256 | pixels: 32 --> 64
        # self.deconv3_bn = nn.BatchNorm2d(128)
        # self.deconv2 = nn.ConvTranspose2d(128, 32, 4, 2, 1)    # channels: 256 --> 128 | pixels: 64 --> 128
        # self.deconv2_bn = nn.BatchNorm2d(32)
        # self.deconv1 = nn.ConvTranspose2d(32, 3, 4, 2, 1)     # channels: 768 --> 512 | pixels: 128 --> 256
        # self.deconv1_bn = nn.BatchNorm2d(3)


        # NEU (minimal): zwei kleine Refine-Blöcke auf 64x64 nach dem Concatenate
        self.refine1 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )


        # Self-Attention auf 128x128 (unverändert)
        self.mha     = MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)


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
        if self.myDebug:
            print("Input shapes:")
            print(f"noiseVector: {noiseVector.shape}, edgeMap: {edgeMap.shape}, blurImg: {blurImg.shape}")

        noiseVector = F.relu(self.deconv1_1_bn(self.deconv1_1(noiseVector)))
        if self.myDebug:
            print(f"After deconv1_1: {noiseVector.shape}")

        #########################################################
        # edgeMap-encoder
        edgeMap = F.relu(self.conv1_1edge(edgeMap))
        edgeMap = F.relu(self.conv1_2edge(edgeMap))
        edgeMap = self.maxpool1edge(edgeMap)
        if self.myDebug:
            print(f"After edgeMap block1: {edgeMap.shape}")
        edgeMap = F.relu(self.conv2_1edge(edgeMap))
        edgeMap = F.relu(self.conv2_2edge(edgeMap))
        edgeMap = self.maxpool2edge(edgeMap)
        if self.myDebug:
            print(f"After edgeMap block2: {edgeMap.shape}")
        edgeMap = F.relu(self.conv3_1edge(edgeMap))
        edgeMap = F.relu(self.conv3_2edge(edgeMap))
        edgeMap = F.relu(self.conv3_3edge(edgeMap))
        edgeMap = self.maxpool3edge(edgeMap)
        if self.myDebug:
            print(f"After edgeMap block3: {edgeMap.shape}")
        edgeMap = F.relu(self.conv4_1edge(edgeMap))
        edgeMap = F.relu(self.conv4_2edge(edgeMap))
        edgeMap = F.relu(self.conv4_3edge(edgeMap))
        edgeMap = self.maxpool4edge(edgeMap)
        if self.myDebug:
            print(f"After edgeMap block4: {edgeMap.shape}")


        # blurImg-encoder
        blurImg = F.relu(self.conv1_1blur(blurImg))
        blurImg = F.relu(self.conv1_2blur(blurImg))
        blurImg = self.maxpool1blur(blurImg)
        if self.myDebug:
            print(f"After blurImg block1: {blurImg.shape}")
        blurImg = F.relu(self.conv2_1blur(blurImg))
        blurImg = F.relu(self.conv2_2blur(blurImg))
        blurImg = self.maxpool2blur(blurImg)
        if self.myDebug:
            print(f"After blurImg block2: {blurImg.shape}")
        blurImg = F.relu(self.conv3_1blur(blurImg))
        blurImg = F.relu(self.conv3_2blur(blurImg))
        blurImg = F.relu(self.conv3_3blur(blurImg))
        blurImg = self.maxpool3blur(blurImg)
        if self.myDebug:
            print(f"After blurImg block3: {blurImg.shape}")
        blurImg = F.relu(self.conv4_1blur(blurImg))
        blurImg = F.relu(self.conv4_2blur(blurImg))
        blurImg = F.relu(self.conv4_3blur(blurImg))
        blurImg = self.maxpool4blur(blurImg)
        if self.myDebug:
            print(f"After blurImg block4: {blurImg.shape}")
        #########################################################

        # Concatenate edgeMap and blurImg heads
        try:
            x = torch.cat([noiseVector, edgeMap, blurImg], 1)
            if self.myDebug:
                print(f"After concatenate: {x.shape}")
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            return
        
        
        x = self.refine1(x)
        if self.myDebug:
            print(f"After refine1: {x.shape}")
        # x = self.refine2(x)   # --> (maybe) not needed anymore because of new layers before concatentate

        x = self.up4(x)                     # 32x32
        if self.myDebug:
            print(f"After deconv4: {x.shape}")
        x = self.up3(x)                     # 64x64
        if self.myDebug:
            print(f"After deconv3: {x.shape}")
            
        B,C,H,W = x.shape
        x = x.view(B,C,-1).permute(0,2,1)             # [B, HW, C]
        x,_ = self.mha(x, x, x, need_weights=False)
        if self.myDebug:
            print(f"After mha: {x.shape}")
        x = x.permute(0,2,1).view(B,C,H,W)
        
        x = self.up2(x)                     # 128x128
        if self.myDebug:
            print(f"After deconv2: {x.shape}")
        x = self.up1(x)                     # 256x256
        x = self.out(x)
        if self.myDebug:
            print(f"Final: {x.shape}")

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
