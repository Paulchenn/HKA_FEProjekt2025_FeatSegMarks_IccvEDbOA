import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



# ===========================================================
# ========================= Helpers =========================
# ===========================================================
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, use_relu=True):
    '''
    Creates a convolutional layer followed by batch normalization and an optional ReLU activation.
    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        k (int): Kernel size for the convolution.
        s (int): Stride for the convolution.
        p (int): Padding for the convolution.
        use_relu (bool): Whether to apply ReLU activation.
    Returns:
        nn.Sequential: A sequential model containing the convolution, batch normalization, and ReLU (if applicable).
    '''

    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
    ]

    if use_relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_in_relu(in_ch, out_ch, k=3, s=1, p=1, use_relu=True, dropout=0.0):
    '''
    Creates a convolutional layer followed by instance normalization and an optional ReLU activation.
    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        k (int): Kernel size for the convolution.
        s (int): Stride for the convolution.
        p (int): Padding for the convolution.
        use_relu (bool): Whether to apply ReLU activation.
        dropout (float): Dropout rate to apply after the ReLU activation.
    Returns:
        nn.Sequential: A sequential model containing the convolution, batch normalization, and ReLU (if applicable).
    '''

    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.InstanceNorm2d(out_ch),
    ]

    if use_relu:
        layers.append(nn.ReLU(inplace=True))

    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)
# ===========================================================



# ===========================================================
# ========================= Encoder =========================
# ===========================================================
class VGG16Encoder(nn.Module):
    '''
    VGG16 Encoder with 2-2-3-3-3 Conv blocks and MaxPool (return_indices=True).
    Provides sizes and indices for the decoder.
    Args:
        in_channels (int): Number of input channels (default is 3 for RGB images).
    Returns:
        dict: A dictionary containing the output feature maps (p1 to p5), their indices (idx1 to idx5), and their sizes (size1 to size5).
    '''
    def __init__(self, in_channels=3):
        super().__init__()
        # Block 1: 2 conv -> pool
        self.enc1 = nn.Sequential(conv_in_relu(in_channels, 64),
                                  conv_in_relu(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 2: 2 conv -> pool
        self.enc2 = nn.Sequential(conv_in_relu(64, 128),
                                  conv_in_relu(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 3: 3 conv -> pool
        self.enc3 = nn.Sequential(conv_in_relu(128, 256),
                                  conv_in_relu(256, 256),
                                  conv_in_relu(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 4: 3 conv -> pool
        self.enc4 = nn.Sequential(conv_in_relu(256, 512),
                                  conv_in_relu(512, 512),
                                  conv_in_relu(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 5: 3 conv -> pool
        self.enc5 = nn.Sequential(conv_in_relu(512, 512),
                                  conv_in_relu(512, 512),
                                  conv_in_relu(512, 512))
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)


    def forward(self, x):
        '''
        forward function of VGG16 Encoder with 2-2-3-3-3 Conv blocks and MaxPool (return_indices=True).
        Provides sizes and indices for the decoder.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is batch size, C is number of channels, H and W are height and width.
        Returns:
            dict: A dictionary containing the output feature maps (p1 to p5), their indices (idx1 to idx5), and their sizes (size1 to size5).
        '''

        skips = []
        # 1.1: 3x256x256 Img -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 64x256x256 FeatMap
        # 1.2: append 1. FeatMap to skips[0]
        # 1.3: 64x256x256 FeatMap -> Pooling -> 64x128x128 FeatMap
        x = self.enc1(x); skips.append(x); x, idx1 = self.pool1(x)

        # 2.1: 64x128x128 FeatMap -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 128x128x128 FeatMap
        # 2.2: append 2. FeatMap to skips[1]
        # 2.3: 128x128x128 FeatMap -> Pooling -> 128x64x64 FeatMap
        x = self.enc2(x); skips.append(x); x, idx2 = self.pool2(x)

        # 3.1: 128x64x64 FeatMap -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 256x64x64 FeatMap
        # 3.2: append 3. FeatMap to skips[2]
        # 3.3: 256x64x64 FeatMap -> Pooling -> 256x32x32 FeatMap
        x = self.enc3(x); skips.append(x); x, idx3 = self.pool3(x)

        # 4.1: 256x32x32 FeatMap -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 512x32x32 FeatMap
        # 4.2: append 4. FeatMap to skips[3]
        # 4.3: 512x32x32 FeatMap -> Pooling -> 512x16x16 FeatMap
        x = self.enc4(x); skips.append(x); x, idx4 = self.pool4(x)

        # 5.1: 512x16x16 FeatMap -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 512x16x16 FeatMap
        # 5.2: append 5. FeatMap to skips[4]
        # 5.3: 512x16x16 FeatMap -> Pooling -> 512x8x8 latent Space
        x = self.enc5(x); skips.append(x); x, idx5 = self.pool5(x)

        return {
            "p5": x, "idx5": idx5, "size5": skips[4].size(),
            "skips": skips,
            "idx4": idx4, "size4": skips[3].size(),
            "idx3": idx3, "size3": skips[2].size(),
            "idx2": idx2, "size2": skips[1].size(),
            "idx1": idx1, "size1": skips[0].size(),
        }
# ===========================================================



# ===========================================================
# ========================= Decoder =========================
# ===========================================================
class VGG16Decoder(nn.Module):
    '''
    VGG16 Decoder with MaxUnPool and 3-3-3-2-2 Conv blocks.
    Takes sizes and indices from the encoder.
    Args:
        decoder_relu (bool): Whether to apply ReLU activation in the decoder.
        out_channels (int): Number of output channels (default is 3 for RGB images).
    Returns:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W) where B is batch size, out_channels is number of output channels, H and W are height and width.
    '''
    def __init__(self, decoder_relu=True, out_channels=3):
        super().__init__()
        self.relu = decoder_relu

        # Add fusion layers for each skip connection
        self.fuse_skip5 = nn.Conv2d(1024, 512, kernel_size=1)
        self.fuse_skip4 = nn.Conv2d(1024, 512, kernel_size=1)
        self.fuse_skip3 = nn.Conv2d(512, 256, kernel_size=1)
        self.fuse_skip2 = nn.Conv2d(256, 128, kernel_size=1)
        self.fuse_skip1 = nn.Conv2d(128, 64, kernel_size=1)

        # Block 5: unpool -> 3 conv
        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.dec5 = nn.Sequential(
            conv_in_relu(512, 512, use_relu=self.relu),
            conv_in_relu(512, 512, use_relu=self.relu),
            conv_in_relu(512, 512, use_relu=self.relu),
        )

        # Block 4: unpool -> 3 conv
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.dec4 = nn.Sequential(
            conv_in_relu(512, 512, use_relu=self.relu),
            conv_in_relu(512, 512, use_relu=self.relu),
            conv_in_relu(512, 256, use_relu=self.relu),
        )

        # Block 3: unpool -> 3 conv
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec3 = nn.Sequential(
            conv_in_relu(256, 256, use_relu=self.relu),
            conv_in_relu(256, 256, use_relu=self.relu),
            conv_in_relu(256, 128, use_relu=self.relu),
        )

        # Block 2: unpool -> 3 conv
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec2 = nn.Sequential(
            conv_in_relu(128, 128, use_relu=self.relu),
            conv_in_relu(128, 64, use_relu=self.relu),
        )

        # Block 1: unpool -> 3 conv
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec1 = nn.Sequential(
            conv_in_relu(64, 64, use_relu=self.relu),
            conv_in_relu(64, 64, use_relu=self.relu),
        )

        # Output layer: conv
        self.to_rgb = nn.Conv2d(64, out_channels, kernel_size=1, bias=True)


    def forward(self, x, edge_feats, blur_feats):
        '''
        forward function of VGG16 Decoder MaxUnPool and 3-3-3-2-2 Conv blocks.
        Takes sizes and indices from the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is batch size, C is number of channels, H and W are height and width.
            edge_feats (dict): Dictionary containing edge map features with keys "skips" for skip connections.
            blur_feats (dict): Dictionary containing blur image features with keys "skips" for skip
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W) where B is batch size, out_channels is number of output channels, H and W are height and width.
        '''

        edge_skips = edge_feats["skips"]
        blur_skips = blur_feats["skips"]

        # 5.1: concatenate edge and blur skips of step 5 and fuse them
        # 5.2: 512x8x8 latent Space -> Unpooling -> 512x16x16 FeatMap
        # 5.3: 512x16x16 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 512x16x16 FeatMap
        skip5 = torch.cat([edge_skips[4], blur_skips[4]], dim=1); skip5 = self.fuse_skip5(skip5)
        x = self.unpool5(x, edge_feats["idx5"], output_size=edge_feats["size5"])
        x = self.dec5(x + skip5)

        # 4.1: concatenate edge and blur skips of step 4 and fuse them
        # 4.2: 512x16x16 FeatMap -> Unpooling -> 512x32x32 FeatMap
        # 4.3: 512x32x32 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 256x32x32 FeatMap
        skip4 = torch.cat([edge_skips[3], blur_skips[3]], dim=1); skip4 = self.fuse_skip4(skip4)
        x = self.unpool4(x, edge_feats["idx4"], output_size=edge_feats["size4"])
        x = self.dec4(x + skip4)

        # 3.1: concatenate edge and blur skips of step 3 and fuse them
        # 3.2: 256x32x32 FeatMap -> Unpooling -> 256x64x64 FeatMap
        # 3.3: 256x64x64 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 128x64x64 FeatMap
        skip3 = torch.cat([edge_skips[2], blur_skips[2]], dim=1); skip3 = self.fuse_skip3(skip3)
        x = self.unpool3(x, edge_feats["idx3"], output_size=edge_feats["size3"])
        x = self.dec3(x + skip3)

        # 2.1: concatenate edge and blur skips of step 2 and fuse them
        # 2.2: 128x64x64 FeatMap -> Unpooling ->128x128x128 FeatMap
        # 2.3: 128x128x128 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 64x128x128 FeatMap
        skip2 = torch.cat([edge_skips[1], blur_skips[1]], dim=1); skip2 = self.fuse_skip2(skip2)
        x = self.unpool2(x, edge_feats["idx2"], output_size=edge_feats["size2"])
        x = self.dec2(x + skip2)

        # 1.1: concatenate edge and blur skips of step 1 and fuse them
        # 1.2: 64x128x128 FeatMap -> Unpooling -> 64x256x256 FeatMap
        # 1.3: 64x256x256 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 64x256x256 FeatMap
        skip1 = torch.cat([edge_skips[0], blur_skips[0]], dim=1); skip1 = self.fuse_skip1(skip1)
        x = self.unpool1(x, edge_feats["idx1"], output_size=edge_feats["size1"]); x = self.dec1(x + skip1)

        # 0.1: 64x256x256 FeatMap -> Conv -> 3x256x256 Img
        # 0.2: 3x256x256 Img -> tanh -> 3x256x256 Img
        x = torch.tanh(self.to_rgb(x))
        return x
# ===========================================================



# ===========================================================
# ======================= Generator =========================
# ===========================================================
class generator(nn.Module):
    '''
    Generator in SegNet topology.
    Combines two separate encoders (edgeMap and blurImg) with a noise vector.
    Args:
        d (dict): Optional dictionary for additional parameters (not used in this implementation).
        img_size (int): Size of the input image (default is 256).
        z_dim (int): Dimension of the noise vector (default is 100).
        decoder_relu (bool): Whether to apply ReLU activation in the decoder.
    Returns:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W) where B is batch size, out_channels is number of output channels, H and W are height and width
    '''
    def __init__(self, d=None, img_size=256, z_dim=100, decoder_relu=True):
        super().__init__()
        
        # Require image size to be divisible by 32 (5x pooling)
        assert img_size % 32 == 0, "img_size must be divisible by 32 (5x pooling)"
        self.img_size = img_size
        self.z_dim = z_dim

        # Edge Map Encoder: 5x Conv + Pooling (return_indices=True)
        self.edge_encoder = VGG16Encoder(in_channels=3)
        # Blur Image Encoder: 5x Conv + Pooling (return_indices=True)
        self.blur_encoder = VGG16Encoder(in_channels=3)

        # 100x1x1 Noise -> 512x8x8x512
        self.z_to_bottleneck = nn.ConvTranspose2d(z_dim, 512, kernel_size=img_size // 32,
                                                  stride=1, padding=0, bias=True)
        
        # Fusion layer: 1x1 conv after concatenation
        self.fusion_conv = nn.Conv2d(1024, 512, kernel_size=1, bias=True)

        # Decoder
        self.decoder = VGG16Decoder(decoder_relu=decoder_relu, out_channels=3)

        # Initialising weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, noiseVector, edgeMap, blurImg):

        # -----------------------------------------------------------
        # ------------------------- Encoder -------------------------
        # -----------------------------------------------------------
        # edgeMap encoder
        edge_feats = self.edge_encoder(edgeMap)
        # bluredImg encoder
        blur_feats = self.blur_encoder(blurImg)
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        # --------------------- Concatentaion -----------------------
        # -----------------------------------------------------------
        # concatenate edge and blur features at p5 level
        bottleneck_cat = torch.cat([edge_feats["p5"], blur_feats["p5"]], dim=1)  # [B,1024,H,W]
        # fuse the concatentated features witha 1x1 conv (1024 channels -> 512 channels)
        bottleneck = self.fusion_conv(bottleneck_cat)  # [B,512,H,W]
        # add Nosie by summing to bottleneck features
        z_feat = self.z_to_bottleneck(noiseVector)
        if z_feat.shape[-2:] != bottleneck.shape[-2:]:
            raise RuntimeError(f"Noise feature size {z_feat.shape[-2:]} != bottleneck {bottleneck.shape[-2:]}")
        bottleneck = bottleneck + z_feat
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        # ------------------------- Decoder -------------------------
        # -----------------------------------------------------------
        out = self.decoder(bottleneck, edge_feats, blur_feats)
        # -----------------------------------------------------------

        return out



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
