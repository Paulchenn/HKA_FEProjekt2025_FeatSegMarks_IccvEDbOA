import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# change to V2.2: adapting to 608x800 (WIP)



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

#TODO: testing for dropout
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
        # 1.1: 3x608x800 Img -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 64x608x800 FeatMap
        # 1.2: append 1. FeatMap to skips[0]
        # 1.3: 64x608x800 FeatMap -> Pooling -> 64x304x400 FeatMap
        x = self.enc1(x); skips.append(x); x, idx1 = self.pool1(x)

        # 2.1: 64x304x400 FeatMap -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 128x304x400 FeatMap
        # 2.2: append 2. FeatMap to skips[1]
        # 2.3: 128x304x400 FeatMap -> Pooling -> 128x152x200 FeatMap
        x = self.enc2(x); skips.append(x); x, idx2 = self.pool2(x)

        # 3.1: 128x152x200 FeatMap -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 256x152x200 FeatMap
        # 3.2: append 3. FeatMap to skips[2]
        # 3.3: 256x152x200 FeatMap -> Pooling -> 256x76x100 FeatMap
        x = self.enc3(x); skips.append(x); x, idx3 = self.pool3(x)

        # 4.1: 256x76x100 FeatMap -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 512x76x100 FeatMap
        # 4.2: append 4. FeatMap to skips[3]
        # 4.3: 512x76x100 FeatMap -> Pooling -> 512x38x50 FeatMap
        x = self.enc4(x); skips.append(x); x, idx4 = self.pool4(x)

        # 5.1: 512x38x50 FeatMap -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> Conv -> BatchNorm -> ReLu -> 512x38x50 FeatMap
        # 5.2: append 5. FeatMap to skips[4]
        # 5.3: 512x38x50 FeatMap -> Pooling -> 512x19x25 latent Space
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

        # NEU: Vor-Projektion der Skips (macht die Cat kleiner)
        self.pre1_e = nn.Conv2d(64, 32, kernel_size=1)
        self.pre1_b = nn.Conv2d(64, 32, kernel_size=1)
        self.pre2_e = nn.Conv2d(128, 64, kernel_size=1)
        self.pre2_b = nn.Conv2d(128, 64, kernel_size=1)
        self.pre3_e = nn.Conv2d(256, 128, kernel_size=1)
        self.pre3_b = nn.Conv2d(256, 128, kernel_size=1)
        self.pre4_e = nn.Conv2d(512, 256, kernel_size=1)
        self.pre4_b = nn.Conv2d(512, 256, kernel_size=1)
        self.pre5_e = nn.Conv2d(512, 256, kernel_size=1)
        self.pre5_b = nn.Conv2d(512, 256, kernel_size=1)

        # Add fusion layers for each skip connection
        self.fuse_skip5 = nn.Conv2d(512, 512, kernel_size=1)   # (256+256) -> 512
        self.fuse_skip4 = nn.Conv2d(512, 512, kernel_size=1)   # (256+256) -> 512
        self.fuse_skip3 = nn.Conv2d(256, 256, kernel_size=1)   # (128+128) -> 256
        self.fuse_skip2 = nn.Conv2d(128, 128, kernel_size=1)   # ( 64+ 64) -> 128
        self.fuse_skip1 = nn.Conv2d( 64,  64, kernel_size=1)   # ( 32+ 32) ->  64

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
        debug_prints = False

        edge_skips = edge_feats["skips"]
        blur_skips = blur_feats["skips"]

        # 5.1: concatenate edge and blur skips of step 5 and fuse them
        # 5.2: 512x19x25 latent Space -> Unpooling -> 512x38x50 FeatMap
        # 5.3: 512x38x50 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 512x38x50 FeatMap
        skip5 = torch.cat([ self.pre5_e(edge_skips[4]), self.pre5_b(blur_skips[4]) ], dim=1)
        skip5 = self.fuse_skip5(skip5)
        x     = self.unpool5(x, edge_feats["idx5"], output_size=edge_feats["size5"])
        x     = self.dec5(x + skip5)
        if debug_prints:
            print(f"Shape after deconv5: {x.shape}")

        # 4.1: concatenate edge and blur skips of step 4 and fuse them
        # 4.2: 512x38x50 FeatMap -> Unpooling -> 512x76x100 FeatMap
        # 4.3: 512x76x100 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 256x76x100 FeatMap
        skip4 = torch.cat([ self.pre4_e(edge_skips[3]), self.pre4_b(blur_skips[3]) ], dim=1)
        skip4 = self.fuse_skip4(skip4)
        x     = self.unpool4(x, edge_feats["idx4"], output_size=edge_feats["size4"])
        x     = self.dec4(x + skip4)
        if debug_prints:
            print(f"Shape after deconv4: {x.shape}")

        # 3.1: concatenate edge and blur skips of step 3 and fuse them
        # 3.2: 256x76x100 FeatMap -> Unpooling -> 256x152x200 FeatMap
        # 3.3: 256x152x200 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 128x152x200 FeatMap
        skip3 = torch.cat([ self.pre3_e(edge_skips[2]), self.pre3_b(blur_skips[2]) ], dim=1)
        skip3 = self.fuse_skip3(skip3)
        x     = self.unpool3(x, edge_feats["idx3"], output_size=edge_feats["size3"])
        x     = self.dec3(x + skip3)
        if debug_prints:
            print(f"Shape after deconv3: {x.shape}")
        # x = self.dec3(x + edge_skips[2])

        # 2.1: concatenate edge and blur skips of step 2 and fuse them
        # 2.2: 128x152x200 FeatMap -> Unpooling -> 128x304x400 FeatMap
        # 2.3: 128x304x400 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 64x304x400 FeatMap
        skip2 = torch.cat([ self.pre2_e(edge_skips[1]), self.pre2_b(blur_skips[1]) ], dim=1)
        skip2 = self.fuse_skip2(skip2)
        x     = self.unpool2(x, edge_feats["idx2"], output_size=edge_feats["size2"])
        x     = self.dec2(x + skip2)
        if debug_prints:
            print(f"Shape after deconv2: {x.shape}")
        # x = self.dec2(x + edge_skips[1])

        # 1.1: concatenate edge and blur skips of step 1 and fuse them
        # 1.2: 64x304x400 FeatMap -> Unpooling -> 64x608x800 FeatMap
        # 1.3: 64x608x800 FeatMap -> Conv -> InstNorm -> ReLu -> Conv -> InstNorm -> ReLu -> 64x608x800 FeatMap
        skip1 = torch.cat([ self.pre1_e(edge_skips[0]), self.pre1_b(blur_skips[0]) ], dim=1)
        skip1 = self.fuse_skip1(skip1)
        x     = self.unpool1(x, edge_feats["idx1"], output_size=edge_feats["size1"])
        x     = self.dec1(x + skip1)
        if debug_prints:
            print(f"Shape after deconv1: {x.shape}")
        # x = self.dec1(x)

        # 0.1: 64x608x800 FeatMap -> Conv -> 3x608x800 Img
        # 0.2: 3x608x800 Img -> tanh -> 3x608x800 Img
        x = torch.tanh(self.to_rgb(x))
        if debug_prints:
            print(f"Shape after tanh: {x.shape}")
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
        img_height (int): Height of the input image (default is 256).
        img_width (int): Width of the input image (default is 256).
        z_dim (int): Dimension of the noise vector (default is 100).
        decoder_relu (bool): Whether to apply ReLU activation in the decoder.
    Returns:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W) where B is batch size, out_channels is number of output channels, H and W are height and width
    '''
    def __init__(self, d=None, img_size=(608, 800), z_dim=100, decoder_relu=True):
        super().__init__()

        # --- parse image size ---
        if isinstance(img_size, int):
            H, W = img_size, img_size
        else:
            H, W = img_size
            assert isinstance(H, int) and isinstance(W, int), "img_size must be int or (H, W)"
        assert H % 32 == 0 and W % 32 == 0, "Both H and W must be divisible by 32 (5x pooling)."

        self.img_size = (H, W)
        self.z_dim = z_dim

        # Edge Map Encoder: 5x Conv + Pooling (return_indices=True)
        self.edge_encoder = VGG16Encoder(in_channels=3)
        # Blur Image Encoder: 5x Conv + Pooling (return_indices=True)
        self.blur_encoder = VGG16Encoder(in_channels=3)

        # Noise -> Bottleneck: can be a rectangle now
        kH, kW = H // 32, W // 32
        self.z_to_bottleneck = nn.ConvTranspose2d(z_dim, 512, kernel_size=(kH, kW),
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
    def __init__(self, num_classes=10, input_size=(608, 800)):
        super(Discriminator, self).__init__()
        self.debug_prints = False

        # --- parse image size ---
        if isinstance(input_size, int):
            H, W = input_size, input_size
        else:
            H, W = input_size
            assert isinstance(H, int) and isinstance(W, int), "input_size must be int or (H, W)"

        self.input_size = (H, W)

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
            *discriminator_block(64, 128),         # 32  -> 16  -> [B,128,H/16,W/16]
        )

        # ---- Wichtig: Global Average Pooling statt Flatten großer Maps ----
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # [B,128,1,1] -> 128-Vector

        # Dynamically compute the size after conv_blocks
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W)
            out = self.conv_blocks(dummy)
            self.flat_features = out.view(1, -1).shape[1]

        self.adv_layer = nn.Linear(128, 1) # nn.Sequential(nn.Linear(self.flat_features, 1), nn.Sigmoid())
        self.use_aux = (num_classes is not None) and (num_classes > 1)
        self.aux_layer = nn.Linear(128, num_classes) if self.use_aux else None # nn.Sequential(nn.Linear(self.flat_features, num_classes), nn.Softmax(dim=1))

    def forward(self, img):
        if self.debug_prints:
            print(f"[Disc] Shape input: {img.shape}")
        x = self.conv_blocks(img)              # [B,128,H/16,W/16]
        if self.debug_prints:
            print(f"[Disc] Shape after conv blocks: {x.shape}")
        x = self.gap(x).flatten(1)             # [B,128]
        if self.debug_prints:
            print(f"[Disc] Shape after flatten: {x.shape}")

        # Heads in FP32 berechnen (stabiler unter AMP)
        if x.dtype != torch.float32:
            x32 = x.float()
        else:
            x32 = x
        if self.debug_prints:
            print(f"[Disc] Shape after float32: {x32.shape}")
        with torch.cuda.amp.autocast(enabled=False):
            adv_logit = self.adv_layer(x32)                        # [B,1]
            aux_logits = self.aux_layer(x32) if self.use_aux else None
        if self.debug_prints:
            print(f"[Disc] Shape adv_logits: {x32.shape}")
            print(f"[Disc] Shape aux_logits: {x32.shape}")

        return adv_logit, aux_logits




def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
