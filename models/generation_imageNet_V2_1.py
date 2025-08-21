import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

# --------- Helfer ----------
def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1, use_relu=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if use_relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class VGG16Encoder(nn.Module):
    """2-2-3-3-3 Conv-Blöcke, MaxPool(return_indices=True). Liefert auch Größen & Indizes."""
    def __init__(self, in_channels=3):
        super().__init__()
        # Block 1: 2 conv -> pool
        self.enc1 = nn.Sequential(conv_bn_relu(in_channels, 64),
                                  conv_bn_relu(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 2: 2 conv -> pool
        self.enc2 = nn.Sequential(conv_bn_relu(64, 128),
                                  conv_bn_relu(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 3: 3 conv -> pool
        self.enc3 = nn.Sequential(conv_bn_relu(128, 256),
                                  conv_bn_relu(256, 256),
                                  conv_bn_relu(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 4: 3 conv -> pool
        self.enc4 = nn.Sequential(conv_bn_relu(256, 512),
                                  conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 5: 3 conv -> pool
        self.enc5 = nn.Sequential(conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512),
                                  conv_bn_relu(512, 512))
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = self.enc1(x); size1 = x.size(); x, idx1 = self.pool1(x)  # 256->128
        x = self.enc2(x); size2 = x.size(); x, idx2 = self.pool2(x)  # 128->64
        x = self.enc3(x); size3 = x.size(); x, idx3 = self.pool3(x)  # 64->32
        x = self.enc4(x); size4 = x.size(); x, idx4 = self.pool4(x)  # 32->16
        x = self.enc5(x); size5 = x.size(); x, idx5 = self.pool5(x)  # 16->8
        return {
            "p5": x, "idx5": idx5, "size5": size5,
            "p4": None, "idx4": idx4, "size4": size4,  # sizes für Decoder
            "p3": None, "idx3": idx3, "size3": size3,
            "p2": None, "idx2": idx2, "size2": size2,
            "p1": None, "idx1": idx1, "size1": size1,
        }

class VGG16Decoder(nn.Module):
    """Spiegelbildlich: MaxUnpool + Conv-Blöcke (3-3-3-2-2 Convs)."""
    def __init__(self, decoder_relu=True, out_channels=3):
        super().__init__()
        self.relu = decoder_relu

        # Unpool5 -> 3 conv (512,512,512)
        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.dec5 = nn.Sequential(
            conv_bn_relu(512, 512, use_relu=self.relu),
            conv_bn_relu(512, 512, use_relu=self.relu),
            conv_bn_relu(512, 512, use_relu=self.relu),
        )

        # Unpool4 -> 3 conv (512,512,256)
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.dec4 = nn.Sequential(
            conv_bn_relu(512, 512, use_relu=self.relu),
            conv_bn_relu(512, 512, use_relu=self.relu),
            conv_bn_relu(512, 256, use_relu=self.relu),
        )

        # Unpool3 -> 3 conv (256,256,128)
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec3 = nn.Sequential(
            conv_bn_relu(256, 256, use_relu=self.relu),
            conv_bn_relu(256, 256, use_relu=self.relu),
            conv_bn_relu(256, 128, use_relu=self.relu),
        )

        # Unpool2 -> 2 conv (128,64)
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec2 = nn.Sequential(
            conv_bn_relu(128, 128, use_relu=self.relu),
            conv_bn_relu(128, 64, use_relu=self.relu),
        )

        # Unpool1 -> 2 conv (64,64)
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec1 = nn.Sequential(
            conv_bn_relu(64, 64, use_relu=self.relu),
            conv_bn_relu(64, 64, use_relu=self.relu),
        )

        # Bildausgabe (Generator): 3 Kanäle + tanh
        self.to_rgb = nn.Conv2d(64, out_channels, kernel_size=1, bias=True)

    def forward(self, x, idxs_sizes):
        # idxs_sizes: Dict aus Encoder (von edgeMap-Encoder)
        x = self.unpool5(x, idxs_sizes["idx5"], output_size=idxs_sizes["size5"]); x = self.dec5(x)
        x = self.unpool4(x, idxs_sizes["idx4"], output_size=idxs_sizes["size4"]); x = self.dec4(x)
        x = self.unpool3(x, idxs_sizes["idx3"], output_size=idxs_sizes["size3"]); x = self.dec3(x)
        x = self.unpool2(x, idxs_sizes["idx2"], output_size=idxs_sizes["size2"]); x = self.dec2(x)
        x = self.unpool1(x, idxs_sizes["idx1"], output_size=idxs_sizes["size1"]); x = self.dec1(x)
        x = torch.tanh(self.to_rgb(x))
        return x

# --------- Dein Generator in SegNet-Topologie ----------
class generator(nn.Module):
    """
    - Zwei VGG16-Encoder (edgeMap & blurImg), jeweils 13 Convs + MaxPool(indizes)
    - Bottleneck: Mittelung beider 8x8x512-Tensoren (keine extra Conv, Schichtanzahl bleibt exakt)
    - Noise: Projektion 100->8x8x512 per ConvTranspose2d und additiv
    - Decoder: Spiegelbild (13 Convs) mit MaxUnpool, Indizes vom edgeMap-Encoder
    - Output: 3-Kanal-Bild mit tanh
    """
    def __init__(self, d=None, img_size=256, z_dim=100, decoder_relu=True):
        super().__init__()
        assert img_size % 32 == 0, "img_size muss durch 32 teilbar sein (5x Pooling)."
        self.img_size = img_size
        self.z_dim = z_dim

        # Zwei getrennte Encoder
        self.edge_encoder = VGG16Encoder(in_channels=3)
        self.blur_encoder = VGG16Encoder(in_channels=3)

        # Noise -> 8x8x512 (bei 256x256 nach 5x Pooling)
        self.z_to_bottleneck = nn.ConvTranspose2d(z_dim, 512, kernel_size=img_size // 32,
                                                  stride=1, padding=0, bias=True)
        # Decoder
        self.decoder = VGG16Decoder(decoder_relu=decoder_relu, out_channels=3)

        # Initialisierung
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
        # --- Encoder (edgeMap) ---
        edge_feats = self.edge_encoder(edgeMap)   # enthält p5 (8x8x512), idx1..5, size1..5

        # --- Encoder (blurImg) ---
        blur_feats = self.blur_encoder(blurImg)   # nur p5 wird verwendet (keine extra Schichten)

        # --- Bottleneck-Fusion (keine zusätzliche Conv) ---
        # 8x8x512 beider Streams mitteln (hält die Schichtanzahl im Encoder/Decoder exakt)
        bottleneck = 0.5 * (edge_feats["p5"] + blur_feats["p5"])

        # --- Noise einmischen ---
        # noiseVector erwartet [B, z_dim, 1, 1]; ConvT macht daraus [B,512,8,8], dann additiv
        z_feat = self.z_to_bottleneck(noiseVector)
        if z_feat.shape[-2:] != bottleneck.shape[-2:]:
            raise RuntimeError(f"Noise feature size {z_feat.shape[-2:]} != bottleneck {bottleneck.shape[-2:]}")
        bottleneck = bottleneck + z_feat

        # --- Decoder (Spiegel zu 13 Convs, MaxUnpool mit edge-Indices) ---
        out = self.decoder(bottleneck, edge_feats)
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
