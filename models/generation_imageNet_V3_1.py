import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


# Auxiliary module for the encoder
class _Encoder(nn.Module):
    def __init__(self, in_channels):
        super(_Encoder, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)

    def forward(self, x):
        indices = []
        x = self.conv_block1(x)
        x = self.pool1(x)
        x = self.conv_block2(x)
        x = self.pool2(x)
        x = self.conv_block3(x)
        x = self.pool3(x)
        x = self.conv_block4(x)
        x = self.pool4(x)
        x = self.conv_block5(x)
        x = self.pool5(x)
        return x


# Auxiliary module for the decoder (with ConvTranspose2d)
class _DecoderTransposed(nn.Module):
    def __init__(self, in_channels):
        super(_DecoderTransposed, self).__init__()
        # 1536 --> 1024 --> 512 --> 256 --> 128 --> 64 --> 32 --> 16 --> 3

        # Decoder Block 8
        # Replace MaxUnpool2d with ConvTranspose2d
        self.deconv8 = nn.ConvTranspose2d(in_channels, 1536, kernel_size=2, stride=2)
        self.conv_block8 = nn.Sequential(
            nn.Conv2d(1536, 1536, kernel_size=3, padding=1), nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
            nn.Conv2d(1536, 1536, kernel_size=3, padding=1), nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
            nn.Conv2d(1536, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True)
        )

        # Decoder Block 7
        # Replace MaxUnpool2d with ConvTranspose2d
        self.deconv7 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        
        # Decoder Block 6
        # Replace MaxUnpool2d with ConvTranspose2d
        self.deconv6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        # Decoder Block 5
        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        
        # Decoder Block 4
        self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        
        # Decoder Block 3
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        
        # Decoder Block 2
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        
        # Decoder Block 1
        self.deconv1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1), # nn.BatchNorm2d(3), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.deconv8(x)
        x = self.conv_block8(x)
        #x = self.deconv7(x)
        x = self.conv_block7(x)
        x = self.deconv6(x)
        x = self.conv_block6(x)
        #x = self.deconv5(x)
        x = self.conv_block5(x)
        x = self.deconv4(x)
        x = self.conv_block4(x)
        #x = self.deconv3(x)
        x = self.conv_block3(x)
        x = self.deconv2(x)
        x = self.conv_block2(x)
        x = self.deconv1(x)
        x = self.conv_block1(x)
        x = torch.tanh(x)  # Final activation for image output
        return x
    

# The main class now only needs to instantiate the new decoder
class generator(nn.Module):
    def __init__(self, d=None, img_size=None, in_channels_img=3, in_channels_edge=3, noise_dim=100):
        super(generator, self).__init__()
        
        self.encoder_img = _Encoder(in_channels=in_channels_img)
        self.encoder_edge = _Encoder(in_channels=in_channels_edge)
        
        self.noise_processor = nn.Sequential(
            nn.Linear(noise_dim, 512 * 8 * 8),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        
        # Instantiate the new decoder with the correct input channel count
        self.decoder = _DecoderTransposed(in_channels=512 * 3)

    def forward(self, noise_vector, edge_map, img):
        encoded_img, _ = self.encoder_img(img)
        encoded_edge, _ = self.encoder_edge(edge_map)
        
        noise_batch_size = noise_vector.size(0)
        processed_noise = self.noise_processor(noise_vector.view(noise_batch_size, -1))
        processed_noise = processed_noise.view(noise_batch_size, 512, 8, 8)
        
        combined_features = torch.cat([encoded_img, encoded_edge, processed_noise], dim=1)
        
        # Pass to the single decoder
        x = self.decoder(combined_features)
        
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
