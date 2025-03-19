import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Residual connection
        return self.relu(out)

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_size, downsampling_factor=2):
        """
        Args:
            in_channels (int): Number of channels in the input.
            latent_dim (int): Number of channels in the latent space.
            hidden_size (int): Number of channels for the hidden feature maps.
            downsampling_factor (int): Factor by which the spatial dimensions are reduced.
        """
        super(AutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.downsampling_factor = downsampling_factor

        # Encoder: Downsample spatially first then refine features.
        num_downsampling = int(torch.log2(torch.tensor(downsampling_factor)))

        def conv_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def deconv_block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=stride, 
                                padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def res_blocks(ch, num=2):
            return nn.Sequential(*[ResidualBlock(ch) for _ in range(num)])

        # Build Encoder
        encoder_layers = [
            conv_block(in_channels, hidden_size, stride=1),
            res_blocks(hidden_size)
        ]
        in_ch = hidden_size
        for i in range(num_downsampling):
            # For the final stage, use latent_dim; otherwise double the channels.
            out_ch = latent_dim if i == num_downsampling - 1 else in_ch * 2
            encoder_layers += [
                conv_block(in_ch, out_ch, stride=2),
                res_blocks(out_ch)
            ]
            in_ch = out_ch

        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder
        decoder_layers = [res_blocks(latent_dim)]
        in_ch = latent_dim
        # Loop in reverse to mirror the encoder.
        for i in reversed(range(num_downsampling)):
            out_ch = hidden_size * (2 ** i)
            decoder_layers += [
                deconv_block(in_ch, out_ch, stride=2),
                res_blocks(out_ch)
            ]
            in_ch = out_ch

        decoder_layers += [nn.Conv2d(hidden_size, in_channels, kernel_size=3, stride=1, padding=1)]
        self.decoder = nn.Sequential(*decoder_layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction