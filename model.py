import torch
from torch import nn
import torchvision.transforms as transforms

class Unet_Encoder(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.input_channel = image_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(image_channels, out_channels = 64, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, out_channels = 128, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(128, out_channels = 128, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, out_channels = 256, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, out_channels = 512, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(512, out_channels = 512, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x_block_1 = self.block1(x)
        x = self.pool(x_block_1)

        x_block_2 = self.block2(x)
        x = self.pool(x_block_2)

        x_block_3 = self.block3(x)
        x = self.pool(x_block_3)

        x_block_4 = self.block4(x)
        x = self.pool(x_block_4)

        x_block_5 = self.block5(x)

        return [x_block_1, x_block_2, x_block_3, x_block_4, x_block_5]


class Unet_Decoder(nn.Module):
    def __init__(self, output_classes):
        super().__init__()
        self.block5 = nn.Sequential(
            nn.Conv2d(1024, out_channels = 1024, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(1024, out_channels = 512, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(512, out_channels = 512, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(512, out_channels = 256,  kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(256, out_channels = 256, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(256, out_channels = 128, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(128, out_channels = 128, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(128, out_channels = 64, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.Conv2d(64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(64, output_classes, kernel_size=1)
    
    def forward(self, encoder_outs):
        encoder_outs_dict = {} 
        for idx, outs in enumerate(encoder_outs):
            key = f"out_{idx+1}"
            encoder_outs_dict[key] = outs

        
        x = self.block5(encoder_outs_dict["out_5"])
        
        x = self.crop_and_concat(x, encoder_outs_dict["out_4"])
        x = self.block4(x)

        x = self.crop_and_concat(x, encoder_outs_dict["out_3"])
        x = self.block3(x)

        x = self.crop_and_concat(x, encoder_outs_dict["out_2"])
        x = self.block2(x)

        x = self.crop_and_concat(x, encoder_outs_dict["out_1"])
        x = self.block1(x)

        out = self.final_conv(x)

        return out
    
    def crop_and_concat(self, deocder_out, encoder_out):
        _, _, H, W = deocder_out.shape
        cropped_feats = transforms.CenterCrop((H, W))(encoder_out)
        out = torch.cat((cropped_feats, deocder_out), dim =1)
        return out

class Unet(nn.Module):
    def __init__(self, input_channels, output_classes):
        super().__init__()
        self.encoder = Unet_Encoder(input_channels)
        self.decoder = Unet_Decoder(output_classes)
    
    def forward(self, image):
        encoder_outs = self.encoder(image)
        out = self.decoder(encoder_outs)
        return out

