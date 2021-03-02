from collections import OrderedDict

import torch
import torch.nn as nn
import pdb

res34_downsample = [3,4,6,3]

downsample_none = 0
downsample_maxpool = 1
downsample_stride = 2

upsample_none = 0
upsample_upsample = 1
upsample_conv = 2

class AttentionBlock(nn.Module):
    def __init__(self, in_channels_encoder, in_channels_decoder, features):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(in_channels_encoder),
            nn.ReLU(),
            nn.Conv2d(in_channels_encoder, features, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(in_channels_decoder),
            nn.ReLU(),
            nn.Conv2d(in_channels_decoder, features, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, 1, 1),
        )

    def forward(self, decoder, encoder):        
        out = self.conv_encoder(encoder) + self.conv_decoder(decoder)
        out = self.conv_attn(out)
        return out * decoder

class ASPP(nn.Module):
    def __init__(self, in_channels, features, downsampling, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        if downsampling == downsample_stride:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),            
            )
        elif downsampling == downsample_maxpool:
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)        
        
        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels, features, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels, features, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels, features, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features),
        )

        self.output = nn.Conv2d(len(rate) * features, features, 1)
        self._init_weights()
        self.downsampling=downsampling

    def forward(self, x):
        if self.downsampling != downsample_none:
            x = self.down(x)
            
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def _layerBlocks(in_channels, features, downsample_method=downsample_maxpool, blocksize=1, squeeze_excite=None, 
                 resblock=False, bn_relu_at_first=False, bn_relu_at_end=False):   
    layers = OrderedDict([])
    for idx in range(blocksize):
        downsample = downsample_method if idx == 0 else downsample_none 
        in_channels = in_channels if idx == 0 else features
        layers["block_"+str(idx)] = Block(in_channels,features,downsample,squeeze_excite,resblock,bn_relu_at_first, bn_relu_at_end)
    return nn.Sequential(layers)
            
class UpSample(nn.Module):
    def __init__(self, in_channels, features, upsample_method=upsample_conv, bias=True):
        super(UpSample, self).__init__()
        if upsample_method == upsample_upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:            
            self.up = nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2, bias=bias)            
            
    def forward(self, x):
        return self.up(x)
    
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, features,kernel_size=3,padding=1,stride=1, bn_relu=True,bias=True):
        super(ConvLayer, self).__init__()
        layers = OrderedDict([])
        layers["conv"] = nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=3,padding=1,stride=stride,bias=bias)         
        if bn_relu:
            layers["bn_relu"] = nn.Sequential(nn.BatchNorm2d(num_features=features),nn.ReLU(inplace=True))
        
        self.layer = nn.Sequential(layers)

    def forward(self, x):
        return self.layer(x)
        
class Block(nn.Module):
    def __init__(self, in_channels, features, downsample=downsample_none, do_squeeze_excite=False, resblock=False, 
                 bn_relu_at_first=False, bn_relu_at_end=False):
        super(Block, self).__init__()
        
        layers = OrderedDict([])        
        
        if bn_relu_at_first:
            self.bn_relu = nn.Sequential(nn.BatchNorm2d(num_features=in_channels),nn.ReLU(inplace=True))
                    
        if downsample == downsample_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if (downsample == downsample_none) or (downsample == downsample_maxpool):
            layers["conv1"] = ConvLayer(in_channels,features)            
        elif downsample == downsample_stride:
            layers["conv1"] = ConvLayer(in_channels,features, padding=0,stride=2)
            
        layers["conv2"] = ConvLayer(features,features, bn_relu=bn_relu_at_end)
        
        if do_squeeze_excite:
            layers["squeeze_excite"] = Squeeze_Excite_Block(features)
        
        self.block = nn.Sequential(layers)
        
        if resblock:
            params_skip_conn = (1,1) if (downsample == downsample_none) or (downsample == downsample_maxpool) else (0,2)
            self.skip_conn = nn.Sequential(nn.Conv2d(in_channels,features,kernel_size=3,padding=params_skip_conn[0],
                                                     stride=params_skip_conn[1],bias=True),nn.BatchNorm2d(num_features=features))
        
        self.resblock = resblock   
        self.downsample = downsample
        self.bn_relu_at_first = bn_relu_at_first 
        
    def forward(self, x):        
        
        if self.bn_relu_at_first:
             x = self.bn_relu(x)
                    
        if self.downsample == downsample_maxpool:
            x = self.maxpool(x)
            
        out = self.block(x)
                                
        if self.resblock:            
            out = self.skip_conn(x)+out
                            
        return out

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, block_sizes_down=[1,1,1,1], blocksize_bottleneck=1, block_sizes_up=[1,1,1,1], downsample_method=downsample_maxpool, upsample_method=upsample_conv, resblock=False, squeeze_excite=False, aspp=False, attention=False, bn_relu_at_first=False, bn_relu_at_end=False):
        
        super(UNet, self).__init__()        
        features = init_features        
        self.squeeze_excite = squeeze_excite
        
        # DECODER
        self.encoder1 = _layerBlocks(in_channels,features,downsample_none,block_sizes_down[0],squeeze_excite, 
                                     resblock, False, bn_relu_at_end)
        
        self.encoder2 = _layerBlocks(features,features*2,downsample_method,block_sizes_down[1],squeeze_excite, 
                                     resblock, bn_relu_at_first, bn_relu_at_end)
        
        self.encoder3 = _layerBlocks(features*2,features*4,downsample_method,block_sizes_down[2],squeeze_excite, 
                                     resblock, bn_relu_at_first, bn_relu_at_end)
        
        self.encoder4 = _layerBlocks(features*4,features*8,downsample_method,block_sizes_down[3],squeeze_excite, 
                                     resblock, bn_relu_at_first, bn_relu_at_end)
        
        # BOTTLENECK
        if aspp:
            self.bottleneck = ASPP(features*8, features*16, downsample_method)
        else:
            self.bottleneck = _layerBlocks(features*8,features*16,downsample_method,blocksize_bottleneck,False,resblock)                
        
        # ENCODER
        if attention:
            self.attn1 = AttentionBlock(features*8, features*16, features * 16)
            
        self.upsample4 = UpSample(features * 16, features * 8, upsample_method)        
        smaple_factor = 8 if upsample_method==upsample_conv else 16
        self.decoder4 = _layerBlocks((features * smaple_factor) * 2,features*smaple_factor,downsample_none,
                                     block_sizes_up[3], False, resblock, bn_relu_at_first, bn_relu_at_end)

        if attention:
            self.attn2 = AttentionBlock(features*4, features*8, features * 8)
        
        self.upsample3 = UpSample(features * 8, features * 4, upsample_method)        
        smaple_factor = 4 if upsample_method==upsample_conv else 8
        self.decoder3 = _layerBlocks((features * smaple_factor) * 2,features*smaple_factor,downsample_none,
                                     block_sizes_up[2], False, resblock, bn_relu_at_first, bn_relu_at_end)
        
        if attention:
            self.attn3 = AttentionBlock(features*2, features*4, features * 4)
        
        self.upsample2 = UpSample(features * 4, features * 2, upsample_method)        
        smaple_factor = 2 if upsample_method==upsample_conv else 4
        self.decoder2 = _layerBlocks((features * smaple_factor) * 2,features*smaple_factor,downsample_none,
                                     block_sizes_up[1], False, resblock, bn_relu_at_first, bn_relu_at_end)

        if attention:
            self.attn4 = AttentionBlock(features, features*2, features * 2)

        self.upsample1 = UpSample(features * 2, features, upsample_method)        
        smaple_factor = 1 if upsample_method==upsample_conv else 2
        self.decoder1 = _layerBlocks((features * smaple_factor) * 2,features*smaple_factor,downsample_none,
                                     block_sizes_up[0], False, resblock, bn_relu_at_first, bn_relu_at_end)

        if aspp:
            smaple_factor = 1 if upsample_method==upsample_conv else 2
            self.out_aspp = ASPP(features*smaple_factor, features*smaple_factor, downsample_none)
        
        smaple_factor = 1 if upsample_method==upsample_conv else 2
        self.out_conv = nn.Conv2d(in_channels=features*smaple_factor, out_channels=out_channels, kernel_size=1)
        
        self.aspp = aspp
        self.attention = attention

    def forward(self, x):
        enc1 = self.encoder1(x)           
        enc2 = self.encoder2(enc1)            
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)                               

        bottle = self.bottleneck(enc4)

        if self.attention:
            bottle = self.attn1(bottle,enc4)
            
        dec4 = self.upsample4(bottle)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        if self.attention:
            dec4 = self.attn2(dec4,enc3)

        dec3 = self.upsample3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        if self.attention:
            dec3 = self.attn3(dec3,enc2)        
        
        dec2 = self.upsample2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        if self.attention:
            dec2 = self.attn4(dec2,enc1)                
        
        dec1 = self.upsample1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        if self.aspp:
            dec1 = self.out_aspp(dec1)
            
        return torch.sigmoid(self.out_conv(dec1))
 
