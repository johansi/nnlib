import torch
import torch.nn as nn
import warnings
import torchvision
import pdb

warnings.filterwarnings(action='ignore')

# https://github.com/lvpeiqing/SAR-U-Net-liver-segmentation/blob/master/models/se_p_resunet/se_p_resunet.py

class AddCoords(nn.Module):

    def __init__(self, with_r=False, with_boundary=False):
        super().__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, boundary_map):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3).cuda()
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3).cuda()

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and (boundary_map is not None):
            # B, 64(H), 64(W)
            boundary_map = boundary_map.view(boundary_map.shape[0],1,boundary_map.shape[1],boundary_map.shape[2])
            boundary_channel = torch.clamp(boundary_map,0.0, 1.0)
            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel>0.05, xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel>0.05, yy_channel, zero_tensor)

            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel], dim=1)

        return ret


class CoordConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, with_r=True, with_boundary=False,**kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r,with_boundary=with_boundary)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        if with_boundary:
            in_size += 2
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x, boundary_map=None):
        ret = self.addcoords(x, boundary_map)
        ret = self.conv(ret)
        return ret

'''
class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r        

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32, device=input_tensor.device)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32, device=input_tensor.device)

            xx_range = torch.arange(dim_y, dtype=torch.int32, device=input_tensor.device)
            yy_range = torch.arange(dim_x, dtype=torch.int32, device=input_tensor.device)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out
'''

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18], use_stride=True):
        super(ASPP, self).__init__()

        self.down_sample = nn.Conv2d(in_dims, in_dims, kernel_size=2, stride=2, padding=0) if use_stride else nn.MaxPool2d(2)
        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x = self.down_sample(x)
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


class ASPP1(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP1, self).__init__()


        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):

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


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            CoordConv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            #nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),            
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),            
            nn.ReLU()
        )        

        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),            
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)        
        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, use_stride=True):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, padding=0) if use_stride else nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch-out_ch, in_ch-out_ch, kernel_size=2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, rate=0.1):
        super(outconv, self).__init__()
        self.dropout = dropout
        if dropout:
            print('dropout', rate)
            self.dp = nn.Dropout2d(rate)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        if self.dropout:
            x = self.dp(x)
        x = self.conv(x)
        return x


class SE_Res_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_features=64,  dropout=False, rate=0.1, bilinear=False):
        super(SE_Res_UNet, self).__init__()
        self.inc = inconv(n_channels, init_features)

        self.selayer1 = SELayer(init_features)

        self.down1 = down(init_features, init_features*2)

        self.selayer2 = SELayer(init_features*2)

        self.down2 = down(init_features*2, init_features*4)

        self.selayer3 = SELayer(init_features*4)

        self.down3 = down(init_features*4, init_features*8)

        self.selayer4 = SELayer(init_features*8)

        self.ASSP = ASPP(init_features*8,init_features*16)

        self.up1 = up(init_features*8 + init_features*16, init_features*8)
        self.up2 = up(init_features*4 + init_features*8, init_features*4)
        self.up3 = up(init_features*2 + init_features*4, init_features*2)
        self.up4 = up(init_features + init_features*2, init_features)
        self.ASSP_1 = ASPP1(init_features,init_features)
        self.outc = outconv(init_features, n_classes, dropout, rate)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.selayer1(x1)
        x2 = self.down1(x1)
        x2 = self.selayer2(x2)
        x3 = self.down2(x2)
        x3 = self.selayer3(x3)
        x4 = self.down3(x3)
        x4 = self.selayer4(x4)
        x5 = self.ASSP(x4)
        x44 = self.up1(x5, x4)
        x33 = self.up2(x44, x3)
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        x00 = self.ASSP_1(x11)
        x0 = self.outc(x00)
        return x0

