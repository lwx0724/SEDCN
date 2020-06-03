
import torch.nn as nn
import torch
from net import common
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,stride=stride,
        padding=(kernel_size//2), bias=bias)

class DecoderBase(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = self._init(in_channels, middle_channels, out_channels)

    def _init(self, in_channels, middle_channels, out_channels):
        raise NotImplementedError

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)

class DecoderDeConv(DecoderBase):
    def _init(self, in_channels, middle_channels, out_channels):
        return nn.Sequential(
            #nn.Dropout2d(p=0.1, inplace=True),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            # Parameters were chosen to avoid artifacts, suggested by https://distill.pub/2016/deconv-checkerboard/
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            #upsample(scale_factor=2)
        )

class UpsamplingBilinear(nn.Module):
    def forward(self, input):
        return F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)

def ConvRelu(*args, **kwargs):
    return nn.Sequential(nn.Conv2d(*args, **kwargs),
                      nn.ReLU(inplace=True))

class DecoderSimpleNBN(DecoderBase):
    """as dsb2018_topcoders
    from https://github.com/selimsef/dsb2018_topcoders/blob/master/selim/models/unets.py#L76
    """
    def _init(self, in_channels, middle_channels, out_channels):
        return nn.Sequential(
            ConvRelu(in_channels, middle_channels, kernel_size=3, padding=1),
            UpsamplingBilinear(),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

def make_model(args):
    return SRhourglass(args)

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction=16,stride=1,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(RCAB, self).__init__()
        modules_body = []
        out_feat = n_feat
        in_feat  = n_feat
        if stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_feat,n_feat,kernel_size=1,stride=stride,bias=False))
        else:
            self.downsample =None

        for i in range(2):
            modules_body.append(conv(in_feat, out_feat, kernel_size,stride = stride,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(out_feat))
            if i == 0:
                modules_body.append(act)
                stride = 1
        modules_body.append(CALayer(in_feat, reduction))
        self.body = nn.Sequential(*modules_body)


    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        if self.downsample:
            x = self.downsample(x)
            res +=x
        else:
            res += x

        return res

class hourglass(nn.Module):
    def __init__(self,blocks,in_feats,Decoder=DecoderDeConv,n_feats=64):
        super(hourglass, self).__init__()
        self.conv1 = nn.Conv2d(in_feats,n_feats,kernel_size=1,stride=1,padding=0,bias=True)

        self.layer1 = self._make_layer(n_feats,blocks[0],True) #48*48
        self.layer2 = self._make_layer(n_feats,blocks[1])
        self.layer3 = self._make_layer(n_feats,blocks[2])
        self.layer4 = self._make_layer(n_feats,blocks[3])

        self.decoder4 = Decoder(n_feats,int(n_feats*0.75),n_feats)
        self.decoder3 = Decoder(n_feats*2,int(n_feats*1.5),n_feats)
        self.decoder2 = Decoder(n_feats*2,int(n_feats*1.5),n_feats)
        self.decoder1 = nn.Conv2d(n_feats*2,n_feats,kernel_size=3,stride=1,padding=3//2,bias=True)

    def _make_layer(self,inplanes,blocks,isFirst=False):
        layers =[]

        if isFirst:
            for i in range(blocks):
                layers.append(RCAB(default_conv,inplanes,kernel_size=3,stride=1))
        else:
            layers.append(RCAB(default_conv,inplanes,kernel_size=3,stride=2))
            for i in range(1,blocks):
                layers.append(RCAB(default_conv,inplanes,kernel_size=3,stride=1))

        return  nn.Sequential(*layers)

    def forward(self,x):
        x  = self.conv1(x)
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4,e3)
        d2 = self.decoder2(d3,e2)
        d1 = self.decoder1(torch.cat((d2,e1),1))

        return  d1

class SRhourglass(nn.Module):
    def __init__(self,args,conv=default_conv):
        super(SRhourglass,self).__init__()
        n_resblocks = 6
        n_feats = args.n_feats
        kernel_size =3
        rgb_mean =(0.4488,0.4371,0.4040)
        rgb_std =(1.0,1.0,1.0)
        blocks=[10,5,3,2]
        scale = args.scale
        self.scale0 = int(args.scale)
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range,rgb_mean,rgb_std)
        modules_head = [conv(args.n_colors,n_feats,kernel_size)]

        modules_body =[hourglass(blocks,in_feats=n_feats*(i+1),n_feats=n_feats) for i in range(n_resblocks)]

        #conv1x1
        self.last_conv1 = nn.Conv2d(n_feats*(n_resblocks+1), n_feats, kernel_size=1, stride=1, padding=0, bias=True)

        # define tail module
        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range,rgb_mean, rgb_std,sign=1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x1 = self.sub_mean(x)
        x = self.head(x1)
        h = self.body[0](x)
        concat_h = torch.cat((x,h),1)

        for i in range(1,6):
            h = self.body[i](concat_h)
            concat_h = torch.cat((concat_h,h),1)

        x =self.last_conv1(concat_h)
        x = self.tail(x)

        x = self.add_mean(x)

        return x


