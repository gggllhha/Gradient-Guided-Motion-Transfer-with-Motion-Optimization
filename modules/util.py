import numbers
import os

import torchvision
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        os.environ['TORCH_HOME'] = os.path.abspath(os.getcwd())
        blocks = [torchvision.models.vgg16(weights='DEFAULT').features[:4].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval(),
                  torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
        y = F.interpolate(y, mode='bilinear', size=(224, 224), align_corners=False)
        perceptual_loss = 0.0
        style_loss = 0.0

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

            perceptual_loss += torch.nn.functional.l1_loss(x, y)

            # b, ch, h, w = x.shape
            # act_x = x.reshape(x.shape[0], x.shape[1], -1)
            # act_y = y.reshape(y.shape[0], y.shape[1], -1)
            # gram_x = act_x @ act_x.permute(0, 2, 1) / (ch * h * w)
            # gram_y = act_y @ act_y.permute(0, 2, 1) / (ch * h * w)
            # style_loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        return perceptual_loss  # , style_loss


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


class FB(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super(FB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               padding=padding)

    def forward(self, x):
        res = self.conv1(x)
        res = F.relu(res)
        res = self.conv2(res)
        res += x
        return res


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(FusionBlock, self).__init__()
        self.fusion_block = nn.Sequential(FB(in_channels, kernel_size, padding),
                                          FB(in_channels, kernel_size, padding),
                                          FB(in_channels, kernel_size, padding),
                                          FB(in_channels, kernel_size, padding))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = self.fusion_block(x)
        res += x
        res = self.conv(res)
        return res


class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_type='reflect'):
        super(FuseBlock, self).__init__()
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=p),
                       nn.InstanceNorm2d(in_channels, affine=True), nn.ReLU(inplace=True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=p),
                       nn.InstanceNorm2d(in_channels, affine=True)]
        self.conv_block = nn.Sequential(*conv_block)
        self.conv_last = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = self.conv_last(out)
        return out


class TPS:
    '''
    TPS transformation, mode 'kp' for Eq(2) in the paper, mode 'random' for equivariance loss.
    '''

    def __init__(self, mode, bs, **kwargs):
        self.bs = bs
        self.mode = mode
        if mode == 'random':
            noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
            self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        elif mode == 'kp':
            kp_1 = kwargs["kp_1"]
            kp_2 = kwargs["kp_2"]
            device = kp_1.device
            kp_type = kp_1.type()
            self.gs = kp_1.shape[1]
            n = kp_1.shape[2]
            K = torch.norm(kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2)
            K = K ** 2
            K = K * torch.log(K + 1e-9)

            one1 = torch.ones(self.bs, kp_1.shape[1], kp_1.shape[2], 1).to(device).type(kp_type)
            kp_1p = torch.cat([kp_1, one1], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 3).to(device).type(kp_type)
            P = torch.cat([kp_1p, zero], 2)
            L = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2)
            L = torch.cat([L, P], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 2).to(device).type(kp_type)
            Y = torch.cat([kp_2, zero], 2)
            one = torch.eye(L.shape[2]).expand(L.shape).to(device).type(kp_type) * 0.01
            L = L + one

            param = torch.matmul(torch.inverse(L), Y)
            self.theta = param[:, :, n:, :].permute(0, 1, 3, 2)

            self.control_points = kp_1
            self.control_params = param[:, :, :n, :]
        else:
            raise Exception("Error TPS mode")

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0).to(frame.device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, frame.shape[2], frame.shape[3], 2]
        if self.mode == 'kp':
            shape.insert(1, self.gs)
        grid = self.warp_coordinates(grid).view(*shape)
        return grid

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type()).to(coordinates.device)
        control_points = self.control_points.type(coordinates.type()).to(coordinates.device)
        control_params = self.control_params.type(coordinates.type()).to(coordinates.device)

        if self.mode == 'kp':
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1)) + theta[:, :, :, 2:]

            distances = coordinates.view(coordinates.shape[0], 1, 1, -1, 2) - control_points.view(self.bs,
                                                                                                  control_points.shape[
                                                                                                      1], -1, 1, 2)

            distances = distances ** 2
            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
            transformed = transformed.permute(0, 1, 3, 2) + result

        elif self.mode == 'random':
            theta = theta.unsqueeze(1)
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
            transformed = transformed.squeeze(-1)
            ances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = ances ** 2

            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        else:
            raise Exception("Error TPS mode")

        return transformed


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """

    coordinate_grid = make_coordinate_grid(spatial_size, kp.type()).to(kp.device)
    number_of_leading_dimensions = len(kp.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = kp.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = kp.shape[:number_of_leading_dimensions] + (1, 1, 2)
    kp = kp.view(*shape)

    mean_sub = (coordinate_grid - kp)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = nn.InstanceNorm2d(in_features, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        # print('encoder:' ,outs[-1].shape)
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []
        self.out_channels = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_channels.append(block_expansion + in_features)
        # self.out_filters = block_expansion + in_features

    def forward(self, x, mode=0):
        out = x.pop()
        outs = []
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
            outs.append(out)
        if (mode == 0):
            return out
        else:
            return outs


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_channels = self.decoder.out_channels
        # self.out_filters = self.decoder.out_filters

    def forward(self, x, mode=0):
        return self.decoder(self.encoder(x), mode)


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def to_homogeneous(coordinates):
    ones_shape = list(coordinates.shape)
    ones_shape[-1] = 1
    ones = torch.ones(ones_shape).type(coordinates.type())

    return torch.cat([coordinates, ones], dim=-1)


def from_homogeneous(coordinates):
    return coordinates[..., :2] / coordinates[..., 2:3]
