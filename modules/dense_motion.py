from torch import nn
import torch.nn.functional as F
import torch

from modules.swin import TransformerWarper2d, SwinTransformer2d
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian, coords_grid
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d, TPS
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        x = x + x * out
        return x


class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks 
                        from K TPS transformations and an affine transformation.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_tps, num_channels,
                 scale_factor=0.25, bg=False, multi_mask=True, kp_variance=0.01, **kwargs):
        super(DenseMotionNetwork, self).__init__()

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        self.hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=(num_channels * (num_tps + 1) + num_tps * 5 + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3))
        if multi_mask:
            up = []
            self.up_nums = int(math.log(1 / scale_factor, 2))
            self.occlusion_num = 4

            channel = [hourglass_output_size[-1] // (2 ** i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            channel = [hourglass_output_size[-i - 1] for i in range(self.occlusion_num - self.up_nums)[::-1]]
            for i in range(self.up_nums):
                channel.append(hourglass_output_size[-1] // (2 ** (i + 1)))
            occlusion = []

            for i in range(self.occlusion_num):
                occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
            self.occlusion = nn.ModuleList(occlusion)
        else:
            occlusion = [nn.Conv2d(hourglass_output_size[-1], 1, kernel_size=(7, 7), padding=(3, 3))]
            self.occlusion = nn.ModuleList(occlusion)

        self.num_tps = num_tps
        self.bg = bg
        self.kp_variance = kp_variance

        self.refine_proj_query_feat = nn.Sequential(
            nn.Conv2d(hourglass_output_size[-1], 94, 1), nn.ReLU()
        )

        self.refine_swin_decoder_1_1 = nn.ModuleList([
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(
                        img_size=(32, 32), embed_dim=96, window_size=4, num_heads=[8],
                        depths=[1], shift=False
                    )
                ),
                ChannelAttention(96),
            ) for i in range(4)
        ])

        self.refine_swin_decoder_1_2 = nn.ModuleList([
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(
                        img_size=(32, 32), embed_dim=96, window_size=4, num_heads=[8],
                        depths=[1], shift=True
                    )
                ),
                ChannelAttention(96),
            ) for i in range(4)
        ])

        self.refine_swin_decoder_2_1 = nn.ModuleList([
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(
                        img_size=(32, 32), embed_dim=96, window_size=8, num_heads=[8],
                        depths=[1], shift=False
                    )
                ),
                ChannelAttention(96),
            ) for i in range(4)
        ])

        self.refine_swin_decoder_2_2 = nn.ModuleList([
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(
                        img_size=(32, 32), embed_dim=96, window_size=8, num_heads=[8],
                        depths=[1], shift=True
                    )
                ),
                ChannelAttention(96),
            ) for i in range(4)
        ])
        self.refine2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(96, 32, (3, 3), padding=(1, 1), bias=True),
                nn.ReLU(),
                nn.Conv2d(32, 2, (3, 3), padding=(1, 1), bias=True),
            ) for i in range(4)
        ])
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.linear_proj = nn.ModuleList([
            nn.Linear(96 * (((32 * (2 ** i)) // 32) ** 2), 96) for i in range(4)
        ])
        self.linear_proj_reverse = nn.ModuleList([
            nn.Linear(96, 96 * (((32 * (2 ** i)) // 32) ** 2)) for i in range(4)
        ])

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):

        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(
            heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        # K TPS transformaions
        bs, _, h, w = source_image.shape
        kp_1 = kp_driving['fg_kp']
        kp_2 = kp_source['fg_kp']
        kp_1 = kp_1.view(bs, -1, 5, 2)
        kp_2 = kp_2.view(bs, -1, 5, 2)  # bs num_tps 5 2
        trans = TPS(mode='kp', bs=bs, kp_1=kp_1, kp_2=kp_2)
        driving_to_source = trans.transform_frame(source_image)  # bs 10 64 64 2

        identity_grid = make_coordinate_grid((h, w), type=kp_1.type()).to(kp_1.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # affine background transformation
        if not (bg_param is None):
            identity_grid = to_homogeneous(identity_grid)
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
            identity_grid = from_homogeneous(identity_grid)

        transformations = torch.cat([identity_grid, driving_to_source], dim=1)
        return transformations

    def create_deformed_source_image(self, source_image, transformations):

        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_tps + 1), -1, h, w)
        transformations = transformations.view((bs * (self.num_tps + 1), h, w, -1))
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, self.num_tps + 1, -1, h, w))
        return deformed

    def dropout_softmax(self, X, P):
        '''
        Dropout for TPS transformations. Eq(7) and Eq(8) in the paper.
        '''
        drop = (torch.rand(X.shape[0], X.shape[1]) < (1 - P)).type(X.type()).to(X.device)
        drop[..., 0] = 1  # background
        drop = drop.repeat(X.shape[2], X.shape[3], 1, 1).permute(2, 3, 0, 1)  # bs 11 h w

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:, 1:, ...] /= (1 - P)
        mask_bool = (drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition

    def apply_dropout(self, dropout, *feats):
        sizes = [x.shape[-2:] for x in feats]
        max_size = max(sizes)
        resized_feats = [F.interpolate(x, size=max_size, mode="nearest") for x in feats]

        channel_list = [x.size(1) for x in feats]
        feats = dropout(torch.cat(resized_feats, dim=1))
        feats = torch.split(feats, channel_list, dim=1)
        recoverd_feats = [
            F.interpolate(x, size=size, mode="nearest") for x, size in zip(feats, sizes)
        ]
        return recoverd_feats

    def normalize_flow(self, coords):
        _, H, W, _ = coords.shape
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)

        return grid

    def window_partition(self, feature_map, h, w, index):
        bs, C, H, W = feature_map.shape

        assert H % h == 0, f"Height {H} must be divisible by h {h}"
        assert W % w == 0, f"Width {W} must be divisible by w {w}"

        # 将特征图划分为 (h, w) 个窗口
        window_feature = feature_map.view(bs, C, h, H // h, w, W // w)
        window_feature = window_feature.permute(0, 2, 4, 1, 3, 5)  # (bs, h, w, C, H_window, W_window)
        window_feature = window_feature.contiguous().view(bs, h, w, -1)
        proj_feature = self.linear_proj[index](window_feature).permute(0, 3, 1, 2)

        return proj_feature

    def window_reverse(self, windows, h, w, H, W, C, index):
        bs, _, _, D = windows.shape
        H_window = H // h
        W_window = W // w

        proj_feature = self.linear_proj_reverse[index](windows)  # (bs, h, w, C*H_window*W_window)
        feature = proj_feature.view(bs, h, w, C, H_window, W_window)
        feature = feature.permute(0, 3, 1, 4, 2, 5)  # (bs, C, h, H_window, w, W_window)
        feature = feature.contiguous().view(bs, C, H, W)

        return feature

    def forward(self, source_image, kp_driving, kp_source, bg_param=None, dropout_flag=False, dropout_p=0):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)  # bs 51 64 64
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)
        deformed_source = self.create_deformed_source_image(source_image, transformations)
        out_dict['deformed_source'] = deformed_source
        # out_dict['transformations'] = transformations
        deformed_source = deformed_source.view(bs, -1, h, w)  # bs 33 64 64
        input = torch.cat([heatmap_representation, deformed_source], dim=1)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input, mode=1)
        contribution_maps = self.maps(prediction[-1])
        if (dropout_flag):
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict['contribution_maps'] = contribution_maps

        # Combine the K+1 transformations
        # Eq(6) in the paper
        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)  # bs 10 2 64 64
        deformation = (transformations * contribution_maps).sum(dim=1)  # bs 2 64 64
        id_grid = coords_grid(bs, 64, 64, deformation.device)
        init_flow = (64 - 1) * (deformation + 1) / 2.0 - id_grid
        multi_flow = []
        sizes = [32, 64, 128, 256]
        # bs 148 64 64
        source_feature = self.refine_proj_query_feat(prediction[-1])
        for i, size in enumerate(sizes):
            refine_feat = F.interpolate(source_feature, size, mode='bilinear', align_corners=True)
            refine_feat = self.apply_dropout(self.dropout2d, refine_feat)[0]
            flow_up = F.interpolate(init_flow, size, mode='bilinear', align_corners=True) * size / 64
            x_t = torch.cat([flow_up, refine_feat], dim=1)  # bs 94+2 256 256
            x_t = self.window_partition(x_t, 32, 32, i)
            x_8 = self.refine_swin_decoder_2_1[i](x_t)
            x_4 = self.refine_swin_decoder_1_1[i](x_t)
            x_t = x_8 + x_4
            x_8 = self.refine_swin_decoder_2_2[i](x_t)
            x_4 = self.refine_swin_decoder_1_2[i](x_t)
            x = x_8 + x_4  # bs 96 32 32
            x = x.permute(0, 2, 3, 1)
            x = self.window_reverse(x, 32, 32, size, size, 96, i)
            flow_refine = self.refine2[i](x)
            curr_flow = flow_up + flow_refine
            # Normalize
            id_grid = coords_grid(bs, size, size, deformation.device)
            flow = (curr_flow + id_grid).permute(0, 2, 3, 1)
            flow = self.normalize_flow(flow)
            multi_flow.append(flow)
        deformation = deformation.permute(0, 2, 3, 1)
        out_dict['deformation_init'] = deformation  # Optical Flow
        out_dict['deformation'] = multi_flow

        occlusion_map = []
        if self.multi_mask:
            for i in range(self.occlusion_num - self.up_nums):
                occlusion_map.append(
                    torch.sigmoid(self.occlusion[i](prediction[self.up_nums - self.occlusion_num + i])))
            prediction = prediction[-1]
            for i in range(self.up_nums):
                prediction = self.up[i](prediction)
                occlusion_map.append(torch.sigmoid(self.occlusion[i + self.occlusion_num - self.up_nums](prediction)))
        else:
            occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))
        # 32*32 64*64 128*128 256*256

        out_dict['occlusion_map'] = occlusion_map  # Multi-resolution Occlusion Masks
        return out_dict


if __name__ == '__main__':
    block = DenseMotionNetwork(block_expansion=64, num_blocks=5, max_features=1024, num_tps=10, num_channels=3,
                               scale_factor=0.25, kp_variance=0.01)
    image = torch.randn(1, 3, 256, 256)
    kp_1 = torch.randn(1, 50, 2)
    kp_2 = torch.randn(1, 50, 2)
    kp_source = {'fg_kp': kp_1}
    kp_driving = {'fg_kp': kp_2}
    output = block(image, kp_driving, kp_source)
