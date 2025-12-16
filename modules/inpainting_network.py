import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, FusionBlock, Get_gradient_nopadding, FuseBlock
from modules.dense_motion import DenseMotionNetwork


class DepthAwareAttention(nn.Module):
    """ depth-aware attention Layer"""

    def __init__(self, in_dim, activation):
        super(DepthAwareAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, source, feat):
        """
            inputs :
                source : input feature maps( B X C X W X H) 256,64,64
                driving : input feature maps( B X C X W X H) 256,64,64
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = source.size()
        proj_query = self.activation(self.query_conv(source)).view(m_batchsize, -1, width * height).permute(0, 2,
                                                                                                            1)  # B X CX(N) [bz,32,64*64]
        proj_key = self.activation(self.key_conv(feat)).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + feat

        return out, attention


class InpaintingNetwork(nn.Module):
    """
    Inpaint the missing regions and reconstruct the Driving image.
    """

    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, multi_mask=True, **kwargs):
        super(InpaintingNetwork, self).__init__()

        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        self.first_g = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        down_blocks_g = []
        up_blocks = []
        up_blocks_g = []
        resblock = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            down_blocks_g.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            decoder_in_feature = out_features * 2
            if i == num_down_blocks - 1:
                decoder_in_feature = out_features
            up_blocks.append(UpBlock2d(decoder_in_feature, in_features, kernel_size=(3, 3), padding=(1, 1)))
            up_blocks_g.append(UpBlock2d(out_features, in_features, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(6):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.down_blocks_g = nn.ModuleList(down_blocks_g)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.up_blocks_g = nn.ModuleList(up_blocks_g)[::-1]
        self.resblock = nn.ModuleList(resblock[::-1])
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.final_g = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        self.get_g_nopadding = Get_gradient_nopadding()
        self.AttnModule = DepthAwareAttention(512, nn.ReLU())

    def occlude_input(self, inp, occlusion_map):
        if not self.multi_mask:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear', align_corners=True)
        out = inp * occlusion_map
        return out

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def forward(self, source_image, dense_motion):
        out = self.first(source_image)
        encoder_map = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)

        out_grad = self.get_g_nopadding(source_image)
        out_grad = self.first_g(out_grad)
        for i in range(len(self.down_blocks)):
            out_grad = self.down_blocks_g[i](out_grad)

        output_dict = {}
        output_dict['contribution_maps'] = dense_motion['contribution_maps']
        output_dict['deformed_source'] = dense_motion['deformed_source']
        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map
        deformation = dense_motion['deformation']
        deformation_init = dense_motion['deformation_init']
        out_ij = self.deform_input(out.detach(), deformation[0])
        out = self.deform_input(out, deformation[0])
        out_ij_init = self.deform_input(out.detach(), deformation_init)

        out_ij = self.occlude_input(out_ij, occlusion_map[0].detach())
        out = self.occlude_input(out, occlusion_map[0])
        out_ij_init = self.occlude_input(out_ij_init, occlusion_map[0].detach())

        warped_encoder_maps = []
        warped_encoder_maps.append(out_ij)
        warped_encoder_maps_init = []
        warped_encoder_maps_init.append(out_ij_init)

        out_grad = self.deform_input(out_grad, deformation_init)
        out_grad = self.occlude_input(out_grad, occlusion_map[0].detach())
        grad_feature = out_grad

        # grad branch part
        out_grad = self.bottleneck(out_grad)
        for i in range(self.num_down_blocks):
            out_grad = self.up_blocks_g[i](out_grad)
        out_grad = self.final_g(out_grad)
        out_grad = torch.sigmoid(out_grad)
        output_dict["prediction_grad"] = out_grad

        out, attention = self.AttnModule(grad_feature, out)

        for i in range(self.num_down_blocks):
            out = self.resblock[2 * i](out)
            out = self.resblock[2 * i + 1](out)
            out = self.up_blocks[i](out)
            flow_index = i + 1
            encode_i = encoder_map[-(i + 2)]
            encode_ij = self.deform_input(encode_i.detach(), deformation[flow_index])
            encode_i = self.deform_input(encode_i, deformation[flow_index])
            encode_ij_init = self.deform_input(encode_i.detach(), deformation_init)

            occlusion_ind = 0
            if self.multi_mask:
                occlusion_ind = i + 1
            encode_ij = self.occlude_input(encode_ij, occlusion_map[occlusion_ind].detach())
            encode_i = self.occlude_input(encode_i, occlusion_map[occlusion_ind])
            encode_ij_init = self.occlude_input(encode_ij_init, occlusion_map[occlusion_ind].detach())
            warped_encoder_maps.append(encode_ij)
            warped_encoder_maps_init.append(encode_ij_init)

            if i == self.num_down_blocks - 1:
                break

            out = torch.cat([out, encode_i], 1)

        deformed_source = self.deform_input(source_image, deformation[-1])
        output_dict["deformed"] = deformed_source
        output_dict["warped_encoder_maps"] = warped_encoder_maps
        output_dict["warped_encoder_maps_init"] = warped_encoder_maps_init

        occlusion_last = occlusion_map[-1]
        if not self.multi_mask:
            occlusion_last = F.interpolate(occlusion_last, size=out.shape[2:], mode='bilinear', align_corners=True)
        out = out * (1 - occlusion_last) + encode_i
        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out

        return output_dict

    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(self.occlude_input(out.detach(), occlusion_map[-1].detach()))
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            out_mask = self.occlude_input(out.detach(), occlusion_map[2 - i].detach())
            encoder_map.append(out_mask.detach())

        return encoder_map
