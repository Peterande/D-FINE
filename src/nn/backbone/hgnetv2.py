import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SelectAdaptivePool2d, DropPath, create_conv2d
import torch.nn.init as init 
from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d
from ...core import register
from ...zoo.dfine.dbb import conv_bn, transI_fusebn, transII_addbranch, transIII_1x1_kxk, \
                             transV_avg, transVI_multiscale, IdentityBasedConv1x1, BNAndPadLayer
from fairscale.nn.checkpoint import checkpoint_wrapper
# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2']

    
class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ACConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            deploy=False,
            rep=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.rep = rep and kernel_size >= 3
        self.bn = nn.BatchNorm2d(out_chs)

        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()
            
        self.deploy = deploy

        if deploy:
            self.fused_conv = create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        elif self.rep:
            if self.conv.padding[0] - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [self.conv.padding[0] - kernel_size // 2, self.conv.padding[0]]
                ver_padding = [self.conv.padding[0], self.conv.padding[0] - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - self.conv.padding[0]
                hor_padding = [0, self.conv.padding[0]]
                ver_padding = [self.conv.padding[0], 0]

            self.ver_conv = create_conv2d(
                in_chs,
                out_chs,
                (kernel_size, 1),
                stride=stride,
                padding=ver_padding,
                groups=groups,
            )

            self.hor_conv = create_conv2d(
                in_chs,
                out_chs,
                (1, kernel_size),
                stride=stride,
                padding=hor_padding,
                groups=groups,
            )
            self.ver_bn = nn.BatchNorm2d(num_features=out_chs)
            self.hor_bn = nn.BatchNorm2d(num_features=out_chs)
            self.single_init()

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.conv, self.bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b

    def convert_to_deploy(self):
        if self.rep:
            deploy_k, deploy_b = self.get_equivalent_kernel_bias()
            self.deploy = True
            self.fused_conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                                        kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                                        padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True,
                                        padding_mode=self.conv.padding_mode)
            self.__delattr__('conv')
            self.__delattr__('bn')
            self.__delattr__('hor_conv')
            self.__delattr__('hor_bn')
            self.__delattr__('ver_conv')
            self.__delattr__('ver_bn')
            self.fused_conv.weight.data = deploy_k
            self.fused_conv.bias.data = deploy_b


    def init_gamma(self, gamma_value):
        init.constant_(self.bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)

    def single_init(self):
        init.constant_(self.bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)

    def forward(self, input):
        if not self.rep:
            return self.lab(self.act(self.bn(self.conv(input))))
        if self.deploy:
            return self.lab(self.act(self.fused_conv(input)))
        else:
            square_outputs = self.conv(input)
            square_outputs = self.bn(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return self.lab(self.act(result))
        
 
class DBBConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            deploy=False,
            rep=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.rep = rep and kernel_size >= 3
        self.bn = nn.BatchNorm2d(out_chs)

        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

        self.kernel_size = kernel_size
        self.out_channels = out_chs
        self.groups = groups
        padding = (kernel_size-1)//2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=1, groups=groups, bias=True)
        elif self.rep:

            internal_channels_1x1_3x3 = in_chs if groups < out_chs else 2 * in_chs   # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            self.dbb_1x1_kxk2 = nn.Sequential()
            if internal_channels_1x1_3x3 == in_chs:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_chs, groups=groups))
                self.dbb_1x1_kxk2.add_module('idconv1', IdentityBasedConv1x1(channels=in_chs, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_chs, out_channels=internal_channels_1x1_3x3,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
                self.dbb_1x1_kxk2.add_module('conv1', nn.Conv2d(in_channels=in_chs, out_channels=internal_channels_1x1_3x3,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            
            self.dbb_1x1_kxk.add_module('bn_rep1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk2.add_module('bn_rep1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_chs,
                                                            kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk2.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_chs,
                                                            kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn_rep2', nn.BatchNorm2d(out_chs))
            self.dbb_1x1_kxk2.add_module('bn_rep2', nn.BatchNorm2d(out_chs))

        self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.conv.weight, self.bn)


        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
            k_1x1_kxk2_first = self.dbb_1x1_kxk2.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
            k_1x1_kxk2_first = self.dbb_1x1_kxk2.conv1.weight
            
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn_rep1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn_rep2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)
        
        k_1x1_kxk2_first, b_1x1_kxk2_first = transI_fusebn(k_1x1_kxk2_first, self.dbb_1x1_kxk2.bn_rep1)
        k_1x1_kxk2_second, b_1x1_kxk2_second = transI_fusebn(self.dbb_1x1_kxk2.conv2.weight, self.dbb_1x1_kxk2.bn_rep2)
        k_1x1_kxk2_merged, b_1x1_kxk2_merged = transIII_1x1_kxk(k_1x1_kxk2_first, b_1x1_kxk2_first, k_1x1_kxk2_second, b_1x1_kxk2_second, groups=self.groups)

        return transII_addbranch((k_origin, k_1x1_kxk_merged, k_1x1_kxk2_merged), (b_origin, b_1x1_kxk_merged, b_1x1_kxk2_merged))

    def convert_to_deploy(self):
        if hasattr(self, 'dbb_reparam') or not self.rep:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                                     kernel_size=self.conv.kernel_size, stride=self.conv.stride,
                                     padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.__delattr__('dbb_1x1_kxk')
        self.__delattr__('dbb_1x1_kxk2')

    def forward(self, inputs):
        if not self.rep:
            return self.lab(self.act(self.bn(self.conv(inputs))))

        if hasattr(self, 'dbb_reparam'):
            return self.lab(self.act(self.dbb_reparam(inputs)))

        out = self.bn(self.conv(inputs))
        out += self.dbb_1x1_kxk(inputs)
        out += self.dbb_1x1_kxk2(inputs)
        return self.lab(self.act(out))

    def init_gamma(self, gamma_value):
        if hasattr(self, "bn"):
            torch.nn.init.constant_(self.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn_rep2.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk2"):
            torch.nn.init.constant_(self.dbb_1x1_kxk2.bn_rep2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "bn"):
            torch.nn.init.constant_(self.bn.weight, 1.0)
            
                   
class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False,
            rep=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        self.conv = create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
            rep=False
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
            rep=rep
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)
    

class HG_Block(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                        rep=False
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                        rep=False
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
                rep=False
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



@register()
class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            }
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            }
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            }
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            }
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            }
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            }
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            }
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab)


        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            prefix = 'weight/hgnetv2/PPHGNetV2_'
            stage1 = "stage1_"
            # stage1 = ""
            if stage1 == "":
                print("Loading stage2 HGNetV2")
            else:
                print("Loading stage1 HGNetV2")
            
            if name == 'B0':
                state = torch.load(prefix + 'B0_ssld_' + stage1 + 'pretrained.pth')
            elif name == 'B1':
                state = torch.load(prefix + 'B1_ssld_' + stage1 + 'pretrained.pth')
            elif name == 'B2':
                state = torch.load(prefix + 'B2_ssld_' + stage1 + 'pretrained.pth')
            elif name == 'B3':
                state = torch.load(prefix + 'B3_ssld_' + stage1 + 'pretrained.pth')
            elif name == 'B4':
                state = torch.load(prefix + 'B4_ssld_' + stage1 + 'pretrained.pth')
            elif name == 'B5':
                state = torch.load(prefix + 'B5_ssld_' + stage1 + 'pretrained.pth')
            elif name == 'B6':
                state = torch.load(prefix + 'B6_ssld_' + stage1 + 'pretrained.pth')
            else:
                raise ValueError(f'Unknown PPHGNetV2 version: {name}')
            
            self.load_state_dict(state, strict=False)
            print(f'Load HGNetV2 state_dict')


    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                if name not in ['ver_bn', 'hor_bn']:
                    _child = self._freeze_norm(child)
                else:
                    _child = child
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.named_parameters():
            if p[0] not in ['ver_conv', 'hor_conv', 'ver_bn', 'hor_bn']:
                p[1].requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs