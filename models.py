from __future__ import division

from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict
import os, sys

from flow_networks.resample2d_package.resample2d import Resample2d
from flow_networks.channelnorm_package.channelnorm import ChannelNorm
from flow_networks import FlowNetC
from flow_networks import FlowNetS
from flow_networks import FlowNetSD
from flow_networks import FlowNetFusion
from flow_networks.submodules import *
import utils

from utils.parse_config import *
from utils.utils import build_targets

#-----------------------------------FlowNet---------------------------------

class FlowNet2(nn.Module):
    def __init__(self, args, batchNorm=False, div_flow=20.):
        super(FlowNet2, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=False)

        if args.fp16:
            self.resample1 = nn.Sequential(
                tofp32(),
                Resample2d(),
                tofp16())
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=False)
        if args.fp16:
            self.resample2 = nn.Sequential(
                tofp32(),
                Resample2d(),
                tofp16())
        else:
            self.resample2 = Resample2d()

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD.FlowNetSD(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')

        if args.fp16:
            self.resample3 = nn.Sequential(
                tofp32(),
                Resample2d(),
                tofp16())
        else:
            self.resample3 = Resample2d()

        if args.fp16:
            self.resample4 = nn.Sequential(
                tofp32(),
                Resample2d(),
                tofp16())
        else:
            self.resample4 = Resample2d()

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion.FlowNetFusion(args, batchNorm=self.batchNorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        heigh, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.)
        for i in range(min_dim):
            weight.data[i, i, :, :] = torch.from_numpy(bilinear)
        return

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)

        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        # if not diff_flownets2_flow.volatile:
        #     diff_flownets2_flow.register_hook(save_grad(self.args.grads, 'diff_flownets2_flow'))

        diff_flownets2_img1 = self.channelnorm((x[:, :3, :, :] - diff_flownets2_flow))
        # if not diff_flownets2_img1.volatile:
        #     diff_flownets2_img1.register_hook(save_grad(self.args.grads, 'diff_flownets2_img1'))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)

        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        # if not diff_flownetsd_flow.volatile:
        #     diff_flownetsd_flow.register_hook(save_grad(self.args.grads, 'diff_flownetsd_flow'))

        diff_flownetsd_img1 = self.channelnorm((x[:, :3, :, :] - diff_flownetsd_flow))
        # if not diff_flownetsd_img1.volatile:
        #     diff_flownetsd_img1.register_hook(save_grad(self.args.grads, 'diff_flownetsd_img1'))

        # concat img1 flownetsd, flownets2, norm_flownetsd, norm_flownets2, diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow,
                             diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)

        # if not flownetfusion_flow.volatile:
        #     flownetfusion_flow.register_hook(save_grad(self.args.grads, 'flownetfusion_flow'))

        return flownetfusion_flow


class FlowNet2C(FlowNetC.FlowNetC):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2C, self).__init__(args, batchNorm=batchNorm, div_flow=div_flow)
        self.rgb_max = args.rgb_max

    def forward(self, inputs):
        # input:[sam_idx_inbatch,channel_idx,2,row_idx,col_idx]
        # rgb_mean球了样本的，三个通道（r,g,b）的的平均值
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        # x1:[sam_idx_inbatch, col_idx,channel_idx，row_idx], represent all first frames in every samples in batch
        # x2:[sam_idx_inbatch, col_idx,channel_idx，row_idx], represent all second frames in every samples in batch

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)

        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)

        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return self.upsample1(flow2 * self.div_flow)


class FlowNet2S(FlowNetS.FlowNetS):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2S, self).__init__(args, input_channels=6, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return self.upsample1(flow2 * self.div_flow)


class FlowNet2SD(FlowNetSD.FlowNetSD):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2SD, self).__init__(args, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)

        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return self.upsample1(flow2 * self.div_flow)


class FlowNet2CS(nn.Module):
    def __init__(self, args, batchNorm=False, div_flow=20.):
        super(FlowNet2CS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=False)

        # if args.fp16:
        #     self.resample1 = nn.Sequential(
        #                     tofp32(),
        #                     Resample2d(),
        #                     tofp16())
        # else:
        self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):

        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        return flownets1_flow


class FlowNet2CSS(nn.Module):
    def __init__(self, args, batchNorm=False, div_flow=20.):
        super(FlowNet2CSS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if args.fp16:
            self.resample1 = nn.Sequential(
                tofp32(),
                Resample2d(),
                tofp16())
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(
                tofp32(),
                Resample2d(),
                tofp16())
        else:
            self.resample2 = Resample2d()

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)

        # flownetc
        # stride: 4 for origin resolution
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)

        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)

        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        #  origin resolution output
        return flownets2_flow


#------------------------------------YOLO--------------------------------------

def create_modules(module_defs, data_num_classes):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    data_num_classes for check
    """

    # first dict save hyperparams
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):

        modules = nn.Sequential()

        # conv module
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.GroupNorm(16,filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

            module_list.append(modules)

        # maxpool module
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)
            module_list.append(modules)

        # updample module for FPN
        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)
            module_list.append(modules)

        # route feature map up-side down
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())
            module_list.append(modules)

        # shortcut module for ResBlock
        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())
            module_list.append(modules)

        # other parameter for yolo anchor
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            assert(data_num_classes == num_classes)
            img_size = int(hyperparams["size"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_%d" % i, yolo_layer)
            module_list.append(modules)
        # Register module list and number of output filters
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_size = img_size
        self.ignore_thres = 0.5
        #self.lambda_coord = 1
        #self.cls_predictor = torch.nn.Softmax(dim = 4)

        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='mean')  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

        self.sm = nn.Softmax(dim=4)

    def forward(self, x, targets=None):
        # print("input shape of yolo_layer:{}".format(x.shape))
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_size / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x 0-1, according grid
        y = torch.sigmoid(prediction[..., 1])  # Center y 0-1,according grid
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf 0-1
        #pred_cls = self.cls_predictor(prediction[..., 5:])  # Cls pred.
        #pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        pred_cls = prediction[..., 5:]  # Cls pred.


        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)

        # 把每个box的predict加上grid_offset,也就是说原始的predict是对于自己的grid的左上角的offset
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        # w和h是基于原始anchor做stride以后在当前map上的感受野大小的pixel数量单位
        # pred_boxes中的w和h是对于resized以后的图的boax的w和h的大小，没有scaled
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            # mask is the mask of gt bbox
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
            )
            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals) if nProposals != 0 else 0

            # Handle masks
            # mask是对应有object的prediction box（对于每个gt object的iou最大的prediction），
            mask = Variable(mask.type(ByteTensor))
            # conf_mask 是计算confidence loss的anchor mask
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # conf_mask_true 是 confidence gt 为1 的mask
            conf_mask_true = mask
            # conf_mask_false 是 confidence gt 为0 的mask
            conf_mask_false = conf_mask - mask

            """
            YOLOv3 predicts an objectness score for each bounding box using logistic regression. 
            YOLOv3 changes the way in calculating the cost function. 
            If the bounding box prior (anchor) overlaps a ground truth object more than others, 
            the corresponding objectness score should be 1. 
            For other priors with overlap greater than a predefined threshold (default 0.5), 
            they incur no cost. 
            Each ground truth object is associated with one boundary box prior only. 
            If a bounding box prior is not assigned, it incurs no classification and localization lost, 
            just confidence loss on objectness.
            """
            vaild_mask = torch.max(mask).item() # check image has at least one object
            if vaild_mask==1:
                loss_x = self.mse_loss(x[mask], tx[mask])
                loss_y = self.mse_loss(y[mask], ty[mask])
                loss_w = self.mse_loss(w[mask], tw[mask])
                loss_h = self.mse_loss(h[mask], th[mask])
                loss_conf = self.bce_loss(pred_conf[conf_mask_false],
                                          tconf[conf_mask_false])*3 \
                            + self.bce_loss(pred_conf[conf_mask_true],
                                            tconf[conf_mask_true])

                loss_cls = self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask],1))

                # print(loss_cls)
                # tep_mask = torch.ByteTensor(pred_cls[mask].shape).fill_(0)
                # for i,idx_max in enumerate(torch.argmax(tcls[mask],1)):
                #     tep_mask[i][idx_max] = 1
                # print(torch.masked_select(pred_cls[mask],tep_mask))

            # for frame has no object.
            else:
                loss_x = torch.tensor(0)
                loss_y = torch.tensor(0)
                loss_w = torch.tensor(0)
                loss_h = torch.tensor(0)
                loss_conf = self.bce_loss(pred_conf[conf_mask], tconf[conf_mask])*4
                loss_cls = torch.tensor(0)

            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        # predict / val
        else:
            # If not in training phase return predictions

            pred_cls = self.sm(pred_cls)
            #print("test predict max:{}".format(pred_cls.max()))
            output = torch.cat(
                (
                    # boxes (x y w h) is unscaled coordinate of resized image
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, args):
        super(Darknet, self).__init__()
        # list of dictionary of model config
        self.module_defs = parse_model_config(args.yolo_config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs,args.data_num_classes)
        self.down_channel_62= torch.nn.Conv2d(1024, 512, 1)
        self.down_channel_37 = torch.nn.Conv2d(512, 256, 1)
        self.down_channel_12 = torch.nn.Conv2d(256, 128, 1)
        self.img_size = int(self.hyperparams["size"])
        self.loss_names = ["loss","x", "y", "w", "h", "conf", "cls", "recall", "precision"]
        self.flow_warp = Resample2d()


    def forward(self,x:torch.Tensor, forward_feats:list ,flow:torch.Tensor ,targets:torch.Tensor=None):
        # flow dim is [h,w,channel]
        # target [50,5()]
        is_training = targets is not None
        output = []
        losses = defaultdict(float)
        layer_outputs = []
        # list of deques for each sample
        output_features = [[] for _ in range(x.shape[0])]
        div = {11:4, 36: 8, 61: 16}
        div_idx = {11:0, 36: 1, 61: 2}
        x = x/255.
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)

            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)

            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]

                #warp and aggregate(cat) at layer 12 37 62, which are the layers before down-sampling
                if i in [11,36,61]:

                    # append feature in each list pre sample
                    for idx in range(x.shape[0]):
                        output_features[idx].append(x[idx])

                    # flow aggregate in  L62/37/12
                    if forward_feats is not None:
                        assert(flow is not None)
                        # resizing flow by bi-linear interpolation
                        _flow = F.interpolate(flow,size=(x.shape[-2],x.shape[-1]),mode="bilinear",align_corners=False)/div[i]
                        _flow = _flow.contiguous()
                        f = torch.stack([l[div_idx[i]] for l in forward_feats])
                        _re = self.flow_warp(f,_flow)
                        x = torch.cat([x,_re],1)
                        if i == 36:
                            x = self.down_channel_37(x)
                        elif i == 11:
                            x = self.down_channel_12(x)
                        else:
                            x = self.down_channel_62(x)
                        #x =  0.7*x + 0.3 *_re


            elif module_def["type"] == "yolo":
                # Train phase: get loss
                # x = x.cpu()
                if is_training:
                    # x, *losses = module[0](x, targets)
                    x = module[0](x,targets)
                    for name, loss in zip(self.loss_names, x):
                        losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        #losses["recall"] /= 3
        #losses["precision"] /= 3
        losses["recall"] /= 2
        losses["precision"] /= 2

        return (losses,output_features) if is_training else (torch.cat(output, 1), output_features)


    # notice: save weight as DataParallel, so the new weight should load by "load_weight()" after "set_multi_gpus()"
    def save_weights(self, path):
        torch.save(self.state_dict(),os.path.join(path,"yolo_f.pth"))


    def load_weights(self, weights_path = None):
        print(weights_path)
        if weights_path is not None and os.path.isfile(weights_path):
            self.load_state_dict(torch.load(weights_path))
        else:
            print('Weight file is not given or not exits, random initialize')
            self.apply(utils.utils.weights_init_normal)


    def load_fit_weights(self, weights_path):
        print(weights_path)
        # random init all weights
        self.apply(utils.utils.weights_init_normal)
        model_dict = self.state_dict()

        # get pre-trained weight
        if weights_path and os.path.isfile(weights_path):
            pretrained_dict = torch.load(weights_path)
        else:
            print('YOLO weight file is not given or not exits, random initialize')
            self.apply(utils.utils.weights_init_normal)
            return

        # print("pretrained layers:")
        # for k,v in pretrained_dict.items():
        #     print("{}:{}".format(k,v.shape))
        # print()
        # print("model layers:")
        # for k, v in model_dict.items():
        #     print("{}:{}".format(k, v.shape))
        # print()


        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

        init_layer = set(model_dict.keys())-set(pretrained_dict.keys())
        if len(init_layer)!= 0:
            print("initialized Yolo model layers:")
            for k in set(model_dict.keys())-set(pretrained_dict.keys()):
                print(k)



    def set_multi_gpus(self,gpu_id_list):
        # use multiple gpu except yoloLayer
        parall_model_list = nn.ModuleList()
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "yolo":
                module = module.cuda()
            else:
                module = nn.parallel.DataParallel(module,device_ids=gpu_id_list).cuda()
            parall_model_list.append(module)
        self.module_list = parall_model_list
        self.down_channel_62 = nn.parallel.DataParallel(self.down_channel_62,gpu_id_list).cuda()
        self.down_channel_37 = nn.parallel.DataParallel(self.down_channel_37,gpu_id_list).cuda()
        self.down_channel_12 = nn.parallel.DataParallel(self.down_channel_12,gpu_id_list).cuda()
        self.flow_warp = nn.parallel.DataParallel(self.flow_warp,gpu_id_list)
        return

#------------------------------------FLOW-YOLO---------------------------------

class FlowYOLO(nn.Module):
    def __init__(self, args):
        super(FlowYOLO, self).__init__()
        self.flow_model = args.flow_model_class(args)
        self.detect_model = args.yolo_model_class(args)
        self.args = args



    def forward(self, flow_input, data, last_feature:list, target=None):

        # data is a torch Tensor [b,3,h,w]
        # flow_input is a torch Tensor [b,6,h,w]
        # last_feature is list of list

        flows_output = self.flow_model(flow_input) if flow_input is not None else None

        result, features = self.detect_model(data,
                                            forward_feats=last_feature,
                                            flow=flows_output,
                                            targets =target)
        if self.args.use_cuda and torch.torch.cuda.is_available() and isinstance(result['loss'],torch.Tensor):
            #print("put result in cuda.")
            result['loss'] = result['loss'].cuda()
        return result, features


    def load_weights(self, flow_weights_path=None, yolo_weights_path=None):

        print("flow_weigth: {}".format(flow_weights_path))
        print("yolo_weights: {}".format(yolo_weights_path))

        # load flownet weight
        if flow_weights_path and os.path.isfile(flow_weights_path):
            self.flow_model.load_state_dict(torch.load(flow_weights_path))
        elif flow_weights_path:
            print(sys.stderr,"Error: flowNet no checkpoint finded.")
            exit(1)
        else:
            print("FlowNet weight not given, random initialization.")
            print("Open update of flowNet as the weight is random initialized.")
            for p in self.parameters():
                p.requires_grad = True

        # load yolo weight
        self.detect_model.load_fit_weights(weights_path=yolo_weights_path)


    def save_weights(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

        # no use format of weight of origin FlowNet, use torch format
        torch.save(self.flow_model.state_dict(),os.path.join(path,"flow.pth"))
        self.detect_model.save_weights(path)
        torch.save(self,path.join(path,"EntireModel.pth"))


    def set_multi_gpus(self,gpu_id_list):
        self.flow_model = nn.parallel.DataParallel(self.flow_model,device_ids=gpu_id_list).cuda()
        self.detect_model.set_multi_gpus(gpu_id_list)




