from networks.network_utils import InvConv2d, InvConv2dLU, HighFreqExtractor, initialize_weights, initialize_weights_xavier
import torch
from torch import nn

class SimpleBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=64):
        super(SimpleBlock, self).__init__()
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(channel_in, gc, kernel_size=3, padding=1))
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv2d(gc, gc, kernel_size=1, padding=0))
        self.conv3 = torch.nn.utils.spectral_norm(nn.Conv2d(gc, channel_out, kernel_size=3, padding=1))

        self.conv1.weight.data.normal_(0, 0.02)
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.normal_(0, 0.02)
        self.conv2.bias.data.zero_()
        self.conv3.weight.data.normal_(0, 0.02)
        self.conv3.bias.data.zero_()

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=64, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu((self.conv1(x)))
        x2 = self.lrelu((self.conv2(torch.cat((x, x1), 1))))
        x3 = self.lrelu((self.conv3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu((self.conv4(torch.cat((x, x1, x2, x3), 1))))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


class ResNetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=64, bias=True):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
        self.conv6 = nn.Conv2d(gc, channel_out, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        initialize_weights(self.conv6, 0)

    def forward(self, x):
        x1 = self.lrelu((self.conv1(x)))
        x2 = self.lrelu(torch.add(x1, self.conv3(self.lrelu((self.conv2(x1))))))
        x3 = self.lrelu(torch.add(x2, self.conv5(self.lrelu((self.conv4(x2))))))
        x4 = self.conv6(x3)

        return x4

class StableAdditiveCoupling(nn.Module):
    def __init__(self, in_channel, block_type='simple', filter_size=64):
        super(StableAdditiveCoupling, self).__init__()

        if block_type == 'simple':
            self.net1 = SimpleBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net2 = SimpleBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net3 = SimpleBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net4 = SimpleBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)

        elif block_type == 'dense':
            self.net1 = DenseBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net2 = DenseBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net3 = DenseBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net4 = DenseBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)

        elif block_type == 'residual':
            self.net1 = ResNetBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net2 = ResNetBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net3 = ResNetBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)
            self.net4 = ResNetBlock(in_channel - in_channel // 4, in_channel // 4, gc=filter_size)

    def forward(self, input):
        in_a, in_b, in_c, in_d = input.chunk(4, 1)

        out_a = in_a - self.net1(torch.cat((in_b, in_c, in_d), 1))
        out_b = in_b - self.net2(torch.cat((out_a, in_c, in_d), 1))
        out_c = in_c - self.net3(torch.cat((out_a, out_b, in_d), 1))
        out_d = in_d - self.net4(torch.cat((out_a, out_b, out_c), 1))

        return torch.cat([out_a, out_b, out_c, out_d], 1)

    def inverse(self, output):
        out_a, out_b, out_c, out_d = output.chunk(4, 1)

        in_d = out_d + self.net4(torch.cat((out_a, out_b, out_c), 1))
        in_c = out_c + self.net3(torch.cat((out_a, out_b, in_d), 1))
        in_b = out_b + self.net2(torch.cat((out_a, in_c, in_d), 1))
        in_a = out_a + self.net1(torch.cat((in_b, in_c, in_d), 1))

        return torch.cat([in_a, in_b, in_c, in_d], 1)


class InvertibleBlock(nn.Module):
    def __init__(self, in_channel, conv_lu=True, block_type='simple'):
        super(InvertibleBlock, self).__init__()
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
        self.coupling = StableAdditiveCoupling(in_channel, block_type=block_type)

    def forward(self, input):
        output = self.invconv(input)
        output = self.coupling(output)
        return output

    def reverse(self, output):
        input = self.coupling.inverse(output)
        input = self.invconv.reverse(input)
        return input

class InvertibleGenerator(nn.Module):
    def __init__(self, in_channel, n_block, squeeze_num, conv_lu=True, block_type='simple'):
        super(InvertibleGenerator, self).__init__()

        self.squeeze_num = squeeze_num
        squeeze_dim = in_channel * squeeze_num**2

        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(InvertibleBlock(squeeze_dim, conv_lu=conv_lu, block_type=block_type))

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        input_high = HighFreqExtractor(input, 6)
        squeezed = input_high.view(b_size, n_channel, height // self.squeeze_num, self.squeeze_num,
                                   width // self.squeeze_num, self.squeeze_num)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * self.squeeze_num**2,
                                         height // self.squeeze_num, width // self.squeeze_num)

        for block in self.blocks:
            out = block(out)

        unsqueezed = out.view(b_size, n_channel, self.squeeze_num, self.squeeze_num, height // self.squeeze_num, width // self.squeeze_num)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed_out_high = unsqueezed.contiguous().view(
            b_size, n_channel, height, width
        )
        unsqueezed_out = input - (input_high - unsqueezed_out_high)

        return unsqueezed_out

    def inverse(self, input):
        b_size, n_channel, height, width = input.shape
        input_high = HighFreqExtractor(input, 6)
        squeezed = input_high.view(b_size, n_channel, height // self.squeeze_num, self.squeeze_num,
                              width // self.squeeze_num, self.squeeze_num)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * self.squeeze_num ** 2, height // self.squeeze_num,
                                         width // self.squeeze_num)

        for block in self.blocks[::-1]:
            out = block.reverse(out)

        unsqueezed = out.view(b_size, n_channel, self.squeeze_num, self.squeeze_num, height // self.squeeze_num,
                              width // self.squeeze_num)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed_out_high = unsqueezed.contiguous().view(
            b_size, n_channel, height, width
        )
        unsqueezed_out = input - (input_high - unsqueezed_out_high)

        return unsqueezed_out