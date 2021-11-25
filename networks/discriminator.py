from torch import nn

class Discriminator_patch(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_64 = nn.Conv2d(1,64,4,2,padding=1, bias=False)
        self.conv1_64.weight.data.normal_(0, 0.02)

        self.conv64_128 = nn.Conv2d(64,128,4,2,padding=1)
        self.conv64_128.weight.data.normal_(0, 0.02)
        self.conv64_128.bias.data.zero_()

        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm128.weight.data.normal_(1.0, 0.02)
        self.batchnorm128.bias.data.zero_()

        self.conv128_256 = nn.Conv2d(128, 256, 4, 1, padding=1)
        self.conv128_256.weight.data.normal_(0, 0.02)
        self.conv128_256.bias.data.zero_()

        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm256.weight.data.normal_(1.0, 0.02)
        self.batchnorm256.bias.data.zero_()

        self.conv256_1 = nn.Conv2d(256,1,4,1,padding=1)
        self.conv256_1.weight.data.normal_(0, 0.02)
        self.conv256_1.bias.data.zero_()

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input):

        x = self.conv1_64(input)
        x = self.leakyrelu(x)

        x = self.conv64_128(x)
        x = self.batchnorm128(x)
        x = self.leakyrelu(x)

        x = self.conv128_256(x)
        x = self.batchnorm256(x)
        x = self.leakyrelu(x)

        x = self.conv256_1(x)

        return x