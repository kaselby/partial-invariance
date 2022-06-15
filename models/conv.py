import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.net = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride),
            nn.BatchNorm2d(out_filters),
            nn.ReLU()
        )

    def size_transform(self, input_size):
        return int(math.floor((input_size - self.kernel_size)/self.stride))
    
    def forward(self, inputs):
        return self.net(inputs)

class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, n_conv=2, pool='max'):
        super().__init__()
        self.n_conv = n_conv
        self.in_filters = in_filters
        self.out_filters = out_filters
        layers = [ConvLayer(in_filters, out_filters, 3)]
        for i in range(n_conv-1):
            layers.append(ConvLayer(out_filters, out_filters, 3))

        if pool == 'max':
            layers.append(nn.MaxPool2d(2,2))
        elif pool == 'avg':
            layers.append(nn.AvgPool2d(2,2))
        else:
            pool = 'none'
        self.pool = pool
        self.net = nn.Sequential(*layers)

    def size_transform(self, input_size):
        conv_size = input_size - 2 * self.n_conv
        pool_size = conv_size if self.pool == 'none' else int(math.floor(conv_size/2))
        return pool_size
    
    def forward(self, inputs):
        return self.net(inputs)

class ConvEncoder(nn.Module):
    def __init__(self, layers, img_size, output_size, avg_pool=False):
        super().__init__()
        self.output_size = output_size
        conv_out_size = self._get_output_size(layers, img_size)
        if avg_pool:
            self.conv = nn.Sequential(*layers, nn.AvgPool2d(conv_out_size))
            self.fc = nn.Linear(layers[-1].out_filters, output_size)
        else:
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(layers[-1].out_filters * conv_out_size * conv_out_size, output_size)

    def _get_output_size(self, layers, input_size):
        x = input_size
        for layer in layers:
            x = layer.size_transform(x)
        return x

    def forward(self, inputs):
        conv_out = self.conv(inputs)
        fc_out = self.fc(conv_out.view(*inputs.size()[:-3], -1))
        return fc_out