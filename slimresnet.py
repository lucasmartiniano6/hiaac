"""This is the slimmed ResNet as used by Lopez et al. in the GEM paper."""
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from avalanche.models import MultiHeadClassifier, MultiTaskModule, DynamicModule


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1),
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2),
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2),
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.feature_extractor = nn.Sequential(*[
            *self.layer1[0],
            *self.layer2[0],
            *self.layer3[0],
            *self.layer4
        ])
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        out = relu(self.bn1(self.conv1(x)))
        out = self.feature_extractor(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SlimResNet34(nclasses, nf=20):
    """Slimmed ResNet18."""
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf)

def main():
    from avalanche.benchmarks.classic import SplitCIFAR10
    benchmark = SplitCIFAR10(n_experiences=10, shuffle=True, seed=42)
    batch = benchmark.train_stream[0].dataset[0]
    x, y, _ = batch

    model = SlimResNet34(10)
    # x.shape = [3, 32, 32]
    out = model(x)
    print(model)

if __name__ == '__main__':
    main()