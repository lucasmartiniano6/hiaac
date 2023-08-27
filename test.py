import torch
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks.utils import AvalancheDataset
import matplotlib.pyplot as plt
import itertools
import random
 
def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
def scratch():
    benchmark = SplitCIFAR100(n_experiences=20, seed=42)

    exemplar = []
    q = 8 # exemplars per class

    for exp in benchmark.train_stream:
        print('exp', exp.current_experience)
        if exp.current_experience == 0:
            for batch in batched(exp.dataset, 8):
                    inputs, labels, *_ = zip(*batch)
                    inputs = torch.stack(inputs) # 8, 3, 32, 32
                    labels = torch.tensor(labels) # 8 
                    # TRAIN
            print(*set(exp.dataset.targets))
            # update herding
            exemplar_idx = random.sample(range(len(exp.dataset)), q)
            for i in exemplar_idx:
                exemplar.append(exp.dataset[i])

        elif exp.current_experience > 0:
            # two-step learning
            for batch_red, batch_black in zip(batched(exp.dataset, 8), batched(itertools.cycle(exemplar), 8)):
                # Red dot - Modified Cross-Distillation Loss
                inputs_red, labels_red, *_ = zip(*batch_red)
                inputs_red = torch.stack(inputs_red)
                labels_red = torch.tensor(labels_red)
                # print('red: ', inputs_red.shape, labels_red.shape)

                # Black dot - Cross-Entropy Loss
                inputs_black, labels_black, *_ = zip(*batch_black)
                inputs_black = torch.stack(inputs_black)
                labels_black = torch.tensor(labels_black)
                # print('black: ', inputs_black.shape, labels_black.shape)
            
                inputs_balanced = torch.cat((inputs_red, inputs_black), 0)
                labels_balanced = torch.cat((labels_red, labels_black), 0)
                # print('red+black: ', inputs_balanced.shape, labels_balanced.shape)
            # update exemplar set with NCM
            exemplar_idx = random.sample(range(len(exp.dataset)), q)
            for i in exemplar_idx:
                exemplar.append(exp.dataset[i])
 
import torch.nn.functional as F
def main():
    import torch
    import torchvision
    import tarfile
    import torch.nn as nn
    import numpy as np
    #from torchvision.datasets.utils import download_url
    from torchvision.datasets import CIFAR100
    #from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import torchvision.transforms as tt
    from torch.utils.data import random_split
    from torchvision.utils import make_grid
    import matplotlib
    import os
    import matplotlib.pyplot as plt

    project_name='resnet-practice-cifar100-resnet' 
    from torchvision.datasets.utils import download_url
    # Dowload the dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"
    download_url(dataset_url, '.')

    # Extract from archive
    with tarfile.open('./cifar100.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')
        
    # Look into the data directory
    data_dir = './data/cifar100'
    print(os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    print(classes)
 
    # Data transforms (normalization & data augmentation)
    #stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) - cifar10
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))      #cifar100

    train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                            tt.RandomHorizontalFlip(), 
                            # tt.RandomRotate
                            # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                            # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            tt.ToTensor(), 
                            tt.Normalize(*stats,inplace=True)])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    # PyTorch datasets
    #train_ds = ImageFolder(data_dir+'/train', train_tfms)
    #valid_ds = ImageFolder(data_dir+'/test', valid_tfms)
    train_ds = CIFAR100(root = 'data/', download = True, transform = train_tfms)
    valid_ds = CIFAR100(root = 'data/', train = False, transform = valid_tfms)

    print(train_ds)
    print(valid_ds)
    print(len(train_ds))
    print(len(valid_ds))
    print('total classes:', len(train_ds.classes))
    print(train_ds.classes) 
    batch_size = 400
    # PyTorch data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    # model = to_device(ResNet9(3, 100), device)
    model = to_device(SlimResNet34(100), device)

    history = [evaluate(model, valid_dl)]
    epochs = 50
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

from torch import nn 
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
class ImageClassificationBase(torch.nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = torch.nn.functional.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


from torch.nn.functional import relu, avg_pool2d
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


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 3 x 32 x 32
        self.conv1 = conv_block(in_channels, 64)         # 64 x 32 x 32
        self.conv2 = conv_block(64, 128, pool=True)      # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128))  # 128 x 16 x 16
        
        self.conv3 = conv_block(128, 256, pool=True)    # 256 x 8 x 8
        self.conv4 = conv_block(256, 512, pool=True)    # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), 
                                  conv_block(512, 512))  # 512 x 4 x 4
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), # 512 x 1 x 1
                                        nn.Flatten(),     # 512
                                        nn.Dropout(0.2),  
                                        nn.Linear(512, num_classes)) # 100
        
    def forward(self, xb):
        out1 = self.conv1(xb)
        out2 = self.conv2(out1)
        out3 = self.res1(out2) + out2
        out4 = self.conv3(out3)
        out5 = self.conv4(out4)
        out6 = self.res2(out5) + out5
        out = self.classifier(out6)
        return out


def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# call model eval before doing any evaluation - good practice
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()      
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

if __name__ == '__main__':
    main()
