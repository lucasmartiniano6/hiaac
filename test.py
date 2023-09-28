import torch
from avalanche.benchmarks.classic import SplitCIFAR100
import matplotlib.pyplot as plt
import itertools
import random
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch import nn 

 
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

def main():
    from torch.utils.data import DataLoader

    benchmark = SplitCIFAR100(n_experiences=20, seed=42)
    train_ds = torch.utils.data.ConcatDataset([benchmark.train_stream[0].dataset, benchmark.train_stream[1].dataset])
    valid_ds = torch.utils.data.ConcatDataset([benchmark.test_stream[0].dataset, benchmark.test_stream[1].dataset])

    batch_size = 8
    # PyTorch data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    # model = to_device(ResNet9(3, 100), device)
    model = to_device(SlimResNet34(100), device)

    history = [evaluate(model, valid_dl)]
    epochs = 30
    max_lr = 0.1
    grad_clip = 0.1
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), max_lr, momentum=0.9, weight_decay=weight_decay)
#   fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, grad_clip=grad_clip, weight_decay=weight_decay, opt_func=optimizer)

    torch.cuda.empty_cache()
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_dl:
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
        result = evaluate(model, valid_dl)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(torch.nn.Module):
    def training_step(self, batch):
        images, labels, _= batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels, _ = batch 
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
class ResNet(ImageClassificationBase):
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
