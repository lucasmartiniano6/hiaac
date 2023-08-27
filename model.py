import torch
from strategy import Strategy
from slimresnet import SlimResNet34

def create_strategy(args, check_plugin=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SlimResNet34(nclasses=args.total_classes) # CIFAR100

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cl_strategy = Strategy(
        model=model, optimizer=optimizer, 
        args=args, device=device 
    )
    return cl_strategy

if __name__ == '__main__':
    pass