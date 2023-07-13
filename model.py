import torch
from strategy import Strategy
from slimrestnet import SlimResNet34

def create_strategy(args, check_plugin=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SlimResNet34(nclasses=args.total_classes) # CIFAR100
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    cl_strategy = Strategy(
        model=model, optimizer=optimizer, 
        args=args, device=device 
    )
    return cl_strategy

if __name__ == '__main__':
    pass