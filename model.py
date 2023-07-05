import torch
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive, GEM
from ilos import Ilos

def create_strategy(args, check_plugin=None):
    model : torch.nn.Module = None
    if args.strat == 'ilos':
        from slimrestnet import SlimResNet18
        model = SlimResNet18(nclasses=args.total_classes) # CIFAR10
    elif args.strat == 'naive':
        from avalanche.models import SlimResNet18
        model = SlimResNet18(nclasses=args.total_classes) # CIFAR10

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loggers=[InteractiveLogger(), TextLogger(open('log.txt', 'w')), TensorboardLogger()]
    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                   forgetting_metrics(experience=True, stream=True),
                                   loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                                   loggers=loggers)
    plugins = [check_plugin]

    cl_strategy = None
    if args.strat == 'ilos':
        cl_strategy = Ilos(
            model=model, optimizer=optimizer, 
            args=args, device=device, evaluator=eval_plugin, plugins=plugins
        )
    elif args.strat == 'naive':
        cl_strategy = Naive(
            model=model, optimizer=optimizer, criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=args.train_mb_size, eval_mb_size=args.eval_mb_size, train_epochs=args.epochs,
            evaluator=eval_plugin, plugins=plugins
        )
    
    return cl_strategy

if __name__ == '__main__':
    pass