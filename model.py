import torch
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive
from avalanche.models import SimpleMLP
from resnet import ResNetBaseline
from ilos import Ilos

def create_strategy(args, check_plugin=None):
    model : torch.nn.Module = None
    if args.strat == 'ilos':
        model = ResNetBaseline(in_channels=args.input_size, num_pred_classes=args.total_classes)
    elif args.strat == 'naive':
        model = SimpleMLP(num_classes=args.total_classes, input_size=args.input_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

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