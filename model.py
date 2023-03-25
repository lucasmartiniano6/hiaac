import torch
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.models import SimpleMLP, IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.supervised import ICaRL, Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from resnet import ResNetBaseline

def create_strategy(args, check_plugin=None):
    model = ResNetBaseline(in_channels=args.input_size, num_pred_classes=args.total_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                loggers=[InteractiveLogger()])
    cl_strategy = Naive(
        model, optimizer, criterion,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=args.eval_mb_size, device=device,
        evaluator=eval_plugin
#        plugins=[check_plugin]  # uncomment this line to activate checkpoints/
    )
    return cl_strategy

if __name__ == '__main__':
    pass