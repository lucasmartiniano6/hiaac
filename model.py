from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_strategy(check_plugin):
    model = SimpleMLP(num_classes=25, input_size=52)

    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                loggers=[InteractiveLogger()])
    cl_strategy = Naive(
        model, optimizer, criterion,
        train_mb_size=10, train_epochs=1, eval_mb_size=100, device=device,
        evaluator=eval_plugin,
        plugins=[check_plugin]
    )
    return cl_strategy