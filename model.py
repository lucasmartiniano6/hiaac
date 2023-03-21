import torch
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_strategy(check_plugin):
    model = SimpleMLP(num_classes=25, input_size=51, hidden_size=512, hidden_layers=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                loggers=[InteractiveLogger()])
    cl_strategy = Naive(
        model, optimizer, criterion,
        train_mb_size=150, train_epochs=10, eval_mb_size=150, device=device,
        evaluator=eval_plugin,
        plugins=[check_plugin]
    )
    return cl_strategy