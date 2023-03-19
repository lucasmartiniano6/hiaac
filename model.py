from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training import GEM
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_strategy(check_plugin):
    model = SimpleMLP(num_classes=25, input_size=52, hidden_size=100, hidden_layers=2, drop_rate=0)

    optimizer = SGD(model.parameters(), lr=0.1)
    criterion = CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                loggers=[InteractiveLogger()])
    cl_strategy = GEM(
        model, optimizer, criterion,
        memory_strength=0.5, patterns_per_exp=5000,
        train_mb_size=100, train_epochs=10, eval_mb_size=200, device=device,
        evaluator=eval_plugin,
        plugins=[check_plugin]
    )
    return cl_strategy