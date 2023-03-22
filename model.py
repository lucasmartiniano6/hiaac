import torch
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training import GEM
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_strategy(check_plugin):
    model = SimpleMLP(num_classes=25, input_size=51, hidden_size=256, hidden_layers=3, drop_rate=0.001)
#    model.load_state_dict(torch.load('saved_model.pth'))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                loggers=[InteractiveLogger()])
    cl_strategy = GEM(
        model, optimizer, criterion,
        memory_strength=0.5, patterns_per_exp=100,
        train_mb_size=100, train_epochs=10, eval_mb_size=100, device=device,
        evaluator=eval_plugin
#        plugins=[check_plugin]  # uncomment this line to activate checkpoints/
    )
    return cl_strategy