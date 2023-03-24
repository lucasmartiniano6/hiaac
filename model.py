import torch
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.models import SimpleMLP, IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.supervised import ICaRL
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger

def create_strategy(args, check_plugin=None):
    model: IcarlNet = make_icarl_net(num_classes=args.total_classes)
    model.apply(initialize_icarl_net)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                loggers=[InteractiveLogger()])
    cl_strategy = ICaRL(
        model.feature_extractor, model.classifier, optimizer,
        buffer_transform=None,
        memory_size=1, fixed_memory=True,
        train_mb_size=1, train_epochs=args.epochs, eval_mb_size=1, device=device,
        evaluator=eval_plugin
#        plugins=[check_plugin]  # uncomment this line to activate checkpoints/
    )
    return cl_strategy

if __name__ == '__main__':
    pass