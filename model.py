import torch
import torch.nn as nn
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from avalanche.training.templates import SupervisedTemplate
from resnet import ResNetBaseline
from avalanche.benchmarks.utils import make_tensor_classification_dataset
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training import HerdingSelectionStrategy

def KDLoss():
    return torch.nn.CrossEntropyLoss()

class Ilos(SupervisedTemplate):
    '''
    Incremental Learning in Online Scenario
    https://arxiv.org/abs/2003.13191 
    '''
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_size: int,
        args,
        device,
        criterion = KDLoss(), 
        evaluator = None,
        plugins = None
    ):
        
        super().__init__(
            model,
            optimizer,
            criterion=criterion,
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            evaluator=evaluator,
            plugins=plugins
        )

        self.args = args
        
        self.mem_size = memory_size
        self.x_memory = []
        self.y_memory = []

    def _after_train_dataset_adaptation(self, **kwargs):
        if self.clock.train_exp_counter != 0: # not first exp
            memory = make_tensor_classification_dataset(
                torch.stack(self.x_memory).cpu(),
                torch.tensor(self.y_memory),
                transform=None,
                target_transform=None
            )      
            self.adapted_dataset = concat_datasets((self.adapted_dataset, memory))
        return super()._after_train_dataset_adaptation(**kwargs)

    def _after_training_exp(self, **kwargs):
        # construct exemplar set (through herding selection as in the original ilos paper)
        self.model.eval()
        self.construct_exemplar_set() 
        return super()._after_training_exp(**kwargs)

    def construct_exemplar_set(self):
        herding = HerdingSelectionStrategy(self.model, 'feature_extractor')
        examplar_idx = herding.make_sorted_indices(self, self.adapted_dataset)
        q = 2 # only take first q samples of exemplar set as in ilos paper
        mem_left = self.mem_size - len(self.x_memory)
        for idx in range(min(q, mem_left)):
            self.x_memory.append(self.adapted_dataset[examplar_idx[idx]][0])
            self.y_memory.append(self.adapted_dataset[examplar_idx[idx]][1])

def create_strategy(args, check_plugin=None):
    model = ResNetBaseline(in_channels=args.input_size, num_pred_classes=args.total_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                loggers=[InteractiveLogger()])
    cl_strategy = Ilos(
        model=model, optimizer=optimizer, 
        memory_size=5, args=args, device=device, evaluator=eval_plugin
#        plugins=[check_plugin]  # uncomment this line to activate checkpoints/
    )
    return cl_strategy

if __name__ == '__main__':
    pass