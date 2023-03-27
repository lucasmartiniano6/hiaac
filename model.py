import torch
import torch.nn as nn
import numpy as np
import copy
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training import HerdingSelectionStrategy
from avalanche.benchmarks.utils import make_tensor_classification_dataset
from avalanche.benchmarks.utils.utils import concat_datasets
from resnet import ResNetBaseline

class Ilos(SupervisedTemplate):
    '''
    Interpretation of the "Incremental Learning in Online Scenario" paper
    https://arxiv.org/abs/2003.13191 

    ResNet (adapted to time-series)
    Memory Replay (construct exemplar set through herding selection)
    Modified cross-distillation loss with accommodation ratio
    '''
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        args,
        device,
        evaluator = None,
        plugins = None
    ):
        if plugins is None:
            plugins = [IlosLoss()] 
        else:
            plugis += [IlosLoss()]

        super().__init__(
            model,
            optimizer,
            criterion=IlosLoss(),
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            evaluator=evaluator,
            plugins=plugins
        )

        self.args = args

        self.q_herding = 2 # takes q samples of exemplar set as in ilos paper
        self.mem_size = args.mem_size
        self.x_memory = []
        self.y_memory = []
        assert (self.args.mem_size > self.q_herding)

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
        mem_left = self.mem_size - len(self.x_memory)
        for idx in range(min(self.q_herding, mem_left)):
            self.x_memory.append(self.adapted_dataset[examplar_idx[idx]][0])
            self.y_memory.append(self.adapted_dataset[examplar_idx[idx]][1])


class IlosLoss(SupervisedPlugin):
    """
    Modified cross-distillation loss with accommodation ratio 
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

        self.old_classes = []
        self.old_model = None
        self.old_logits = None

    def before_forward(self, strategy, **kwargs):
        if self.old_model is not None:
            with torch.no_grad():
                self.old_logits = self.old_model(strategy.mb_x)

    def __call__(self, logits, targets):
        predictions = torch.sigmoid(logits)

        one_hot = torch.zeros(
            targets.shape[0],
            logits.shape[1],
            dtype=torch.float,
            device=logits.device,
        )
        one_hot[range(len(targets)), targets.long()] = 1

        if self.old_logits is not None:
            print("\nAAAAAAAAAAA\n")
            old_predictions = torch.sigmoid(self.old_logits)
            one_hot[:, self.old_classes] = old_predictions[:, self.old_classes]
            self.old_logits = None

            alpha = beta = 0.5

        return self.criterion(predictions, one_hot)

    def after_update(self, strategy, **kwargs):
        print("called")
        if self.old_model is None:
            old_model = copy.deepcopy(strategy.model)
            old_model.eval()
            self.old_model = old_model.to(strategy.device)

        self.old_model.load_state_dict(strategy.model.state_dict())

        self.old_classes += np.unique(
            strategy.experience.dataset.targets
        ).tolist()


def create_strategy(args, check_plugin=None):
    model = ResNetBaseline(in_channels=args.input_size, num_pred_classes=args.total_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_plugin = EvaluationPlugin(accuracy_metrics(minibatch=True,epoch=True,experience=True,stream=True),
                                loggers=[InteractiveLogger()])
    cl_strategy = Ilos(
        model=model, optimizer=optimizer, 
        args=args, device=device, evaluator=eval_plugin
        # plugins=[check_plugin]  # activates checkpoints/
    )
    return cl_strategy

if __name__ == '__main__':
    pass