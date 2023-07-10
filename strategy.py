import torch
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from types import SimpleNamespace
from itertools import islice, cycle
from tqdm import tqdm

class Strategy:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        args,
        device,
        evaluator = None,
        plugins = None
    ):
        self.criterion = torch.nn.CrossEntropyLoss() 

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.q_herding = 5 # takes q samples of exemplar set as in ilos paper
        self.mem_size = args.mem_size

        self.exemplar_set = ParametricBuffer(
            max_size=self.args.mem_size, 
            groupby='class',
            selection_strategy=RandomExemplarsSelectionStrategy()
        )
    
 
    def batched(self, iterable, n):
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    def update_exemplars(self, inputs, labels):
        # Update exemplar set with data from batch
        exp = SimpleNamespace(dataset=TensorDataset(inputs, labels))
        exp.dataset.targets = labels.tolist()

        self.exemplar_set.update(SimpleNamespace(experience=exp))
        # print(f"Max buffer size: {self.exemplar_set.max_size}, current size: {len(self.exemplar_set.buffer)}")
        # print(*sorted(self.exemplar_set.buffer.targets))

    def _train_batch(self, batch):
        inputs, labels, *_ = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return inputs, labels

    def train(self, experience):
        self.model.train() 
        for epoch in range(self.args.epochs):
            batches = self.batched(experience.dataset, self.args.train_mb_size)
            for batch, part in tqdm(zip(batches, cycle(["offline", "online"]))):
                inputs, labels = self._train_batch(batch)
                if part == "offline":
                    self.update_exemplars(inputs, labels)
                elif part == "online":
                    # combine current batch and exemplar set
                    balanced_dl = ReplayDataLoader( # takes too long
                        AvalancheDataset(TensorDataset(inputs, labels)),
                        self.exemplar_set.buffer,
                        batch_size=self.args.train_mb_size,
                        oversample_small_tasks=True
                    )
                    for new_batch in self.batched(balanced_dl.data, self.args.train_mb_size):
                        inputs, labels = self._train_batch(new_batch)


    def eval(self, test_stream):
        with torch.no_grad():
            for exp in test_stream:
                total, correct = 0, 0
                for batch in self.batched(exp.dataset, self.args.eval_mb_size):
                    inputs, labels, _ = zip(*batch)
                    inputs = torch.stack(inputs)
                    labels = torch.tensor(labels)

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total 
                print(f'Accuracy for experience: {exp.current_experience} is {accuracy:.2f} %')