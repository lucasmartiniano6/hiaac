import torch
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from types import SimpleNamespace
from itertools import islice 
from tqdm import tqdm

class Strategy:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        args,
        device,
    ):
        self.criterion = torch.nn.CrossEntropyLoss() 

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.q_herding = 5 # takes q samples of exemplar set as in ilos paper
        self.mem_size = args.mem_size

        self.old_classes = None

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

        curr_classes = torch.cat((self.old_classes, labels), 0) if self.old_classes is not None else labels
        self.old_classes = torch.unique(curr_classes)

    def train(self, experience):
        self.model.train() 
        data = experience.dataset
        if(len(self.exemplar_set.buffer) != 0):
            print("Using exemplar set...")
            dataLoader = ReplayDataLoader( # slow
                experience.dataset,
                self.exemplar_set.buffer,
                batch_size=self.args.train_mb_size,
                oversample_small_tasks=True
            )
            data = dataLoader.data
        for epoch in range(self.args.epochs):
            # two-step learning
            for batch in tqdm(self.batched(experience.dataset, self.args.train_mb_size)):
                # Red dot - Modified Cross-Distillation Loss
                if experience.current_experience > 0:
                    self.criterion = CustomLoss(self.old_classes)
                    self._train_batch(batch)
            for batch in tqdm(self.batched(data, self.args.train_mb_size)):
                # Black dot - Cross-Entropy Loss
                self.criterion = torch.nn.CrossEntropyLoss()
                self._train_batch(batch)
        print("Updating exemplar set...")
        self.exemplar_set.update(SimpleNamespace(experience=experience))

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

class CustomLoss:
    # Modified Cross-Distillation Loss
    def __init__(self, old_classes):
        self.old_classes = old_classes
        self.temperature = 2.0
        self.alpha = 0.5
        self.beta = 0.5

    def __call__(self, outputs, labels):
        curr_classes = torch.cat((self.old_classes, labels), 0) 
        curr_classes = torch.unique(curr_classes)

        n, m = len(self.old_classes), len(curr_classes)
        self.alpha = n / (n+m)

        p_hat = outputs[:, self.old_classes]
        p = outputs[:, curr_classes]
        
        p_hat /= self.temperature
        p_hat -= p_hat.max(dim=1, keepdim=True)[0] # for numerical stability
        p_hat = p_hat.exp() / p_hat.exp().sum(dim=1, keepdim=True)

        p /= self.temperature
        p -= p.max(dim=1, keepdim=True)[0] # for numerical stability
        p = p.exp() / p.exp().sum(dim=1, keepdim=True)

        
        # knowledge distillation loss
        LD = -torch.mean(torch.sum(p_hat * torch.log(p[:, :n]), dim=1))

        # modified cross-entropy loss
        y_hat = torch.zeros(len(labels), p.shape[1], dtype=torch.float, requires_grad=False, device=outputs.device)
        for i in range(len(labels)):
            pos = torch.where(labels[i] == curr_classes)[0]
            y_hat[i, pos] = 1

        p_tilde = p.clone() # (, n+m) ; does this need to be normalized?
        p_tilde_hat = p_hat.clone() # (, n)
        p_tilde[:, :n] = self.beta * p_tilde[:, :n] + (1-self.beta) * p_tilde_hat
        LC = -torch.mean(torch.sum(y_hat * torch.log(p_tilde), dim=1))

        # final loss
        loss = self.alpha * LD + (1-self.alpha) * LC

        # print('labels', labels) 
        # print('old classes', self.old_classes)
        # print('curr classes', curr_classes)

        return loss