import torch
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader, GroupBalancedDataLoader
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from types import SimpleNamespace
from itertools import islice 
from tqdm import tqdm
from torch.utils .tensorboard import SummaryWriter

class Strategy:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        args,
        device,
    ):
        self.CE = torch.nn.CrossEntropyLoss() 
        self.Mod_CD = CustomLoss()

        self.criterion = self.CE

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.q_herding = 5 # takes q samples of exemplar set as in ilos paper
        self.mem_size = args.mem_size

        self.old_classes = None
        self.curr_classes = None

        self.exemplar_set = ParametricBuffer(
            max_size=self.args.mem_size, 
            groupby='class',
            selection_strategy=RandomExemplarsSelectionStrategy()
        )

        self.writer = SummaryWriter('tb_data/ilos')
        self.training_loss = 0.0
        self.global_step = 0
    
 
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

    def _train_batch(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.curr_classes = torch.cat((self.old_classes, labels), 0) if self.old_classes is not None else labels
        self.curr_classes = torch.unique(self.curr_classes)

        self.training_loss += loss.item()
        self.global_step += 1
        self.writer.add_scalar('Loss/train', self.training_loss, self.global_step)        


    def train(self, experience):
        self.model.train() 
        if(len(self.exemplar_set.buffer) != 0):
            print("Using exemplar set...")
            print("Exemplar set: ", *set(sorted(self.exemplar_set.buffer.targets)))
            # buf = [v.buffer for v in self.exemplar_set.buffer_groups.values()]
            buf = [self.exemplar_set.buffer, experience.dataset]
            dataLoader = GroupBalancedDataLoader( # slow
                buf,
                batch_size=100, #TODO CHANGE
                oversample_small_groups=False
            ) # TODO: BUFFER NO BALANCING WITH PREVIOUS CLASSES data = dataLoader.data
            dl_groups = {}
            for x, y, *_ in dataLoader:
                for target in y.tolist():
                    if target not in dl_groups.keys():
                        dl_groups[target] = 1
                    else:
                        dl_groups[target] += 1
            dl_groups = {k: v for k, v in sorted(dl_groups.items(), key=lambda item: item[1])} 
            print(dl_groups)
            print('Size of balanced', sum(dl_groups.values()))
        for epoch in range(self.args.epochs):
            mb_size = self.args.train_mb_size
            if experience.current_experience > 0:
                # two-step learning
                steps = zip(self.batched(experience.dataset, mb_size), dataLoader)
                i = 0
                for batch_red, batch_balanced in tqdm(steps):
                    # Red dot - Modified Cross-Distillation Loss
                    self.Mod_CD.set_old_classes(self.old_classes)
                    self.criterion = self.Mod_CD 
                    inputs, labels, *_ = zip(*batch_red)
                    inputs = torch.stack(inputs)
                    labels = torch.tensor(labels)
                    self._train_batch(inputs, labels)

                    i += labels.shape[0]

                    # Black dot - Cross-Entropy Loss
                    self.criterion = self.CE 
                    inputs, labels, *_ = batch_balanced 
                    self._train_batch(inputs, labels)

                    i += labels.shape[0]
                print("Trained on ", i, "samples")
            else:
                for batch in tqdm(self.batched(experience.dataset, mb_size)):
                    inputs, labels, *_ = zip(*batch)
                    inputs = torch.stack(inputs)
                    labels = torch.tensor(labels)
                    self.criterion = self.CE
                    self._train_batch(inputs, labels)

        print("Updating exemplar set...")
        self.exemplar_set.update(SimpleNamespace(experience=experience))
        self.old_classes = self.curr_classes
        torch.save(self.model.state_dict(), 'pth/saved_model.pth')


    def eval(self, test_stream):
        with torch.no_grad():
            for exp in test_stream:
                total, correct = 0, 0
                step = 0
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
                            
                    self.writer.add_scalar(f'Accuracy/Exp{exp.current_experience}', correct/total, step)
                    step += 1

                accuracy = 100 * correct / total 
                print(f'Accuracy for experience: {exp.current_experience} is {accuracy:.2f} %')

class CustomLoss:
    # Modified Cross-Distillation Loss
    def __init__(self):
        self.old_classes = None
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
    
    def set_old_classes(self, old_classes):
        self.old_classes = old_classes