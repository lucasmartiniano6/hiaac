import torch
from storage_policy import HerdingSelectionStrategy
import itertools
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

"""
ILOS 

caso (primeira experiencia):
    exemplar set = 0
    for batch in experience:
        treinar tudo com cross-entropy padrão
    adiciona novas classes no exemplar set (herding selection)
caso (resto das experiencias): 
    exemplar set != 0
    for batch in experience:
        treinar classes novas -> cross-entropy modificada
        treinar mistura balanceada do (classes novas + exemplar set\ciclado) -> cross-entropy padrão
    adiciona novas classes no exemplar set (herding selection)
    *update exemplar set com (novas observações de classes antigas) (ref.: algoritmo1 NCM)
(...)

"""

class Strategy:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        args,
        device,
    ):
        torch.manual_seed(42)
        random.seed(42)
        self.CE = torch.nn.CrossEntropyLoss() 
        self.Mod_CD = CustomLoss()

        self.criterion = self.CE

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.eval_mb_size = args.eval_mb_size

        self.old_classes = None # 
        self.curr_classes = None # 

        self.exemplar = []
        self.q = 8 # exemplars per class

        self.writer = SummaryWriter('tb_data/ilos')
        self.training_loss = 0.0
        self.global_step = 0
    
 
    def batched(self, iterable, n):
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

    def _train_batch(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.training_loss += loss.item()
        self.global_step += 1
        self.writer.add_scalar('Loss/train', self.training_loss, self.global_step)        

    def train(self, experience):
        self.model.train() 
        self.experience = experience
        if(len(self.exemplar) != 0):
            dl_groups = {}
            for i in self.exemplar:
                dl_groups[i[1]] = dl_groups.get(i[1], 0) + 1
            print("Exemplar set: ", dl_groups)
        for epoch in range(self.args.epochs):
            mb_size = self.args.train_mb_size
            self.criterion = self.CE
            if experience.current_experience > 0:
                steps = zip(self.batched(self.experience.dataset, mb_size), self.batched(itertools.cycle(self.exemplar), mb_size))
                for batch_red, batch_black in tqdm(steps): 
                    # Red dot - Current classes
                    inputs_red, labels_red, *_ = zip(*batch_red)
                    inputs_red = torch.stack(inputs_red)
                    labels_red = torch.tensor(labels_red)

                    # Black dot - Old classes
                    inputs_black, labels_black, *_ = zip(*batch_black)
                    inputs_black = torch.stack(inputs_black)
                    labels_black = torch.tensor(labels_black)
                
                    # Balanced - Current + Old classes
                    inputs_balanced = torch.cat((inputs_red, inputs_black), 0)
                    labels_balanced = torch.cat((labels_red, labels_black), 0)
                    self._train_batch(inputs_balanced, labels_balanced)
            else:
                for batch in tqdm(self.batched(self.experience.dataset, mb_size)):
                    inputs, labels, *_ = zip(*batch)
                    inputs = torch.stack(inputs) # 8, 3, 32, 32
                    labels = torch.tensor(labels) # 8 

                    self._train_batch(inputs, labels)

        # update exemplar set with herding selection
        print("Updating exemplar set...")
        herding = HerdingSelectionStrategy(self.model, 'feature_extractor')
        exemplar_idx = iter(herding.make_sorted_indices(self, self.experience.dataset))
        book = {}
        while sum(book.values()) < self.q * len(self.experience.classes_in_this_experience):
            sample = self.experience.dataset[next(exemplar_idx)]
            if(book.get(sample[1], 0) < self.q):
                book[sample[1]] = book.get(sample[1], 0) + 1
                self.exemplar.append(sample)

        print('Seen classes: ', self.experience.previous_classes)
        # torch.save(self.model.state_dict(), 'pth/saved_model.pth')


    @torch.no_grad()
    def eval(self, test_stream, curr_exp):
        acc_list = [] * self.args.n_exp
        mean_acc = 0.0 # mean accuracy for whole test stream
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
            acc_list[exp.current_experience] = accuracy
            mean_acc += accuracy
        print('Acc list: ', *acc_list)
        seen_curr = sum(acc_list[:curr_exp+1]) / (curr_exp+1)
        mean_acc /= len(test_stream)
        if len(test_stream) == 1:
            open('res.txt', 'w').close()
        with open("res.txt", "a") as f:
            pass

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
