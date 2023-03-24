import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class IncrementalLearningDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

class IncrementalLearningModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(IncrementalLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class IncrementalLearningTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader):
        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct / len(test_loader.dataset)

        return test_loss, accuracy

    def incremental_train(self, train_loader, num_classes, task_idx):
        self.model.train()

        # Freeze the weights of the previous layers
        for i in range(task_idx):
            for param in self.model.fc1.parameters():
                param.requires_grad = False

        # Add new output nodes to the network
        self.model.fc2 = nn.Linear(100, num_classes)

        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

        return loss.item()

def train():
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    task_idx = 0
    num_tasks = 5
    num_classes_per_task = 2

    for task in range(num_tasks):
        train_classes = list(range(task*num_classes_per_task, (task+1)*num_classes_per_task))
        test_classes = list(range(task*num_classes_per_task, (task+1)*num_classes_per_task))

        train_idx = []
        for i in train_classes:
            train_idx += list((train_dataset.targets == i).nonzero().squeeze())

        test_idx = []
        for i in test_classes:
            test_idx += list((test_dataset.targets == i).nonzero().squeeze())

        train_data = torch.utils.data.Subset(train_dataset, train_idx)
        test_data = torch.utils.data.Subset(test_dataset, test_idx)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

        input_size = train_data[0][0].shape[0] * train_data[0][0].shape[1]
        output_size = num_classes_per_task*(task+1)

        model = IncrementalLearningModel(input_size, output_size)
        trainer = IncrementalLearningTrainer(model)

        for epoch in range(5):
            trainer.train(train_loader)
        
        test_loss, accuracy = trainer.evaluate(test_loader)
        print(f'Task {task}: Test Loss: {test_loss:.4f}')


train()