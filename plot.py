import matplotlib.pyplot as plt
import numpy as np

path = "compare/"
def avg_acc_per_class_step():
    with open(path + "res.txt", "r") as f:
        y_seen, y_all = [], []
        for line in f.readlines():
            a = line.split()
            if len(a) > 0 and a[0] == 'avg_acc_seen_classes:':
                y_seen.append(float(a[1]))
            elif len(a) > 0 and a[0] == 'avg_acc_all_classes:':
                y_all.append(float(a[1]))
        
        x = np.linspace(5,100, len(y_seen))
        plt.scatter(x,y_seen, c='r')
        plt.plot(x, y_seen, 'r-', label=path+'avg_acc_seen_classes')
        plt.scatter(x,y_all, c='b')
        plt.plot(x, y_all, 'b-', label=path+'avg_acc_all_classes')
        plt.legend()
        plt.ylim(0, 100)
        plt.ylabel('Avg Accuracy')
        plt.xlabel('Number of classes')
        plt.title('Avg Accuracy per Number of Classes (TEST SET)')

def avg_acc_per_epoch(experience):
    epoch = []
    avg_acc = []
    with open(path  + "log.txt", "r") as f:
       for line in f.readlines():
            a = line.split()
            if len(a) > 0 and a[0] == 'exp'+str(experience):
                epoch.append(int(a[3]))
                avg_acc.append(float(a[6]))
    plt.scatter(epoch,avg_acc, c='b')
    plt.plot(epoch, avg_acc, 'b-', label=path+f'exp{str(experience)}_acc_per_epoch')
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel(f'Exp{str(experience)} Accuracy')
    plt.xlabel('Epoch')
    plt.title(f'Exp{str(experience)} Accuracy per Epoch')


if __name__ == '__main__':
    path = "compare/ce_com_exemplar"
    avg_acc_per_class_step()
    path = "compare/ce_sem_exemplar"
    avg_acc_per_class_step()
    plt.show()

    # avg_acc_per_epoch(0)
