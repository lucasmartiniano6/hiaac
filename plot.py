import matplotlib.pyplot as plt
import numpy as np

def avg_acc_per_class_step(path, color):
    with open(path + "res.txt", "r") as f:
        y_seen, y_all = [], []
        for line in f.readlines():
            a = line.split()
            if len(a) > 0 and a[0] == 'avg_acc_seen_classes:':
                y_seen.append(float(a[1]))
            elif len(a) > 0 and a[0] == 'avg_acc_all_classes:':
                y_all.append(float(a[1]))
        
        x = np.linspace(5,100, len(y_seen))
        plt.scatter(x,y_seen, c=color)
        plt.plot(x, y_seen, color+'-', label=path+'avg_acc_seen_classes')
        #plt.scatter(x,y_all, c='b')
        #plt.plot(x, y_all, 'b-', label=path+'avg_acc_all_classes')
        plt.legend()
        plt.ylim(0, 100)
        plt.ylabel('Avg Accuracy')
        plt.xlabel('Number of classes')
        plt.title('Avg Accuracy per Number of Classes (TEST SET)')

def avg_acc_per_epoch(experience, path, color):
    epoch = []
    avg_acc = []
    with open(path  + "log.txt", "r") as f:
       for line in f.readlines():
            a = line.split()
            if len(a) > 0 and a[0] == 'exp'+str(experience):
                epoch.append(int(a[3]))
                avg_acc.append(float(a[6]))
    plt.scatter(epoch,avg_acc, c=color)
    plt.plot(epoch, avg_acc, color+'-', label=path+f'exp{str(experience)}_acc_per_epoch')
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel(f'Exp{str(experience)} Accuracy')
    plt.xlabel('Epoch')
    plt.title(f'Exp{str(experience)} Accuracy per Epoch')

def loss_per_epoch(experience, path, color):
    # reescrever
    epoch, loss = [], []
    i, cnt = 0, 0
    with open(path  + "loss.txt", "r") as f:
        save = False
        while True:
            line = f.readline().split()
            if(len(line) >= 2 and line[0] == 'EXP:' and int(line[1]) > experience):
                break
            if(len(line) >= 2 and line[0] == 'EXP:' and int(line[1]) == experience):
                save = True
            i = cnt // 125
            if save and len(line) == 1 and '=' not in line[0]:
                epoch.append(min(i,9))
                loss.append(float(line[0]))
            cnt += 1
    plt.plot(epoch, loss, color+'-', label=path+f'exp{str(experience)}_loss_per_epoch')
    plt.legend()
    plt.ylabel(f'Exp{str(experience)} Loss')
    plt.xlabel('Epoch')
    plt.title(f'Exp{str(experience)} Loss per Epoch')


def acc(paths, exp=0):
    plt.clf()
    avg_acc_per_class_step(paths[0], color='r')
    avg_acc_per_class_step(paths[1], color='b')
    #plt.show()
    plt.savefig('cmp/acc_class_' + ''.join((paths[0] + paths[1]).split('/')))

    plt.clf()
    avg_acc_per_epoch(exp, paths[0], color='r')
    avg_acc_per_epoch(exp, paths[1], color='b')
    #plt.show()
    plt.savefig('cmp/acc_epoch_' + ''.join((paths[0] + paths[1]).split('/')))

def loss(paths, exp=0):
    plt.clf()
    loss_per_epoch(exp, paths[0], color='r')
    loss_per_epoch(exp, paths[1], color='b')
    #plt.show()
    plt.savefig('cmp/loss_epoch_' + ''.join((paths[0] + paths[1]).split('/')))
 
if __name__ == '__main__':
    acc(["cmp/ce_com_exemplar/", "cmp/ce_sem_exemplar/"], exp=0)
    acc(["cmp/ce_com_exemplar/", "cmp/ceMod_com_exemplar/"], exp=0)
    loss(["cmp/ce_com_exemplar/", "cmp/ce_sem_exemplar/"], exp=0)
    loss(["cmp/ce_com_exemplar/", "cmp/ceMod_com_exemplar/"], exp=0)
