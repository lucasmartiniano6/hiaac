import matplotlib.pyplot as plt
import numpy as np

with open("res.txt", "r") as f:
    y = []
    for line in f.readlines():
        curr = [float(i) for i in line.split()]
        y.append(curr)
    y = np.mean(y, axis=0)
    print('mean_acc', y)
    x = np.linspace(5,100, len(y))
    plt.scatter(x,y, c='r')
    plt.plot(x, y, 'r-')
    plt.ylim(0, 100)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of classes')
    plt.show()