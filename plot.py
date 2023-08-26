import matplotlib.pyplot as plt
import numpy as np

with open("res.txt", "r") as f:
    y = f.read().split()
    x = np.linspace(5, 100, len(y))
    print(x)
    print(y)
    plt.plot(x, y)
    plt.gca().invert_yaxis()
    plt.show()