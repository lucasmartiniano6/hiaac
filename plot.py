import matplotlib.pyplot as plt
import numpy as np

# 38.60 36.10 27.60 22.95 14.92 13.03 10.49 10.02 8.60 10.62 8.02 7.68 6.94 6.31 6.39 5.36 7.65 4.74 3.80 3.80

with open("res.txt", "r") as f:
    y = f.read().split()
    x = np.linspace(5,100, len(y))
    print(x)
    print(y)
    plt.plot(x, y)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()