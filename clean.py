# data cleanse
import numpy as np
from tqdm import tqdm


def clean_file(file_id):
    data = np.genfromtxt(f'PAMAP2_Dataset/Protocol/subject{file_id}.dat')

    new = []
    for i in range(len(data)):
        if(data[i][1] != 0):
            new.append(data[i][1:])


    new = np.stack(new)
    np.savetxt(f'clean/{file_id}.txt', new, delimiter=' ', newline='\n')


for i in tqdm(range(101,110)):
    clean_file(str(i))