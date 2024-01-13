import numpy as np
with open('./rotation_loss.txt', 'r') as f:
    losses = f.readlines()

    loss_1 = []
    name_2 = []

    for j in losses:
        loss_1.append(j[:-1].split('_')[0])
        name_2.append(j[:-1].split('_')[1])

    # s = np.array(loss_1)[:3]
    # s = ['0.09868380427360535', '0.36675024032592773', '1.9401080862735398e-05']
    s = [0.09868380427360535, 0.36675024032592773, 1.9401080862735398e-05]

    print("S is: ", s)
    sort_index = np.argsort(s)
    print("Sort Index: ", sort_index)
    x = sort_index.tolist()
    print("X Before Reverse: ", x)
    x.reverse()
    print("X After Reverse: ", x)
    sort_index = np.array(x)
    print("Sort Index: ", sort_index)