import numpy as np

f = open("./outputs/cub/logfile_4.out", "r")

content = f.readlines()

best = []

for i, l in enumerate(content):
    if "model_best" in l:
        x = content[i - 13].replace("\n", "")
        x = x.split("Unlabelled_v1: All ")[1]
        all_1, x = x.split(" | Old ")
        old_1, new_1 = x.split(" | New ")
        
        x = content[i - 12].replace("\n", "")
        x = x.split("Unlabelled_v2: All ")[1]
        all_2, x = x.split(" | Old ")
        old_2, new_2 = x.split(" | New ")
        
    if "Epoch 99, Test ACC_v2:" in l:
        best.append([float(all_1), float(old_1), float(new_1), float(all_2), float(old_2), float(new_2)])
        
print(len(best), np.array(best)[:, 0])
best = np.mean(best[:5], 0) * 100
print("V1 All: {:.2f} Old: {:.2f} New: {:.2f} | V2 All: {:.2f} Old: {:.2f} New: {:.2f}".format(best[0], best[1], best[2], best[3], best[4], best[5]))        