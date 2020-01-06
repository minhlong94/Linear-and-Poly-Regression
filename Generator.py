import csv
import numpy as np

with open('multiLinear.csv', mode='w', newline='') as e:
    e = csv.writer(e, delimiter=',', quotechar=" ", quoting=csv.QUOTE_ALL)
    np.random.seed(0)

    for _ in range(200):
        x1 = np.random.normal()
        x2 = np.random.normal()
        y = -1.524 + 3.4519*x1 - 1.6818*x2 + np.random.normal()
        e.writerow([x1, x2, y])