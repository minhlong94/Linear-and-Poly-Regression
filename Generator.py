import csv
import numpy as np

with open('polydata.csv', mode='w', newline='') as e:
    e = csv.writer(e, delimiter=',', quotechar=" ", quoting=csv.QUOTE_ALL)
    np.random.seed(0)

    for _ in range(100):
        x = np.random.normal()
        y = -1.524 + 3.4519*x - 1.6818*x**2 - 1.7581*(x ** 3) + 2.3*(x ** 4) + np.random.normal()
        e.writerow([x, y])
