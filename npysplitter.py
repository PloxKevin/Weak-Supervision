import numpy as np
dir = "datasets/preselection_top"
dir2 = "datasets/split/preselection_top"

dir_mnist = "datasets/mnist_test.csv"
dir_mnist2 = "datasets/split/"

top_ranges = [1, 2, 4, 8, 16, 32]

for i in top_ranges:
    data = np.load(dir + str(i) + '.npy')
    np.random.shuffle(data)
    np.save(dir2 + str(i) + '_60', data[:3000])
    np.save(dir2 + str(i) + '_30', data[3000:4500])
    np.save(dir2 + str(i) + '_10', data[4500:5000])
print(len(data[:3000]), len(data[3000:4500]), len(data[4500:5000]))

mnist = np.genfromtxt(dir_mnist, dtype =int, delimiter=",")
np.random.shuffle(mnist)
print(len(mnist[:3000]), len(mnist[3000:4500]), len(mnist[4500:5000]))
np.savetxt(dir_mnist2+"60.csv",mnist[:3000].astype(int), fmt="%i", delimiter=",")
np.savetxt(dir_mnist2+"30.csv",mnist[3000:4500].astype(int),fmt="%i", delimiter=",")
np.savetxt(dir_mnist2+"10.csv",mnist[4500:5000].astype(int),fmt="%i", delimiter=",")