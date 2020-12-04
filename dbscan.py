import numpy as np
import pandas as pd
from numba import cuda
import math
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def plot(data, labels):
    colors = [  "blue",  "green",   "red",        "cyan",   "orange",
                "gray",  "magenta", "yellow",     "purple", "pink",
                "sienna" "lime",    "lightcoral", "olive",  "black"   ]
    attributes = ["Attribute %s"%i for i in range(1,data.shape[1]+1)]
    length = len(attributes)
    fig, axs = plt.subplots(length,length)
    for x in range(length):
        for y in range(length):
            if x == y:
                axs[x, y].plot(data[x,:],c=colors[labels])
            else:
                axs[x, y].scatter(data[x,:], data[y,:],c=colors[labels])

    for ax, i in zip(axs.flat, range(length)):
        ax.set(xlabel=attributes[i], ylabel=attributes[i])
    plt.show()

@cuda.jit()
def DBSCAN_kernel(n_neighborhoods, core_samples, labels):
    #pos = cuda.grid(1)
    label_num = 0
    for pos in range(labels.shape[0]):
        if labels[pos] == -1 and core_samples[pos]:
            labels[pos] = label_num
            if core_samples[pos]:
                neighb = n_neighborhoods[pos]
                for i in range(neighb.shape[0]):
                    v = neighb[i]
                    if v != -1:
                        if labels[v] == -1:
                            labels[v] = label_num
        label_num += 1

df = pd.read_csv('car.csv')
data = df.values
eps = 5.2

neighbors_model = NearestNeighbors(radius=eps, algorithm='auto',
                                   leaf_size=30, metric='euclidean',
                                   metric_params=None, p=None, n_jobs=None)
neighbors_model.fit(data)
neighborhoods = neighbors_model.radius_neighbors(data, return_distance=False)

labels = np.full(data.shape[0], -1, dtype=np.int32)
n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])

n_neighborhoods = []
largest_clust = np.max(n_neighbors)
for i in range(neighborhoods.shape[0]):
    arr = neighborhoods[i]
    size = neighborhoods[i].shape[0]
    if size != largest_clust:
        for i in range(size, largest_clust):
            arr = np.append(arr, [-1])
    n_neighborhoods.append(arr)

n_neighborhoods = np.vstack(n_neighborhoods)

core_samples = np.asarray(n_neighbors >= data.shape[1], dtype=np.int32)
d_labels = cuda.to_device(labels)

threadsperblock = 32
blockspergrid = math.ceil(labels.shape[0] / threadsperblock)
DBSCAN_kernel[blockspergrid, threadsperblock](n_neighborhoods, core_samples, d_labels)

result = d_labels.copy_to_host()
print(result)
plot(data, result)
