import csv
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot
from copy import deepcopy
import math
import random
from scipy.spatial import distance
from scipy.stats import norm
from sys import maxsize
from scipy.stats import multivariate_normal


# Read data from CSV
with open('audioData.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = list(reader)

data = np.array(data, dtype=np.float32)

np.random.seed(10)

# calculate mean of the data
mean= np.mean(data, axis=0)

# Normalise the data
max= np.amax(data, axis=0)
min= np.amin(data, axis=0)

data=data-mean/ (max-min)

# cov_mat = (data).T.dot((data)) / (data.shape[0]-1)
# print(cov_mat[1])
# calulate covariance of data
cov=np.cov(data.T)
print(np.shape(cov))

eigen_value,eigen_vector=np.linalg.eig(cov)

for i in np.transpose(eigen_vector):
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(i))

# combine eigen value and eigen vector
eigen_pairs = [(np.abs(eigen_value[i]), eigen_vector[:,i]) for i in range(len(eigen_value))]

# Sort by eigen values and select top two columns
eigen_pairs.sort(key=lambda eigen_pairs: eigen_pairs[0], reverse=True)
max_var = np.hstack((eigen_pairs[0][1].reshape(13,1),eigen_pairs[1][1].reshape(13,1)))

# form the new matrix by taking dot product of original data with maximum variation
pca_data = data.dot(max_var)
print(np.shape(pca_data))

# copy data back
data=np.copy(pca_data)

print('Data is matrix with shape:')
print(np.shape(data))

# K is in range of 2 to 10
k = np.arange(2, 11)
print(k)

# loss matrix for each k
loss = np.zeros(len(k), dtype=np.float32)

# 3 different ways were tested for calculating distance
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def dist2(a, b):
    squared_distance = 0
    for i in range(len(a)):
        squared_distance = squared_distance + ((a[i] - b[i]) ** 2)
        # print(squared_distance)
    return squared_distance

def dist3(a, b):
    return distance.euclidean(a, b)

# loop for each value of k
for u in range(len(k)):

    losses = np.zeros(20, dtype=np.float32)

    # for better clustering loop for 20 times for each K and then take minimum loss value
    for iter in range(20):

        # select random k rows from 0 to 128
        k_ran = np.random.randint(0, 128, size=k[u])

        # form initial centroid matrix
        k_mat = np.zeros((k[u], data.shape[1]), dtype=np.float32)
        for i in range(k[u]):
            k_mat[i] = data[k_ran[i]]

        # To store the value of centroids when it updates
        k_old = np.zeros(k_mat.shape)

        clusters = np.zeros(len(data), dtype=np.int)
        cluster_sum = np.zeros((k[u], data.shape[1]), dtype=np.float32)
        cluster_size = np.zeros(k[u])
        distances = np.zeros(k[u], dtype=np.float32)

        # # Error func. - Distance between new centroids and old centroids
        error = dist(k_mat, k_old, None)

        # Loop till the error becomes zero
        while error != 0:
            # Assigning each value to its closest cluster
            for i in range(len(data)):
                for j in range(k[u]):
                    #calculate distance from mean
                    distances[j] = distance.euclidean(data[i, :], k_mat[j, :])

                # print(distances)
                clusters[i] = np.argmin(distances)
                # print(clusters)

            # copy current centroid values
            k_old = np.copy(k_mat)

            # Finding the new centroids by caluculating mean of new clusters
            for m in range(k[u]):
                points = [data[n] for n in range(len(data)) if clusters[n] == m]
                points=np.array(points)
                cluster_size[m] = len(points)
                # print(cluster_size)

            # to consider the case if a cluster has zero elements
            for l in range(k[u]):
                if cluster_size[l] == 0:
                    # print('here')

                    # find the largest to split into 2
                    max_cluster = np.argmax(cluster_size)
                    # print(max_cluster)

                    # find the points in the largest cluster
                    max_indices = np.where(clusters == max_cluster)
                    max_indices = np.array(max_indices[0])

                    # shuffling the sample points in largest cluster to  split in 2
                    random.shuffle(max_indices)

                    changed_mat = max_indices[0:(max_indices.shape[0]) // 2]
                    # print(changed_mat)
                    for m in range(0, changed_mat.shape[0]):
                        # assign label of the cluster whose size was 0
                        clusters[changed_mat[m]] = l

                    # Again calculate cluster size for another zero matrix
                    for m in range(k[u]):
                        points = [data[n] for n in range(len(data)) if clusters[n] == m]
                        points=np.array(points)
                        cluster_size[m] = len(points)

            # when all cluster has min 1 elements calculate final cluster size
            for m in range(k[u]):
                points = [data[n] for n in range(len(data)) if clusters[n] == m]
                points = np.array(points)
                k_mat[m] = np.mean(points, axis=0)

            # calculate error
            error = dist(k_mat, k_old, None)

        # calculate losses for each value of k
        for m in range(k[u]):
            for n in range(len(data)):
                if clusters[n] == m:
                    losses[iter] += distance.euclidean(data[n,:], k_mat[m,:])**2

    # take minimum value of 20 losses as the final loss value for each k
    loss[u]=np.min(losses)
    # print("For k = ", u + 2)
    # print("loss is = ", loss[u])


print(loss)

# create graph for losses vs k value
fig = pyplot.figure()
pyplot.plot(k, loss, label="Loss vs K values for KMeans")
pyplot.ylabel('Loss')
pyplot.xlabel('K Values')
pyplot.show()
fig.savefig("Kmeans_with_pca.png")
