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
print('Data is matrix with shape:')
print(np.shape(data))

np.random.seed(10)
# select two initial random rows for our 2 clusters
r1 = np.random.randint(0, 64, size=1)
r2 = np.random.randint(65, 128, size=1)
# print(r2)
# print(r1)

#initialise mean to values in random row +1
mean1 = np.zeros(data.shape[1], dtype=np.float32)
mean2 = np.zeros(data.shape[1], dtype=np.float32)

mean1= data[r1]
mean1=np.transpose(mean1)
mean2= data[r2]
mean2=np.transpose(mean2)

# matrix to store all labels
label = np.zeros(len(data), dtype=np.float32)

data1=data[0:64,:]
data2=data[65:128,:]

cov1 = np.cov(np.transpose(data1))
cov2 = np.cov(np.transpose(data2))

print(np.shape(cov1))

# Random covariance matrices for 2 clusters
# c1 = [1,1.5,0.5,1,0.75,1.5,1,1.5,0.5,1,0.75,1.5,0.5]
# cov1 = np.diag(c1)
#
# c2 = [1,1.25,0.5,1,0.75,1.25,1.5,1.75,0.25,1,0.75,1.15,0.15]
# cov2 = np.diag(c2)

# cov1 = [ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#          ]

# cov2 = [ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#          ]

#Assume data is divided equally into the two clusters
weight=[0.5,0.5]


# function for calculating distance
def dist(a, b):
    return distance.euclidean(a, b)**2


def dist2(a, b):
    squared_distance = 0

    # Assuming correct input to the function where the lengths of two features are the same
    for i in range(len(a)):
        squared_distance = squared_distance + ((a[i] - b[i]) ** 2)
        # print(squared_distance)

    return squared_distance**0.5


# to calculate probability of each point belonging to a cluster
def calc_prob(x, mean, cov, weight):
    new_weight=weight

    for i in range(len(x)):
        new_weight=new_weight * multivariate_normal.pdf(x[i],mean[i],cov[i][i])

    return new_weight

# Estimation step which measures probability and assigns label accordingly
def e_step(data,label, mean1, mean2, cov1, cov2, weight):

    print('In step e')
    #print('weight',weight)

    for i in range(data.shape[0]):
        in_1=calc_prob(data[i],mean1, cov1, weight[0])
        in_2=calc_prob(data[i], mean2, cov2, weight[1])
        # if iter == 2:
        #     print('1', in_1)
        #     print('2', in_2)

        if in_1>in_2:

            label[i]=1
        else:
            label[i]=2

    return label

# Maximization step which updates the variable
def m_step(data,label,mean1, mean2, cov1, cov2, weight):

    print('In step m')

    # Assign cluster label 1
    points_in_1 = [data[n] for n in range(len(data)) if label[n] == 1]
    points_in_1 = np.array(points_in_1)
    print('Number of points in cluster 1: ',len(points_in_1))

    # Assign cluster label 2
    points_in_2 = [data[n] for n in range(len(data)) if label[n] == 2]
    points_in_2 = np.array(points_in_2)
    print('Number of points in cluster 2: ', len(points_in_2))

    # calculate weight again on basis of new clusters
    weight=[(len(points_in_1)/float(len(data))),(1-(len(points_in_1)/float(len(data))))]

    # calculate mean for new clusters
    mean1=np.mean(points_in_1, axis=0)
    mean2=np.mean(points_in_2, axis=0)

    # calculate and form  covariance matrices  for new clusters
    # c1 = np.std(points_in_1, axis=0)
    # cov1=np.diag(c1)
    # c2 = np.std(points_in_2, axis=0)
    # cov2 = np.diag(c2)

    # calculate and form  covariance matrices  for new clusters
    cov1=np.cov(np.transpose(points_in_1))
    cov2=np.cov(np.transpose(points_in_2))

    return mean1, mean2, cov1, cov2, weight

shift = 128
print(shift)
e = 0.00001
iter = 0

while shift > e:
    iter = iter + 1
    # E-step
    label= e_step(data, label, mean1, mean2, cov1, cov2, weight)

    # copy value for future comparison
    old_mean1 = np.copy(mean1)
    old_mean2 = np.copy(mean2)
    # M-step

    print("\nFor iteration {}".format(iter))
    mean1, mean2, cov1, cov2, weight = m_step(data, label, mean1, mean2, cov1, cov2, weight)

    # calculate variation in both means
    shift1 = dist2(old_mean1,mean1)
    shift2 = dist2(old_mean2,mean2)

    # get total variation
    shift=shift1+shift2


    print("Shift {}".format(shift))

fig = pyplot.figure()
pyplot.scatter(data[:,0], data[:,1], 128,c=label)
pyplot.show()
fig.savefig("GMM_without_PCA{}.png".format(iter))