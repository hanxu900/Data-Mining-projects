import numpy as np

def mm_EuDistance(X, Y):
    X2_sum = np.matrix(np.sum(X**2, axis=1))
    X2_sum_ex = np.tile(X2_sum.T, (1, np.dot(X, Y.T).shape[1]))
    Y2_sum = np.sum(Y**2, axis=1)
    Y2_sum_ex = np.tile(Y2_sum, (np.dot(X, Y.T).shape[0], 1))
    D = X2_sum_ex + Y2_sum_ex - 2*np.dot(X, Y.T)
    D[D<0]=0.0
    return np.sqrt(np.array(D))

# 在K-means中，我们将中心点取为当前cluster中所有数据点的平均值，
# 在K-medoids算法中，我们将从当前cluster中选取这样一个点——它到其他所有（当前cluster中的）点的距离之和最小——作为中心点。
def cal_center(X, method = "mean"):
    if method == "mean":
        center = np.mean(X, axis=0)
    if method == "medoid":
        distance_tmp = mm_EuDistance(X,X)
        center = X[np.argmin(np.sum(distance_tmp, axis=0), axis=0)]
    return center