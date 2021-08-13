# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:11:59 2020

@author: xuhan
"""

# 用模拟数据测试Kmeans类并选出最优k值

# 导入Kmeans类
import numpy as np
import os
os.chdir("D:/1s/数据挖掘/1大作业/聚类/k_means")
from Kmeans import Kmeans

# 随机生成100个七维数据，k的取值范围从2到\sqrt{100}=10
np.random.seed(5)
data = np.random.randn(100,3)
k_list = list(range(2,10))
best_k = cal_best_k(data, k_list)
print(best_k)

# 用二维数据画出聚类结果（取k=4）
np.random.seed(7)
data2 = np.random.randn(100,2)
k2 = Kmeans(data2, 10, 25, 4)
k2.cal_all()
k2.plot_cluster()

# 用真实数据测试Kmeans类——对比k-means, k-medoids

iris = pd.read_csv("D:/1s/数据挖掘/1大作业/聚类/data/iris.csv")
# 生成乱序index来打乱数据顺序，并按照2:1的比例分为训练系和测试集
np.random.seed(7)
index1 = np.arange(len(iris))  # 生成下标
np.random.shuffle(index1)

iris_train = np.array(iris)[index1[:100]]
iris_test = np.array(iris)[index1[100:]]
#iris_x = iris[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']]
km_iris1 = Kmeans(np.array(iris_train[:, :-1]), 40, 5, 3, center_method = "mean")
km_iris1.cal_all()
train_mean = km_iris1.point_cluster_assign+1
train_medoid = km_iris2.point_cluster_assign+1
print(train_mean)
km_iris2 = Kmeans(np.array(iris_train[:, :-1]), 40, 5, 3, center_method = "medoid")
km_iris2.cal_all()
print(iris_test[:,-1])
test_mean = km_iris1.new_assign(iris_test[:, :-1])
test_medoid = km_iris2.new_assign(iris_test[:, :-1])
print(test_mean)
print(test_medoid)

train_mean[train_mean==3] = 4
train_mean[train_mean==2] = 3
train_mean[train_mean==4] = 2

train_medoid[train_medoid==3] = 4
train_medoid[train_medoid==2] = 3
train_medoid[train_medoid==4] = 2

accuracy_train_mean = len(np.array(range(len(iris_train)))[np.where(train_mean == np.array(iris_train[:, -1]))]) / len(iris_train)
accuracy_train_medoid = len(np.array(range(len(iris_train)))[np.where(train_medoid == np.array(iris_train[:, -1]))]) / len(iris_train)
print("iris数据训练集——k-means聚类准确率为：")
print(accuracy_train_mean)
print("iris数据训练集——k-medoids聚类准确率为：")
print(accuracy_train_medoid)

test_mean[test_mean==3] = 4
test_mean[test_mean==2] = 3
test_mean[test_mean==4] = 2

test_medoid[test_medoid==3] = 4
test_medoid[test_medoid==2] = 3
test_medoid[test_medoid==4] = 2

accuracy_test_mean = len(np.array(range(len(iris_test)))[np.where(test_mean == np.array(iris_test[:, -1]))]) / len(iris_test)
accuracy_test_medoid = len(np.array(range(len(iris_test)))[np.where(test_medoid == np.array(iris_test[:, -1]))]) / len(iris_test)
print("iris数据测试集——k-means聚类准确率为：")
print(accuracy_test_mean)
print("iris数据测试集——k-medoids聚类准确率为：")
print(accuracy_test_medoid)

# 自定义Kmeans类与sklearn库中Kmeans方法准确度比较

from sklearn.cluster import KMeans
np.random.seed(1)
clf = KMeans(n_clusters=3, max_iter = 100)
s = clf.fit(iris_train[:, :-1])
train_predict = s.predict(iris_train[:, :-1])
test_predict = s.predict(iris_test[:, :-1])
accuracy_train_sklearn = len(np.array(range(len(iris_train)))[np.where(train_predict == np.array(iris_train[:, -1]))]) / len(iris_train)
accuracy_test_sklearn = len(np.array(range(len(iris_test)))[np.where(test_predict == np.array(iris_test[:, -1]))]) / len(iris_test)
print("sklearn在测试集上的准确率为：")
print(accuracy_train_sklearn)
print("sklearn在训练集上的准确率为：")
print(accuracy_test_sklearn)

accuracy = pd.DataFrame({'self_Kmeans':[accuracy_train_mean,accuracy_test_mean],
                         'self_Kmedoids':[accuracy_train_medoid, accuracy_test_medoid],
                         'sklearn_Kmeans':[accuracy_train_sklearn, accuracy_test_sklearn]}).rename(index={0: 'train', 1: 'test'})
print(accuracy)