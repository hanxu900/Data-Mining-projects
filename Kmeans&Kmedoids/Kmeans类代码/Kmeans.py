import numpy as np
import matplotlib.pyplot as plt
from src import mm_EuDistance, cal_center


class Kmeans(object):
    def __init__(self, data, minpts, p, k, center_method="mean"):
        self.data = data
        self.minpts = minpts  # 用以xi为中心，包含常数minpts个数据对象的半径来衡量xi的密度
        self.n = data.shape[0]  # 数据对象的个数
        self.dim = data.shape[1]  # 每个对象的维度
        self.p = p  # 取p个密度较大的点作为初始中心的备选点
        self.k = k  # 聚成的类的个数
        self.center_method = center_method  # 计算中心的方法
        self.distance = mm_EuDistance(data, data)  # 对象两两之间的距离的矩阵，distance[i][j]表示第i和对象和第j个对象之间的距离
        self.density = np.zeros(self.n)  # 每个对象的密度
        self.centroid = np.zeros([k, self.dim])  # 每个类的中心坐标，初始值为由上述方法选出的k个点
        self.J = 0  # 判断迭代是否结束的目标函数，若两次相邻迭代后J保持不变，则停止迭代
        self.point_cluster_dist = np.zeros([self.n, k])  # 每个对象到k个类中心的距离
        self.point_cluster_assign = np.zeros(self.n)  # 每个对象距离最近的中心在centroid数组中的索引
        self.cluster = []  # cluster列表存放每个类中的对象在data中的索引
        self.avg_BWP = 0  # 当前参数下的平均BWP指标值，用于选择最优参数

    # 初始化每个对象的密度
    def init_density(self):
        for i in range(self.n):
            nearest = self.distance[i].argsort()[:self.minpts + 1]  # 每个对象距离最近的minpts个点（包含对象自身）
            self.density[i] = self.distance[nearest][:, nearest].sum()  # 这minpts个对象的两两距离之和，density越小，密度越大

    # 初始化中心
    def init_centroid(self):
        high = self.density.argsort()[:self.p]  # 取density中前p个对象，即密度最大的p个对象
        used = [high[0]]  # used列表表示已经选出的作为初始中心的对象
        unused = high[1:]  # 剩下的没有作为初始中心的对象
        pick = [0] * self.k
        # 选剩下的k-1个初始中心
        for i in range(self.k - 1):
            pick[i] = np.min(self.distance[used][:, unused])
            used.append(np.where(self.distance == np.max(pick[i]))[0].tolist()[0])
            used.append(np.where(self.distance == np.max(pick[i]))[0].tolist()[1])
            used = np.unique(used)
            used = used.tolist()
            unused = list(filter(lambda x: x not in used, high))
        self.centroid = self.data[used]

    def k_means(self):
        iteration = 0  # 记录迭代次数
        j_changed = True
        while j_changed:
            # 根据每个类对象的均值(中心对象)，计算每个对象与这些中心对象的距离
            self.point_cluster_dist = mm_EuDistance(self.data, self.centroid)
            # 根据最小距离重新对相应对象进行划分
            self.point_cluster_assign = np.argmin(self.point_cluster_dist, axis=1)
            # 重新计算每个类的均值（中心对象）
            # centroid[0] = cal_mean(np.where(point_cluster_assign==0))
            self.centroid = np.array(
                list(map(lambda j: cal_center(self.data[np.where(self.point_cluster_assign == j)], self.center_method),
                         list(range(self.k)))))
            # print(centroid)
            iteration += 1
            # 判断目标函数J是否改变，并更新J
            j_new = np.sum(self.point_cluster_dist)
            if self.J == j_new:
                j_changed = False
            self.J = j_new
        print("迭代次数为：")
        print(iteration)
        return self.centroid

    # 只能画出二维数据
    def plot_cluster(self):
        if self.dim != 2:
            print("对不起，本模块无法画出三维及以上数据!")
            return 1

        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        if self.k > len(mark):
            print("对不起，本模块无法画出十一及以上类别")
            return 1

        # data_sort = self.data.append(self.point_cluster_assign, axis=1)

        for i in range(self.n):
            markIndex = int(self.point_cluster_assign[i])  # 为样本指定颜色
            plt.plot(self.data[i][0], self.data[i][1], mark[markIndex])

        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 画出中心点
        for i in range(self.k):
            plt.plot(self.centroid[i][0], self.centroid[i][1], mark[i], markersize=12)
        plt.show()

    # 计算b(i,j), w(i,j)以及BWP(i,j)，并求出该类数k下的平均BWP（avg_BWP)
    def cal_BWP(self):
        cluster = list(
            map(lambda j: np.array(list(range(self.n)))[np.where(self.point_cluster_assign == j)], list(range(self.k))))
        # 定义第j类的第i个样本的最小类间距离b(j，i)：该样本到其他每个类中样本平均距离的最小值
        self.cluster = cluster
        b_mat = list(range(self.k))
        for j in range(self.k):
            other_list = list(range(j)) + list(range(j + 1, self.k))
            b_mat[j] = np.min(
                np.array(list(map(lambda x: np.mean(self.distance[cluster[j]][:, cluster[x]], axis=1), other_list))),
                axis=0)
        # 第j类的第i个样本的类内距离w(j，i)：该样本到第j类中其他所有样本的平均距离
        w_mat = list(range(self.k))
        for j in range(self.k):
            w_mat[j] = np.mean(self.distance[cluster[j]][:, cluster[j]], axis=1)
        baw = np.array(b_mat**2) + np.array(w_mat**2)
        bsw = np.array(b_mat**2) - np.array(w_mat**2)
        BWP = bsw / baw
        self.avg_BWP = np.array(list(map(lambda i: np.mean(BWP[i]), list(range(self.k))))).mean()

    # 将上述步骤汇总到一个函数中，方便调用
    def cal_all(self):
        self.init_density()
        self.init_centroid()
        self.k_means()
        self.cal_BWP()

    # 对于新数据进行分类（将新数据划分到已经聚好的类中）
    def new_assign(self, new_data):
        return np.argmin(mm_EuDistance(new_data, self.centroid), axis=1) + 1
