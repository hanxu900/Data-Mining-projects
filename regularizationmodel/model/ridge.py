import numpy as np
from src.ridge_pure import ridge_pure


class Ridge(object):
    def __init__(self, x=0, y=0, inter=True, is_scale=True, lam_vector=None):
        if inter:                                     # inter=True: 传入x已包含常数列
            self.x = x[:, 1:]
        else:
            self.x = x
        self.n, self.p = np.shape(self.x)
        self.y = y
        self.xtx = np.dot(self.x.T, self.x)
        self.xty = np.dot(self.x.T, self.y)
        self.x_mean = np.mean(self.x, axis=0)
        self.y_mean = np.mean(self.y)
        self.inter = inter
        self.is_scale = is_scale
        self.lam_vector = lam_vector
        self.lam_min = 0
        self.cv_err = 0
        self.coe = 0

    def ridge(self):
        b0, b1 = ridge_pure(self.xtx, self.xty, self.x_mean, self.y_mean, self.n, self.p, self.is_scale, self.lam_min)
        self.coe = np.append(b0, b1)
        return self.coe

    def cverr(self, index, lam):
        tx = self.x[index]
        ty = self.y[index]
        tn, _ = np.shape(tx)
        if tn == 1:
            tx = tx.reshape((1, self.p))
        txx_ = self.xtx - np.dot(tx.T, tx)
        txy_ = self.xty - np.dot(tx.T, ty)
        tn_ = self.n - tn
        ty_sum = np.sum(ty)
        tx_sum = np.sum(tx, axis=0)
        ty_mean_ = (self.n * self.y_mean - ty_sum)/tn_
        tx_mean_ = (self.n * self.x_mean - tx_sum)/tn_
        tb0, tb1 = ridge_pure(txx_, txy_, tx_mean_, ty_mean_, tn_, self.p, self.is_scale, lam)
        return np.sum((ty - tb0 - np.dot(tx, tb1)) ** 2)

    def cv(self, k=10):
        indexes = np.array_split(np.random.permutation(np.arange(0, self.n)), k)
        err_m = [[self.cverr(index, lam) for lam in self.lam_vector] for index in indexes]
        self.cv_err = np.sum(err_m, axis=0) / self.n
        self.lam_min = self.lam_vector[self.cv_err.argmin()]

    def predict(self, xn):
        tn, _ = np.shape(xn)
        if self.inter:
            xn_ = xn
        else:
            xn_ = np.c_[np.ones((tn, 1)), xn]
        return np.dot(self.coe, xn_.T)

    def predict_err(self, xn, yn):
        err = yn - self.predict(xn)
        err = err * err
        return np.mean(err)
