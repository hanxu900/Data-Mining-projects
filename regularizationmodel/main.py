import numpy as np
from scipy.stats import norm
from src.scale import scale
from src.sim import sim
from model.pcr import PCR
from model.pls import PLS
from model.ridge import Ridge


x = np.loadtxt("./data/x.txt", delimiter=",")
y = np.loadtxt("./data/y.txt", delimiter=",")
index = np.loadtxt("./data/index.txt", delimiter=",", dtype=bool)
names = np.loadtxt("./data/names.txt", delimiter=",", dtype=str)
x, _, _ = scale(x)
x_train = x[index]
x_test = x[~index]
y_train = y[index]
y_test = y[~index]
n, p = x_train.shape
pcr1 = PCR(x_train, y_train, names[0:p], is_scale=True, is_var_exp=True)
pcr1.pcr()
# pcr1.b
pcr1.cv(n)
pcr1.report_coe()
pcr1.report_var_exp()
pcr1.predict_err(x_test, y_test)
pcr1.test_err(x_test, y_test)
############
is_scale = False
is_var_exp = True
pls1 = PLS(x_train, y_train, names[0:p], is_scale, is_var_exp)
pls1.pls()
pls1.report_var_exp()
pls1.cv(n)
pls1.report_coe()
pls1.predict_err(x_test, y_test)
pls1.test_err(x_test, y_test)
############
inter = False
is_scale = True
Lam = [1, 2, 3, 4, 5]
ridge1 = Ridge(x_train, y_train, inter, is_scale, Lam)
ridge1.cv(n)
ridge1.ridge()
ridge1.predict_err(x_test, y_test)
###################################
n, p, rho = 1000, 10, 0.75
mu = norm.rvs(size=p, scale=1)
beta0, beta1 = 0.5, 0.5 * np.ones(p, dtype=float)
x, y = sim(n, p, rho, mu, beta0, beta1)
n1 = 2000
x1, y1 = sim(n, p, rho, mu, beta0, beta1)
names = list(range(p))
pcr1 = PCR(x, y, names, is_scale=True, is_var_exp=True)
pcr1.pcr()
# pcr1.b
pcr1.cv(n)
pcr1.report_coe()
pcr1.report_var_exp()
pcr1.predict_err(x1, y1)
pcr1.test_err(x1, y1)
############
is_scale = False
is_var_exp = True
pls1 = PLS(x, y, names, is_scale, is_var_exp)
pls1.pls()
pls1.report_var_exp()
pls1.cv(n)
pls1.report_coe()
pls1.predict_err(x1, y1)
pls1.test_err(x1, y1)
############
inter = False
is_scale = True
Lam = [1, 2, 3, 4, 5]
ridge1 = Ridge(x, y, inter, is_scale, Lam)
ridge1.cv(n)
ridge1.ridge()
ridge1.predict_err(x1, y1)
