import numpy as np
from src.solve_sym import solve_sym


def ridge_pure(xtx, xty, x_mean, y_mean, n, p, is_scale=True, lam=0):
    xtx_scale = xtx - np.outer(x_mean, x_mean) * n
    xty_scale = xty - x_mean * y_mean * n
    x_std = None
    if is_scale:
        x_std = np.sqrt(np.diag(xtx_scale)/(n-1))
        x_std_mat = 1 / np.repeat(x_std.reshape((1, p)), p, axis=0)
        xtx_scale = x_std_mat.T * xtx_scale * x_std_mat
        xty_scale = xty_scale / x_std
    lam_mat = np.identity(p) * lam
    b1 = solve_sym(xtx_scale + lam_mat, xty_scale)
    if is_scale:
        b1 = b1 / x_std
    b0 = y_mean - np.dot(b1, x_mean)
    return b0, b1
