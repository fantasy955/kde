import numpy as np
import scipy.stats as st
import matplotlib.pyplot as pl
import pymysql.cursors
from scipy import linalg
import time


class Kde2:
    def __init__(self, samples):
        self.initKde2Params(samples)

    # 根据已有数据集初始化相关参数
    def initKde2Params(self, samples):
        self.d, self.n = samples.shape
        if not self.d == 2:
            msg = "维度不为2"
            raise ValueError(msg)
        self.dataset = samples
        self.factor = self.n ** (-1. / (2 + 4))
        self.weights = np.zeros(self.n)
        self.weights[:] = 1 / self.n
        self.cov1 = np.cov(samples)

        self.inv_cov1 = linalg.inv(self.cov1)
        self.cov2 = self.cov1 * (self.factor ** 2)
        self.inv_cov2 = self.inv_cov1 / (self.factor ** 2)
        self.norm_factor = np.sqrt(linalg.det(2 * np.pi * self.cov2))

    # 估计输入点概率
    def evaluateKde2(self, positions):
        row, col = positions.shape
        if not self.d == row:
            msg = "维度不一致"
            raise ValueError(msg)

        points = positions
        result = np.zeros((col,), dtype=float)
        whitening = linalg.cholesky(self.inv_cov2)

        scaled_dataset = np.dot(whitening, self.dataset) #

        scaled_points = np.dot(whitening, points)

        if col >= self.n:
            for i in range(self.n):
                diff = scaled_dataset[:, i, None] - scaled_points
                energy = np.sum(diff * diff, axis=0) / 2.0
                # result += self.weights[i] * np.exp(-energy)
                result += np.exp(-energy)
        else:
            for i in range(col):
                diff = scaled_dataset - scaled_points[:, i, None]
                energy = np.sum(diff * diff, axis=0) / 2.0
                # result[i] = np.sum(np.exp(-energy) * self.weights, axis=0)
                result[i] = np.sum(np.exp(-energy) , axis=0)

        result = result / self.norm_factor
        return result
