import numpy as np
import torch
import torch.nn as nn

import numpy as np


class MatrixModifier:
    def __init__(self, size, center, radius):
        """
        初始化MatrixModifier对象。

        参数：
        size (tuple): 矩阵的大小，如 (768, 768)
        center (tuple): 圆心的坐标，如 (384, 384)
        radius (int): 圆的半径，如 307
        """
        self.size = size
        self.center = center
        self.radius = radius
        self.matrix = np.ones(self.size)

    def modify_matrix(self):
        """
        修改矩阵，将以中心为圆心、半径为指定值的区域置为0。
        """
        y, x = np.ogrid[:self.size[0], :self.size[1]]
        distance_from_center = np.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
        mask = distance_from_center >= self.radius
        self.matrix[mask] = 0

        return self.matrix

    def display_matrix(self):
        """
        打印矩阵。
        """
        print(self.matrix)


# 示例用法
# modifier = MatrixModifier(size=(768, 768), center=(384, 384), radius=307)