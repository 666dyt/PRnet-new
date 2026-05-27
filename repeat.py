import numpy as np
import pandas as pd

# 读取你 vega_50_epoch 的预测值和输入对照组
y_pre = pd.read_csv('./results/lincs/random_split_0_y_pre_array.csv', header=None).values
x_true = pd.read_csv('./results/lincs/random_split_0_x_true_array.csv', header=None).values

# 计算预测值和对照组的平均绝对误差
diff = np.mean(np.abs(y_pre[:, 1:] - x_true[:, 1:])) 
print(f"预测值和输入值的平均差距: {diff}")