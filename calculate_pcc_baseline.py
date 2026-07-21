import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

print("Loading perfectly aligned prediction arrays from author's results...")

base_dir = '/media/mldadmin/home/s125mdg35_08/PRnet baseline/PRnet/results/lincs/'
# 1. 加载 4 个核心文件
y_true = pd.read_csv(base_dir + 'random_split_0_y_true_array.csv', header=None).values
y_pre = pd.read_csv(base_dir + 'random_split_0_y_pre_array.csv', header=None).values
x_true = pd.read_csv(base_dir + 'random_split_0_x_true_array.csv', header=None).values

with open(base_dir + 'random_split_0_cov_drug_array.csv', 'r', encoding='utf-8') as f:
    drugs_cov = [line.strip().strip('"').strip("'") for line in f.readlines()]

if y_true.shape[1] > 978:
    y_true = y_true[:, 1:]
    y_pre = y_pre[:, 1:]
    x_true = x_true[:, 1:]

min_len = min(len(y_true), len(y_pre), len(x_true), len(drugs_cov))
y_true = y_true[:min_len]
y_pre = y_pre[:min_len]
x_true = x_true[:min_len]
drugs_cov = drugs_cov[:min_len]

# 2. 计算最纯粹的配对差分 (logFC)
print("Calculating pure logFC (Paired Subtraction)...")
delta_true = y_true - x_true
delta_pre = y_pre - x_true

# ==========================================
# 指标 1: 单样本实例级评估
# ==========================================
print("Calculating Instance-level PCC...")
pccs_inst = []
for t, p in zip(delta_true, delta_pre):
    r, _ = pearsonr(t, p)
    if not np.isnan(r):
        pccs_inst.append(r)
pcc_inst_final = np.mean(pccs_inst)

# ==========================================
# 指标 2: 细粒度聚合评估 Cov_compounds
# ==========================================
print("Calculating Group-level (Cov_compounds) PCC...")
df_true = pd.DataFrame(delta_true, index=drugs_cov)
df_pre = pd.DataFrame(delta_pre, index=drugs_cov)

mean_true_cov = df_true.groupby(level=0).mean()
mean_pre_cov = df_pre.groupby(level=0).mean()

pccs_cov = []
for i in range(len(mean_true_cov)):
    r, _ = pearsonr(mean_true_cov.iloc[i].values, mean_pre_cov.iloc[i].values)
    if not np.isnan(r):
        pccs_cov.append(r)
pcc_cov_final = np.mean(pccs_cov)

# 3. 绘图输出
plt.figure(figsize=(8, 6))
# 选一个均值点作图
plt.scatter(mean_true_cov.iloc[0], mean_pre_cov.iloc[0], alpha=0.6, s=10, color='#8c564b')
plt.plot([-3, 3], [-3, 3], 'k--', lw=1.5)
plt.title(f'compounds (0.4+): {pcc_inst_final:.4f} | Cov_compounds (0.22+): {pcc_cov_final:.4f}')
plt.xlabel('True logFC')
plt.ylabel('Predicted logFC')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('./results/lincs/pcc_random_split_0_final.png')

print(f"\n --- 最终结果 --- ")
print(f"Instance-level Pearson (你期待的 0.4+): {pcc_inst_final:.4f}")
print(f"Cov_compounds Pearson (你期待的 0.22+): {pcc_cov_final:.4f}")
