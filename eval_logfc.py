import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import warnings
warnings.filterwarnings('ignore') # 忽略一些 groupby 时的警告

# ==========================================
# ⚙️ 核心配置区
# ==========================================
BASE_DIR = '/media/mldadmin/home/s125mdg35_08/PRnet baseline/PRnet/results/PRnet_Bio_En/'
PREFIX = 'random_split_0'  # 🔴 每次测新模型只改这里！例如 'vega_try2_50'

print(f"🚀 开始执行【原论文最高严谨标准】的 logFC 评估，目标模型: [{PREFIX}]...")

# 1. 自动拼接文件路径并加载数据
y_true_path = os.path.join(BASE_DIR, f'{PREFIX}_y_true_array.csv')
y_pre_path = os.path.join(BASE_DIR, f'{PREFIX}_y_pre_array.csv')
x_true_path = os.path.join(BASE_DIR, f'{PREFIX}_x_true_array.csv')
cov_drug_path = os.path.join(BASE_DIR, f'{PREFIX}_cov_drug_array.csv')

print("📂 正在加载预测数组...")
y_true = pd.read_csv(y_true_path, header=None).values
y_pre = pd.read_csv(y_pre_path, header=None).values
x_true = pd.read_csv(x_true_path, header=None).values

with open(cov_drug_path, 'r', encoding='utf-8') as f:
    cov_compounds_labels = [line.strip().strip('"').strip("'") for line in f.readlines()]

# 🔴 关键步骤：从 cov_compounds 提取 pure compounds (纯药物)
# PRnet 的 cov_drug 通常格式为 "CellLine_DrugName" (如 A549_Erlotinib)
# 我们使用 split('_')[-1] 取最后一部分作为纯药物名。如果你的格式不同，请修改这里！
compounds_labels = [label.split('_')[-1] for label in cov_compounds_labels]

# 对齐特征维度与样本数
if y_true.shape[1] > 978:
    y_true, y_pre, x_true = y_true[:, 1:], y_pre[:, 1:], x_true[:, 1:]

min_len = min(len(y_true), len(y_pre), len(x_true), len(cov_compounds_labels))
y_true, y_pre, x_true = y_true[:min_len], y_pre[:min_len], x_true[:min_len]
cov_compounds_labels = cov_compounds_labels[:min_len]
compounds_labels = compounds_labels[:min_len]

# ==========================================
# 2. 核心数学转换：计算纯 logFC
# ==========================================
print("🧮 正在计算细胞基线差值 (logFC)...")
delta_true = y_true - x_true
delta_pre = y_pre - x_true

df_true = pd.DataFrame(delta_true)
df_pre = pd.DataFrame(delta_pre)

# 将标签加入 DataFrame 以便 groupby
df_true['cov_compounds'] = cov_compounds_labels
df_true['compounds'] = compounds_labels
df_pre['cov_compounds'] = cov_compounds_labels
df_pre['compounds'] = compounds_labels

def calc_mean_pcc(df_t, df_p, group_col):
    """通用的按指定列 groupby 求均值并算 PCC 的函数"""
    # 丢掉另一个文本列，防止算均值时报错
    drop_cols = ['cov_compounds', 'compounds']
    
    mean_t = df_t.drop(columns=[c for c in drop_cols if c != group_col]).groupby(group_col).mean()
    mean_p = df_p.drop(columns=[c for c in drop_cols if c != group_col]).groupby(group_col).mean()
    
    pccs = []
    # 确保对齐索引
    common_idx = mean_t.index.intersection(mean_p.index)
    for idx in common_idx:
        r, _ = pearsonr(mean_t.loc[idx].values, mean_p.loc[idx].values)
        if not np.isnan(r):
            pccs.append(r)
    return np.mean(pccs), mean_t, mean_p

# ==========================================
# 指标 1: cov_compounds (论文指标：同细胞系+同药物求均值)
# ==========================================
print("📊 正在计算 Pearson of log(FC) in cov_compounds...")
pcc_cov, mean_t_cov, mean_p_cov = calc_mean_pcc(df_true, df_pre, 'cov_compounds')

# ==========================================
# 指标 2: compounds (论文指标：仅同药物，跨细胞系求大均值)
# ==========================================
print("📊 正在计算 Pearson of log(FC) in compounds...")
pcc_comp, mean_t_comp, mean_p_comp = calc_mean_pcc(df_true, df_pre, 'compounds')

# ==========================================
# 3. 论文级可视化输出
# ==========================================
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1: cov_compounds 散点 (选第一个组合为例)
sample_cov = mean_t_cov.index[0]
axes[0].scatter(mean_t_cov.loc[sample_cov], mean_p_cov.loc[sample_cov], alpha=0.7, s=20, color='#1f77b4')
axes[0].plot([-3, 3], [-3, 3], 'k--', lw=1.5)
axes[0].set_title(f'cov_compounds ({sample_cov})\nPCC = {pcc_cov:.4f}', fontsize=14, fontweight='bold')
axes[0].set_xlabel('True logFC', fontsize=12)
axes[0].set_ylabel('Predicted logFC', fontsize=12)
axes[0].set_xlim([-3, 3])
axes[0].set_ylim([-3, 3])

# 图2: compounds 散点 (选第一个药物为例)
sample_comp = mean_t_comp.index[0]
axes[1].scatter(mean_t_comp.loc[sample_comp], mean_p_comp.loc[sample_comp], alpha=0.7, s=20, color='#d62728')
axes[1].plot([-3, 3], [-3, 3], 'k--', lw=1.5)
axes[1].set_title(f'compounds ({sample_comp})\nPCC = {pcc_comp:.4f}', fontsize=14, fontweight='bold')
axes[1].set_xlabel('True logFC', fontsize=12)
axes[1].set_ylabel('Predicted logFC', fontsize=12)
axes[1].set_xlim([-3, 3])
axes[1].set_ylim([-3, 3])

plt.tight_layout()
save_path = os.path.join(BASE_DIR, f'{PREFIX}_Paper_Metrics.png')
plt.savefig(save_path, dpi=300)

print(f"\n=============================================")
print(f"🏆 --- {PREFIX} 论文级评估结果 --- 🏆")
print(f"=============================================")
print(f"1. Pearson in cov_compounds (中观 - 细胞系特定药效): {pcc_cov:.4f}")
print(f"2. Pearson in compounds     (宏观 - 普适性药效指纹): {pcc_comp:.4f}")
print(f"评估散点图已生成并保存至: {save_path}")
print(f"=============================================\n")