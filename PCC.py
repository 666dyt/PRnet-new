import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ==========================================
# 第一步：自动扫描定位你的 CSV 文件
# ==========================================
def find_csv_file(keyword):
    for root, dirs, files in os.walk('./results/'):
        for file in files:
            if keyword in file and file.endswith('.csv'):
                return os.path.join(root, file)
    raise FileNotFoundError(f"在 ./results/ 及其子目录下找不到包含 '{keyword}' 的 CSV 文件！")

x_true_path = find_csv_file('x_true_array')
y_true_path = find_csv_file('y_true_array')
y_pred_path = find_csv_file('y_pre_array')
drugs_path = find_csv_file('cov_drug_array')

print("✅ 成功定位到 CSV 文件：")
print(f"x_true 路径 -> {x_true_path}")
print(f"y_true 路径 -> {y_true_path}")
print(f"y_pred 路径 -> {y_pred_path}")
print(f"drugs  路径 -> {drugs_path}\n")

# ==========================================
# 第二步：智能读取（区分数值矩阵与文本标签）
# ==========================================
def load_numeric_csv(path):
    """安全读取数值型的 CSV 矩阵"""
    df = pd.read_csv(path)
    if df.columns[0].startswith('Unnamed'):
        df = df.iloc[:, 1:]
    return df.values

def load_drug_labels(path):
    """物理防弹读取：专门对付名称里带逗号的药物标签"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    
    cleaned_drugs = []
    for i, line in enumerate(lines):
        # 跳过表头 (如 'Unnamed' 或 '0' 或 'drug')
        if i == 0 and ('Unnamed' in line or '0' == line or 'drug' in line.lower()):
            continue
        
        # 如果保存时带了数字索引 (如 "0,Vorinostat")，剥离前面的数字
        if ',' in line:
            parts = line.split(',', 1) # 只切分第一个逗号
            if parts[0].isdigit():
                line = parts[1]
                
        # 剥离 Pandas 保存时可能自动加上的双引号
        line = line.strip('"').strip("'")
        cleaned_drugs.append(line)
        
    return np.array(cleaned_drugs)

print("正在读取庞大的 CSV 矩阵，这可能需要几秒钟...")
x_true = load_numeric_csv(x_true_path)
y_true = load_numeric_csv(y_true_path)
y_pred = load_numeric_csv(y_pred_path)

# 使用专属文本读取器加载药物标签
drugs = load_drug_labels(drugs_path)

# 确保把一维的药物标签展平为一维数组
if drugs.ndim > 1:
    drugs = drugs.flatten()

# 安全校验：防止标签数量与细胞数量对不上
min_len = min(len(x_true), len(drugs))
x_true = x_true[:min_len]
y_true = y_true[:min_len]
y_pred = y_pred[:min_len]
drugs = drugs[:min_len]

print(f"✅ 成功加载 {len(drugs)} 个细胞的测试数据。正在计算 logFC PCC...\n")

# ==========================================
# 第三步：严格对齐 Nature 论文的 logFC PCC 计算
# ==========================================
# 1. 计算对数折叠变化（药物效应向量）
true_logfc = y_true - x_true
pred_logfc = y_pred - x_true

# 2. 按化合物 (drug) 分组并求 978 个基因的平均值
df_true = pd.DataFrame(true_logfc)
df_true['drug'] = drugs
mean_true_logfc_per_drug = df_true.groupby('drug').mean()

df_pred = pd.DataFrame(pred_logfc)
df_pred['drug'] = drugs
mean_pred_logfc_per_drug = df_pred.groupby('drug').mean()

# 3. 遍历计算 PCC
pcc_list = []
unique_drugs = mean_true_logfc_per_drug.index

for drug in unique_drugs:
    vec_true = mean_true_logfc_per_drug.loc[drug].values
    vec_pred = mean_pred_logfc_per_drug.loc[drug].values
    
    # 排除方差为0的异常情况，防止 pearsonr 报错
    if np.std(vec_true) > 0 and np.std(vec_pred) > 0:
        r, _ = pearsonr(vec_true, vec_pred)
        pcc_list.append(r)

final_paper_pcc = np.mean(pcc_list)

print("="*45)
print(f"📊 严格对齐 Nature 论文的最终指标结果:")
print(f"➡️ Pearson of log(FC) in compounds: {final_paper_pcc:.4f}")
print("="*45)

# ==========================================
# 第四步：附赠高颜值橙色 PCC 分布图
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")

plt.figure(figsize=(9, 6))
sns.histplot(pcc_list, bins=30, kde=True, color="#ff9f43", edgecolor="white", stat="density", alpha=0.7)
plt.axvline(final_paper_pcc, color='red', linestyle='--', linewidth=2.5, label=f'Mean PCC = {final_paper_pcc:.4f}')
plt.title("Pearson Correlation of log(FC) in Compounds", fontsize=14, pad=12)
plt.xlabel("Pearson Correlation Coefficient (PCC)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()
sns.despine()
plt.tight_layout()

# 将图片保存在自动找到的 x_true 文件所在的同级目录中
save_dir = os.path.dirname(x_true_path)
save_img_path = os.path.join(save_dir, 'Paper_Aligned_PCC_Distribution.png')
plt.savefig(save_img_path, dpi=300)
print(f"✅ 高清分布图已保存至: {save_img_path}")