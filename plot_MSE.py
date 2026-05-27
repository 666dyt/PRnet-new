import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 第一步：读取日志并折算为 150 个 Epoch
# ==========================================
log_file_path = './checkpoint/bio_results/random_split_0_loss_comb.csv' # 请确保这里的路径是对的

df = pd.read_csv(log_file_path)
if df.columns[0].startswith('Unnamed'):
    df = df.rename(columns={df.columns[0]: 'Step'})

# 自动计算每个 Epoch 包含多少个 Step
total_steps = len(df)
total_epochs = 150
steps_per_epoch = total_steps // total_epochs

# 给每一行数据打上对应的 Epoch 标签 (1 到 150)
df['Epoch'] = (df.index // steps_per_epoch) + 1
df.loc[df['Epoch'] > total_epochs, 'Epoch'] = total_epochs # 防止除法余数越界

# 按 Epoch 分组，计算每个 Epoch 的平均 Loss
df_epoch = df.groupby('Epoch')['Loss_PGM'].mean().reset_index()

# ==========================================
# 第二步：智能计算 Y 轴显示范围 (避开初始巨大峰值)
# ==========================================
# 我们忽略前 3 个 Epoch 的极高初始值，去寻找真实收敛阶段的最大值和最小值
zoom_start_epoch = 3 
y_max = df_epoch[df_epoch['Epoch'] > zoom_start_epoch]['Loss_PGM'].max()
y_min = df_epoch['Loss_PGM'].min()

# 上下各留 10% 的空白，让图表更好看
padding = (y_max - y_min) * 0.1
ylim_top = y_max + padding
ylim_bottom = y_min - padding

# ==========================================
# 第三步：绘制高清放大的折线图
# ==========================================
sns.set_theme(style="ticks", context="talk")
plt.rcParams['font.sans-serif'] = ['Arial']

plt.figure(figsize=(10, 6))

# 画线
sns.lineplot(data=df_epoch, x='Epoch', y='Loss_PGM', color='red', linewidth=2.5)

# 核心操作：放大 Y 轴！
plt.ylim(ylim_bottom, ylim_top)

# 设置标题和标签
plt.title('Training Loss over Epochs (Averaged & Zoomed)', fontsize=16, pad=15)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('PGM Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

sns.despine()
plt.tight_layout()

# 保存图片
save_path = './results/Training_Loss_Curve_Fixed.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"✅ 修正后的放大版学习曲线已保存至: {save_path}")