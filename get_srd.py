import pandas as pd
import sklearn.model_selection as sklm
import numpy as np
import math
import srd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('zz.csv', index_col=0, header=0)

# Set reference method
ref = 'max'
refVector = srd.calc_ref(df, ref)

"""
# Rank-transform
dfr = df.rank(axis=1)       #  axis=1 注意修改这一行
rVr = refVector.rank()
diffs = dfr.subtract(rVr, axis=0)
"""

df_aug = df.copy()
df_aug['Ideal'] = refVector  # 添加“理想方法”列
dfr = df_aug.rank(axis=1, ascending=False)  # 注意：指标越大越好，应设 ascending=False

# 抽出参考列的排名向量
rVr = dfr['Ideal']
dfr = dfr.drop(columns=['Ideal'])  # 去掉“理想”列，仅保留实际方法

# 对比：每个方法与“理想”的排名差异
diffs = dfr.subtract(rVr, axis=0)

srd_values = diffs.abs().sum()
#srd_values = np.sqrt((diffs ** 2).sum(axis=0))


# Compute max SRD and normalize
k = math.floor(len(df) / 2)
maxSRD = 2 * k**2 if len(df) % 2 == 0 else 2 * k * (k + 1)
c = srd_values / maxSRD * 100


# ... (之前的代码保持不变，直到计算 SRD 值部分)

# 对 SRD 值进行排序（从小到大）
srd_values_sorted = (srd_values.sort_values())/10
breakpoint()
# 使用排序后的值创建柱状图
fig, ax = plt.subplots()
my_colors = ['red','blue','green','purple','orange','brown','magenta','teal','violet','lime',
             'darkorange', 'darkorchid', 'darkred', 'darkseagreen', 'darkslateblue', 'darkslategrey',
             'darkturquoise', 'darkviolet','indianred', 'indigo','black']

# 创建柱状图（使用排序后的值）
bars = ax.bar(range(len(srd_values_sorted)), srd_values_sorted, 
              color=my_colors[:len(srd_values_sorted)])

# 设置x轴刻度标签（使用排序后的名称）
ax.set_xticks(range(len(srd_values_sorted)))
ax.set_xticklabels(srd_values_sorted.index, rotation=45, ha='right')

# 设置坐标轴范围
ax.set_xlim(left=-0.5, right=len(srd_values_sorted)-0.5)
ax.set_ylim(bottom=0, top=60, )
ax.set_yticks(np.arange(0, 61, 20))

# 创建图例（右上角）
#ax.legend(bars, srd_values_sorted.index, loc='upper right', bbox_to_anchor=(1.3, 1))

# 计算随机SRD值分布和显著性点
[x, y, XX1, Med, XX19] = srd.crrn(len(df))


# 调整显著性线的位置（因为x轴现在是排序后的顺序）
if XX1:
    # 找到XX1对应的位置
    xx1_pos = np.interp(XX1, [0, 100], [0, len(srd_values_sorted)-1])
    ax.axvline(x=xx1_pos, ymax=0.95, label='XX1', color='red', linestyle='--')
if Med:
    med_pos = np.interp(Med, [0, 100], [0, len(srd_values_sorted)-1])
    ax.axvline(x=med_pos, ymax=0.95, label='Med', color='red', linestyle='--')
if XX19:
    xx19_pos = np.interp(XX19, [0, 100], [0, len(srd_values_sorted)-1])
    ax.axvline(x=xx19_pos, ymax=0.95, label='XX19', color='red', linestyle='--')



# 添加第二y轴（分布图）
ax2 = ax.twinx()
# 调整分布图的x坐标以匹配排序后的柱状图
x_adj = np.linspace(0, len(srd_values_sorted)-1, len(x))
ax2.plot(x_adj, y, color='black')
ax2.set_ylabel('Rel. frequencies of SRD')
ax2.set_ylim(bottom=0)

ax.set_title('SRD results')
ax.set_xlabel('Methods (sorted by SRD value)')
ax.set_ylabel('SRD (%)')

fig.tight_layout()
plt.savefig('srd_plot_sorted.png', dpi=300, bbox_inches='tight')
plt.show()