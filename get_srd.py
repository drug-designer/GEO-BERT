import pandas as pd
import sklearn.model_selection as sklm
import numpy as np
import math
import srd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('srd_geobert.csv', index_col=0, header=0)

# Set reference method
ref = 'max'
refVector = srd.calc_ref(df, ref)

"""
# Rank-transform
dfr = df.rank(axis=1)       
rVr = refVector.rank()
diffs = dfr.subtract(rVr, axis=0)
"""

df_aug = df.copy()
df_aug['Ideal'] = refVector  
dfr = df_aug.rank(axis=1, ascending=False)  

rVr = dfr['Ideal']
dfr = dfr.drop(columns=['Ideal'])  

diffs = dfr.subtract(rVr, axis=0)
srd_values = diffs.abs().sum()
#srd_values = np.sqrt((diffs ** 2).sum(axis=0))

# Compute max SRD and normalize
k = math.floor(len(df) / 2)
maxSRD = 2 * k**2 if len(df) % 2 == 0 else 2 * k * (k + 1)
c = srd_values / maxSRD * 100
srd_values_sorted = (srd_values.sort_values())/10

fig, ax = plt.subplots()
my_colors = ['red','blue','green','purple','orange','brown','magenta','teal','violet','lime',
             'darkorange', 'darkorchid', 'darkred', 'darkseagreen', 'darkslateblue', 'darkslategrey',
             'darkturquoise', 'darkviolet','indianred', 'indigo','black']

bars = ax.bar(range(len(srd_values_sorted)), srd_values_sorted, 
              color=my_colors[:len(srd_values_sorted)])

ax.set_xticks(range(len(srd_values_sorted)))
ax.set_xticklabels(srd_values_sorted.index, rotation=45, ha='right')

ax.set_xlim(left=-0.5, right=len(srd_values_sorted)-0.5)
ax.set_ylim(bottom=0, top=60, )
ax.set_yticks(np.arange(0, 61, 20))

#ax.legend(bars, srd_values_sorted.index, loc='upper right', bbox_to_anchor=(1.3, 1))

[x, y, XX1, Med, XX19] = srd.crrn(len(df))

if XX1:
    xx1_pos = np.interp(XX1, [0, 100], [0, len(srd_values_sorted)-1])
    ax.axvline(x=xx1_pos, ymax=0.95, label='XX1', color='red', linestyle='--')
if Med:
    med_pos = np.interp(Med, [0, 100], [0, len(srd_values_sorted)-1])
    ax.axvline(x=med_pos, ymax=0.95, label='Med', color='red', linestyle='--')
if XX19:
    xx19_pos = np.interp(XX19, [0, 100], [0, len(srd_values_sorted)-1])
    ax.axvline(x=xx19_pos, ymax=0.95, label='XX19', color='red', linestyle='--')


ax2 = ax.twinx()
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
