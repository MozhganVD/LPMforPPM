from scipy import stats
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import Orange

HO = [6, 5, 6, 6, 5, 4, 5, 5, 3]
HO_b = [4, 4, 2, 2, 3, 3, 3, 3, 5]
HO_F = [5, 3, 3, 3, 4, 5, 4, 4, 4]
EMb = [2, 2, 5, 5, 2, 2, 2, 2, 1]
EMb_LA = [1, 1, 1, 1, 1, 1, 1, 1, 2]
EMb_L = [3, 6, 4, 4, 6, 6, 6, 6, 6]

print(stats.friedmanchisquare(HO, HO_F, HO_b, EMb, EMb_L, EMb_LA))

# Combine three groups into one array
data = np.array([HO, HO_F, HO_b, EMb, EMb_LA, EMb_L])

# Conduct the Nemenyi post-hoc test
print(sp.posthoc_nemenyi_friedman(data.T))

# input data
HO = 2.667
WHO_B = 1.333
WHO_F = 2

EMb_Act = 2.111
EMb_ActLPMs = 1.111
EMb_LPMs = 2.778

# names = ['HO', 'WHO_B', 'WHO_F', 'EMb_Act', 'EMb_ActLPMs', 'EMb_LPMs']
#
# avranks = [HO, WHO_B, WHO_F, EMb_Act, EMb_ActLPMs, EMb_LPMs]

names = ['EMb_Act', 'EMb_ActLPMs', 'EMb_LPMs']
avranks = [EMb_Act, EMb_ActLPMs, EMb_LPMs]

cd = Orange.evaluation.compute_CD(avranks, 9, alpha="0.05", test="nemenyi")  # tested on 14 datasets
print("CD ", cd)

Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.show()

# limits = (6,1)
#
# fig, ax = plt.subplots(figsize=(5, 2))
# plt.subplots_adjust(left=0.2, right=0.8)
#
# # set up plot
# ax.set_xlim(limits)
# ax.set_ylim(0,1)
# ax.spines['top'].set_position(('axes', 0.6))
# #ax.xaxis.tick_top()
# ax.xaxis.set_ticks_position('top')
# ax.yaxis.set_visible(False)
# for pos in ["bottom", "left", "right"]:
#     ax.spines[pos].set_visible(False)
#
# # CD bar
# ax.plot([limits[0],limits[0]-cd], [.9,.9], color="k")
# ax.plot([limits[0],limits[0]], [.9-0.03,.9+0.03], color="k")
# ax.plot([limits[0]-cd,limits[0]-cd], [.9-0.03,.9+0.03], color="k")
# ax.text(limits[0]-cd/2., 0.92, "CD", ha="center", va="bottom")
#
# # annotations
# bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
# arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=90")
# kw = dict(xycoords='data',textcoords="axes fraction",
#           arrowprops=arrowprops, bbox=bbox_props, va="center")
# ax.annotate("C4.5", xy=(c, 0.6), xytext=(0,0.25),ha="right",  **kw)
# ax.annotate("C4.5+cf", xy=(ccf, 0.6), xytext=(0,0),ha="right",  **kw)
# ax.annotate("C4.5+m+cf", xy=(cmcf, 0.6), xytext=(1.,0.25),ha="left",  **kw)
# ax.annotate("C4.5+m", xy=(cm, 0.6), xytext=(1.,0),ha="left",  **kw)
#
# #bars
# ax.plot([ccf,c],[0.55,0.55], color="k", lw=3)
# ax.plot([ccf,cmcf],[0.48,0.48], color="k", lw=3)
#
# plt.show()
