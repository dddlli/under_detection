import matplotlib.pyplot as plt

# 设置字体格式

from matplotlib import rcParams

import matplotlib.patches as patches

size = 12
# 全局字体大小


# 设置英文字体


config = {
    "font.family": 'serif',
    "font.size": size,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],

}

rcParams.update(config)

# 设置中文宋体

fontcn = {
    'family': 'SimSun',
    'size': size}

label_size = size
text_size = size

# 数据

mAP = [
    45.99,
    47.32,
    46.41,
    44.60,
    44.55,
    46.52]

param = [
    11.32,
    10.22,
    10.19,
    19.46,
    28.14,
    11.72]

param2 = [i * 30 for i in param]

# FPS = [
#     20.9,
#     24.2,
#     20.5,
#     22.3,
#     21.6,
#     23.1
# ]

FPS = [
    37.41,
    38.17,
    39.96,
    57.23,
    69.90,
    47.02
]

# 绘制参数量标准

param_legend = [
    5,
    15,
    30,
    50,
    70]

param_legend = [i * 30 for i in param_legend]

param_x = [35,
           41.3,
           50.6,
           60.3,
           70.7]

param_y = [
    41.5,
    41.5,
    41.5,
    41.5,
    41.5]

param_color = [(0.45, 0.45, 0.45)] * 5

param_text = [
    '30G',
    '40G',
    '50G',
    '60G',
    '70G']

# 参数设置

lw = 2
ms = 15

my_text = [
    'Tood_Effb0',
    'Ours',
    'Tood_MobileV2',
    'Faster_MobileV2',
    'Faster_R18',
    'Retina_Effb0', ]

my_text2 = '-S'

color = [
    'C8',
    'C1',
    'C0',
    'C6',
    'C4',
    'C9']

# 绘制 mAP-Param


plt.figure()

plt.scatter(FPS, mAP, s=param2, color=color, alpha=0.6)

plt.scatter(param_x, param_y, s=param_legend, color=param_color, alpha=0.2)

# 绘制矩形框

ax = plt.gca()

rect = patches.Rectangle(xy=(32.6, 40.5), width=42, height=2.8, linewidth=0.5, linestyle='-', fill=False,
                         edgecolor='gray')

ax.add_patch(rect)

# 添加数学公式标签


plt.ylabel('$M_\mathrm{mAP}$ (%)', fontsize=label_size)

plt.xlabel('$N_\mathrm{GFlops}$ (gflops)', fontsize=label_size)

plt.xlim([30, 80])

plt.ylim([40, 50])

# 添加方法名


plt.text(FPS[0] - 3.5, mAP[0] - 0.8, my_text[0], color="k", fontsize=text_size)

plt.text(FPS[1] - 2, mAP[1] + 0.5, my_text[1], color="k", fontsize=text_size)

plt.text(FPS[2] - 5.5, mAP[2], my_text[2], color="k", fontsize=text_size)

plt.text(FPS[3] - 5, mAP[3] + 0.8, my_text[3], color="k", fontsize=text_size)

plt.text(FPS[4] - 4, mAP[4] + 1, my_text[4], color="k", fontsize=text_size)

plt.text(FPS[5] - 2.2, mAP[5] + 0.5, my_text[5], color="k", fontsize=text_size)

# plt.text(FPS[6] - 9, mAP[6] + 0.8, my_text[6], color="k", fontdict=fontcn)
#
# plt.text(FPS[6] + 4.5, mAP[6] + 0.8, my_text2, color="k", fontsize=text_size)
#
# plt.text(FPS[7] - 9, mAP[7] + 0.8, my_text[7], color="k", fontdict=fontcn)
#
# plt.text(FPS[7] + 4.5, mAP[7] + 0.8, my_text2, color="k", fontsize=text_size)

# 添加参数量标准大小


plt.text(param_x[0] - 1.5, 42.5, param_text[0], color="k", fontsize=text_size)

plt.text(param_x[1] - 1.8, 42.5, param_text[1], color="k", fontsize=text_size)

plt.text(param_x[2] - 1.8, 42.5, param_text[2], color="k", fontsize=text_size)

plt.text(param_x[3] - 1.8, 42.5, param_text[3], color="k", fontsize=text_size)

plt.text(param_x[4] - 1.8, 42.5, param_text[4], color="k", fontsize=text_size)

plt.grid(linestyle='--')

plt.savefig('map_gflops.svg', dpi=300, bbox_inches='tight')

plt.show()
