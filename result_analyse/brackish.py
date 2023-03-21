import matplotlib.pyplot as plt

# 设置字体格式

from matplotlib import rcParams

from matplotlib.ticker import MultipleLocator

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


fontcn = {'family': 'SimSun',
          'size': size}

label_size = size

text_size = size

# 数据1

baseline_mAP = [
    36.4,
    45.4,
    53.9,
    57.2,
    58.0]

baseline_param = [
    1.79,
    7.07,
    20.95,
    46.2,
    86.2]

map1 = [81.18]
param1 = [11.32]
map2 = [82.88]
param2 = [10.22]
map3 = [82.62]
param3 = [10.19]
map4 = [79.76]
param4 = [19.46]
map5 = [80.81]
param5 = [28.14]
map6 = [81.20]
param6 = [11.72]

# 数据2


my_mAP = [
    39.9,
    50.6,
    56.2,
    58.9,
    59.8]

my_param = [
    1.55,
    6.09,
    18.8,
    42.7,
    81.1]

# 参数设置


lw = 2

ms = 8

my_text = [
    'N',
    'S',
    'M',
    'L',
    'X']

# 绘制 mAP-Param

plt.figure(figsize=(6.4, 4.8))
plt.title('Brackish', loc='center', fontsize=label_size)

plt.plot(param1, map1, label='Tood_Effb0', c='C8', lw=lw, marker='o', markersize=ms, ls='-')
plt.plot(param2, map2, label='Ours', c='r', lw=lw, marker='*', markersize=ms, ls='-')
plt.plot(param3, map3, label='Tood_MobileV2', c='C0', lw=lw, marker='p', markersize=ms, ls='-')
plt.plot(param4, map4, label='Faster_MobileV2', c='C6', lw=lw, marker='8', markersize=ms, ls='-')
plt.plot(param5, map5, label='Faster_R18', c='C4', lw=lw, marker='h', markersize=ms, ls='-')
plt.plot(param6, map6, label='Retina_Effb0', c='C9', lw=lw, marker='^', markersize=ms, ls='-')

plt.legend(loc='lower right', prop=fontcn)

plt.ylabel('$M_\mathrm{mAP}$ (%)', fontsize=label_size)

plt.xlabel('$N_\mathrm{Param}$ (M)', fontsize=label_size)

# 设置坐标轴间隔

x_major_locator = MultipleLocator(10)

y_major_locator = MultipleLocator(3.5)

ax = plt.gca()

ax.xaxis.set_major_locator(x_major_locator)

ax.yaxis.set_major_locator(y_major_locator)

plt.xlim([0, 40])

plt.ylim([75, 85])

# plt.text(my_param[0] - 3.5, my_mAP[0] + 0.8, my_text[0], color="k", fontsize=text_size)
#
# plt.text(my_param[1] - 2, my_mAP[1] + 0.8, my_text[1], color="k", fontsize=text_size)
#
# plt.text(my_param[2] - 2, my_mAP[2] + 0.8, my_text[2], color="k", fontsize=text_size)
#
# plt.text(my_param[3] - 3, my_mAP[3] + 1.0, my_text[3], color="k", fontsize=text_size)
#
# plt.text(my_param[4] - 2, my_mAP[4] + 0.8, my_text[4], color="k", fontsize=text_size)

plt.grid(linestyle='--')
plt.savefig('brackish.png', dpi=300, bbox_inches='tight')

plt.show()
