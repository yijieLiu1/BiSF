import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ========================
# 数据准备
# ========================
sizes = [1000, 5000, 10000, 20000, 50000, 100000]
plt.rcParams['font.sans-serif'] = ['SimHei']        # 支持中文
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示问题

# 密文生成与发送
cipher_agg_ms = [76, 56, 61, 77, 58, 67]

# 边缘节点
# edge_sum_ms = [264, 521, 791, 1080, 1093+954, 1853+1747]
edge_sum_ms = [35, 35, 34, 34, 32, 31]
edge_mean_ms = [36, 35, 32, 35, 33, 33]
edge_var_ms  = [39, 40, 40, 37, 33, 36]

# 中心服务器
center_sum_ms  = [5, 9, 9, 11, 5, 9]
center_mean_ms = [9, 11, 10, 10, 7, 17]
center_var_ms  = [9, 16, 19, 12, 12, 10]
center_extreme_ms = [76, 56, 61, 77, 58, 67]

# 极值计算（两列：泛化 vs 全遍历）
edge_extreme_general_ms = [298, 76, 721, 696, 1189, 1730]
edge_extreme_general_cmp = [14, 2, 20, 24, 39, 53]
extreme_full_ms = [22172, 44035, 214438, 406691, 1029449, 1881390]
extreme_full_cmp = [998, 1998, 9998, 19998, 49998, 91896]



# ========================
# 图1：密文生成与发送（折线图）
# ========================
plt.figure(figsize=(7,5))
plt.plot(sizes, cipher_agg_ms, marker='o', color='royalblue')
plt.title("中心服务器极值计算", fontsize=13)
plt.xlabel("样本量（条）", fontsize=11)
plt.ylabel("耗时（毫秒）", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plot_cipher_agg.png", dpi=300)
plt.show()

# 通用参数
bar_width = 0.35
x = np.arange(len(sizes))

# ========================
# 图2：求和（柱状图）
# ========================
plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, edge_sum_ms, bar_width, label='边缘节点', color='tab:blue')
plt.bar(x + bar_width/2, center_sum_ms, bar_width, label='中心服务器', color='tab:orange')
plt.title("求和：边缘节点 vs 中心服务器", fontsize=13)
plt.xlabel("样本量（条）", fontsize=11)
plt.ylabel("耗时（毫秒）", fontsize=11)
plt.xticks(x, sizes)
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plot_sum_bar.png", dpi=300)
plt.show()

# ========================
# 图3：求均值（柱状图）
# ========================
plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, edge_mean_ms, bar_width, label='边缘节点', color='tab:blue')
plt.bar(x + bar_width/2, center_mean_ms, bar_width, label='中心服务器', color='tab:orange')
plt.title("求均值：边缘节点 vs 中心服务器", fontsize=13)
plt.xlabel("样本量（条）", fontsize=11)
plt.ylabel("耗时（毫秒）", fontsize=11)
plt.xticks(x, sizes)
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plot_mean_bar.png", dpi=300)
plt.show()

# ========================
# 图4：求方差（柱状图）
# ========================
plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, edge_var_ms, bar_width, label='边缘节点', color='tab:blue')
plt.bar(x + bar_width/2, center_var_ms, bar_width, label='中心服务器', color='tab:orange')
plt.title("求方差：边缘节点 vs 中心服务器", fontsize=13)
plt.xlabel("样本量（条）", fontsize=11)
plt.ylabel("耗时（毫秒）", fontsize=11)
plt.xticks(x, sizes)
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plot_var_bar.png", dpi=300)
plt.show()

# ========================
# 图5：求极值（泛化 vs 全遍历） + 比较次数（双轴折线图）
# ========================
fig, ax1 = plt.subplots(figsize=(8,5))

# 主y轴（耗时）
ax1.plot(sizes, edge_extreme_general_ms, 'o-', label='泛化标签（候选区间）', color='tab:blue')
ax1.plot(sizes, extreme_full_ms, 's-', label='全遍历比较', color='tab:orange')
ax1.set_xlabel("样本量（条）", fontsize=11)
ax1.set_ylabel("耗时（毫秒）", fontsize=11, color='black')
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, which="both", ls="--", alpha=0.6)

# 副y轴（比较次数）——使用黑色和绿色区分
ax2 = ax1.twinx()
ax2.plot(sizes, edge_extreme_general_cmp, 'd--', color='black', alpha=0.8, label='比较次数（泛化）')
ax2.plot(sizes, extreme_full_cmp, 'x--', color='green', alpha=0.8, label='比较次数（全遍历）')
ax2.set_ylabel("密文比较次数", fontsize=11, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.set_yscale('log')

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)

plt.title("求极值：泛化技术 vs 全遍历（含比较次数）", fontsize=13)
plt.tight_layout()
plt.savefig("plot_extreme.png", dpi=300)
plt.show()
# ========================
# 绘制紧凑型柱状图
# ========================
plt.figure(figsize=(7,5))
x = np.arange(len(sizes))  # 等距排列
bars = plt.bar(x, center_extreme_ms, color='tab:blue', edgecolor='black', width=0.5)

# 图表标题与标签
plt.title("中心服务器极值计算耗时", fontsize=13)
plt.xlabel("样本量（条）", fontsize=11)
plt.ylabel("耗时（毫秒）", fontsize=11)
plt.xticks(x, sizes)  # 使用等距横坐标标签

# 在柱形上标注具体数值
for bar, value in zip(bars, center_extreme_ms):
    plt.text(bar.get_x() + bar.get_width()/2, value + 1, f"{value}ms",
             ha='center', va='bottom', fontsize=10)

# 美化
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()

# 保存与展示
plt.savefig("center_extreme_bar_compact.png", dpi=300)
plt.show()
# ========================
# 数据表输出
# ========================
df = pd.DataFrame({
    "样本量": sizes,
    "密文聚合(ms)": cipher_agg_ms,
    "边缘-求和(ms)": edge_sum_ms,
    "中心-求和(ms)": center_sum_ms,
    "边缘-求均值(ms)": edge_mean_ms,
    "中心-求均值(ms)": center_mean_ms,
    "边缘-求方差(ms)": edge_var_ms,
    "中心-求方差(ms)": center_var_ms,
    "极值-泛化(ms)": edge_extreme_general_ms,
    "极值-泛化-比较次数": edge_extreme_general_cmp,
    "极值-全遍历(ms)": extreme_full_ms,
    "极值-全遍历-比较次数": extreme_full_cmp,
})

print("\n性能测试数据总览：")
print(df)
