import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font family to Nimbus Roman No9L
rcParams['font.family'] = 'Times New Roman'
# /home/pengys/miniconda3/envs/rtdetr/lib/python3.9/site-packages/matplotlib/mpl-data/matplotlibrc
# 数据示例
networks = ['Ours', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv10', 
            'RT-DETR+HGNetV2', 'RT-DETR', 'RT-DETR+Objects365', 'Gold-YOLO', 'LW-DETR']

param_data = [
    [10, 12, 17, 24, 31, 61],  # Ours
    [1.9, 7.2, 21.2, 46.5, 86.7],  # YOLOv5
    [4.7, 18.5, 34.9, 59.6],  # YOLOv6
    [36, 71],  # YOLOv7
    [3.2, 11.2, 25.6, 43.7, 68.2],  # YOLOv8
    [2.3, 7.2, 15.4, 19.1, 24.4, 29.5],  # YOLOv10
    [32, 67],  # RT-DETR+HG
    [20, 31, 36, 42, 76],  # RT-DETR
    [20, 42, 76],  # RT-DETR+Obj365
    [5.6, 21.5, 41.3, 75.1],  # Gold-YOLO
    [12.1, 14.6, 28.2, 46.8, 118], # LW-DETR
    ]


# latency_data = [

# ]

coco_ap_data = [

    # [50.2, 52.6, 54.3, 55.4, 56.9, 57.9],  # Ours+Obj365
    [44.9, 47.3, 50.1, 51.8, 54.0, 55.1],  # Ours
    [28.0, 37.4, 45.4, 49.0, 50.7],  # YOLOv5
    [37.5, 45.0, 50.0, 52.8],  # YOLOv6
    [51.4, 53.1],  # YOLOv7
    [37.3, 44.9, 50.2, 52.9, 53.9],  # YOLOv8
    [38.5, 46.3, 51.1, 52.5, 53.2, 54.4],  # YOLOv10
    [53.0, 54.8],  # RT-DETR+HG
    [46.5, 48.9, 51.3, 53.1, 54.3],  # RT-DETR
    [49.2, 55.4, 56.2],  # RT-DETR+Obj365
    [39.9, 46.4, 51.1, 53.3],  # Gold-YOLO
    [42.6, 48.0, 52.5, 56.1, 58.3], # LW-DETR

]

# Define the color palette with more prominent and less prominent colors
primary_colors = ['green', 'olive']  # Prominent colors for Ours and Ours+Obj365
secondary_colors = ['brown', 'purple', 'gray', 'blue', 'orange', 'navy', 'salmon', 'red', 'cyan', 'pink', 'olive']

# Ensure secondary_colors length matches required
secondary_colors = secondary_colors[:len(networks) - len(primary_colors)]

# Define markers
markers = ['^', 's', '*', 'P', 'X', 'D', 'v', 'o', 'x', 'p', 'd', 'h']

# Assign colors to each network, with Ours and Ours+Obj365 being prominent
colors = primary_colors + secondary_colors[:len(networks) - len(primary_colors)]
linestyles = ['-', '-', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--']

# Create chart
fig, ax = plt.subplots(figsize=(8, 8))  # Modify to be square

# Plot each line
for i, network in enumerate(networks):
    ax.plot(param_data[i], coco_ap_data[i], color=colors[i], marker=markers[i], linestyle=linestyles[i], label=network, linewidth=2, markersize=8)

# Set axis labels and title
ax.set_xlabel('Params (M)', fontsize=16)
# ax.set_xlabel('Params (M)', fontsize=16)
ax.set_ylabel('COCO AP (%)', fontsize=16)
# ax.set_title('Performance vs Params', fontsize=18, fontweight='bold')
ax.set_title('Performance vs Params.', fontsize=18, fontweight='bold')
# Set y-axis range
ax.set_ylim(43, 60)  # Start y-axis from 43

# Add vertical arrow
# arrowprops = dict(facecolor='black', arrowstyle='->', lw=1.5)
# ax.annotate('+Obj365', xy=(4.76, 55.8), xytext=(4.76, 54.0),
#             arrowprops=arrowprops,
#             fontsize=12, fontweight='bold')

# Set legend
ax.legend(fontsize=10, loc='lower right')

# Set grid lines
# ax.grid(True, linestyle='--', alpha=0.6)

# Save chart to file
fig.tight_layout()
plt.savefig('/home/pengys/code/rtdetrv2_pytorch/vis/performance_vs_params.png')
print("图表已保存为 'performance_vs_params.png'")
