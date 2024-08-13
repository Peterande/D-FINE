import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font family to Nimbus Roman No9L
rcParams['font.family'] = 'Times New Roman'
# /home/pengys/miniconda3/envs/dfine/lib/python3.9/site-packages/matplotlib/mpl-data/matplotlibrc
# 数据示例
networks = ['Ours', 'YOLOv6-v3.0', 'YOLOv8', 'YOLOv10', 'YOLOV9',
            'DFINE+HGNetV2', 'DFINE', 'DFINEv2', 'DFINE (Objects365)', 'Gold-YOLO']


latency_data = [
    [1.09, 1.41, 1.84, 2.48, 3.19],  # Ours
    # [6.60, 6.65, 6.95, 7.33, 8.05],  # YOLOv5
    [1.36, 1.53, 2.00, 2.62],  # YOLOv6
    # [7.24, 7.85],  # YOLOv7
    [2.11, 2.24, 2.70, 3.19, 3.86],  # YOLOv8
    [0.92, 1.27, 1.53, 2.00, 2.18, 2.85],  # YOLOv10
    [2.6, 2.98, 4.05],  # YOLOv9
    [2.55, 3.22],  # DFINE
    [1.41, 1.83, 1.84, 2.48, 3.19],  # DFINE
    [1.41, 1.83, 1.84, 2.48, 3.19],  # DFINEv2
    [1.41, 2.48, 3.19],  # DFINE+Obj365
    [1.72, 1.91, 2.43, 3.05],  # Gold-YOLO-
    # [0.77, 0.92, 1.83, 2.63, 7.90],  # LW-DETR
]
coco_ap_data = [
#   [B0  , B1  , B2* , B3* , B4* , B5* ]
    [47.0, 49.9, 52.9, 54.0, 55.3],  # Ours
    # [28.0, 37.4, 45.4, 49.0, 50.7],  # YOLOv5
    [37.5, 45.0, 50.0, 52.8],  # YOLOv6
    # [51.4, 53.1],  # YOLOv7
    [37.3, 44.9, 50.2, 52.9, 53.9],  # YOLOv8
    [38.5, 46.3, 51.1, 52.5, 53.2, 54.4],  # YOLOv10
    [51.4, 53.0, 55.6],  # YOLOv9
    [53.0, 54.8],  # DFINE*
    [46.5, 48.9, 51.3, 53.1, 54.3],  # DFINE
    [47.9, 49.9, 51.9, 53.4, 54.3],  # DFINEv2
    [49.2, 55.4, 56.2],  # DFINE+Obj365
    [39.9, 46.4, 51.1, 53.3],  # Gold-YOLO
    # [42.6, 48.0, 52.5, 56.1, 58.3]  # LW-DETR

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
    ax.plot(latency_data[i], coco_ap_data[i], color=colors[i], marker=markers[i], linestyle=linestyles[i], label=network, linewidth=2, markersize=8)

# Set axis labels and title
ax.set_xlabel('Latency (ms)', fontsize=16)
# ax.set_xlabel('Params (M)', fontsize=16)
ax.set_ylabel('COCO AP (%)', fontsize=16)
# ax.set_title('Performance vs Params', fontsize=18, fontweight='bold')
ax.set_title('Performance vs Latency', fontsize=18, fontweight='bold')
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
plt.savefig('/home/pengys/code/dfine_pytorch/vis/performance_vs_latency.png')
print("图表已保存为 'performance_vs_latency.png'")
