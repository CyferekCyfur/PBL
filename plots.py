import numpy as np
import matplotlib.pyplot as plt

# Data for YOLOStereo3D
yolo_bbox = [97.14, 86.83, 69.32]
yolo_bev = [60.67, 43.86, 33.72]
yolo_3d = [55.18, 39.71, 30.26]
yolo_aos = [96.03, 84.67, 67.42]

# Data for SMOKE
smoke_bbox = [94.4864, 86.2933, 79.2313]
smoke_bev = [58.6409, 43.5922, 39.2054]
smoke_3d = [53.6863, 39.8296, 34.5969]
smoke_aos = [91.59, 82.51, 74.49]

# Data for MonoCon12_17
mono_bbox = [94.8223, 85.6043, 78.3067]
mono_bev = [58.3053, 41.9833, 36.3988]
mono_3d = [53.1444, 37.1733, 32.6952]
mono_aos = [91.98, 81.90, 73.64]

# Plotting the bar charts
difficulty = ['easy', 'moderate', 'hard']
colors = ['green', 'blue', 'red']  # Define colors for each method

plt.figure(figsize=(16, 6)) # To make the figure bigger

# Plotting bbox AP
plt.subplot(2, 2, 1)
bar_width = 0.2
plt.bar(np.arange(len(difficulty)), yolo_bbox, color=colors[0], width=bar_width, label='YOLOStereo3D')
plt.bar(np.arange(len(difficulty)) + bar_width, smoke_bbox, color=colors[1], width=bar_width, label='SMOKE')
plt.bar(np.arange(len(difficulty)) + 2 * bar_width, mono_bbox, color=colors[2], width=bar_width, label='MonoCon12_17')
plt.title('Detection Method Comparison - bbox AP')
plt.xlabel('Difficulty')
plt.ylabel('Average Precision (%)')
plt.xticks(np.arange(len(difficulty)) + bar_width, difficulty)
plt.ylim(0, 100)
plt.legend()

# Plotting bev AP
plt.subplot(2, 2, 2)
plt.bar(np.arange(len(difficulty)), yolo_bev, color=colors[0], width=bar_width, label='YOLOStereo3D')
plt.bar(np.arange(len(difficulty)) + bar_width, smoke_bev, color=colors[1], width=bar_width, label='SMOKE')
plt.bar(np.arange(len(difficulty)) + 2 * bar_width, mono_bev, color=colors[2], width=bar_width, label='MonoCon12_17')
plt.title('Detection Method Comparison - bev AP')
plt.xlabel('Difficulty')
plt.ylabel('Average Precision (%)')
plt.xticks(np.arange(len(difficulty)) + bar_width, difficulty)
plt.ylim(0, 100)
plt.legend()

# Plotting 3d AP
plt.subplot(2, 2, 3)
plt.bar(np.arange(len(difficulty)), yolo_3d, color=colors[0], width=bar_width, label='YOLOStereo3D')
plt.bar(np.arange(len(difficulty)) + bar_width, smoke_3d, color=colors[1], width=bar_width, label='SMOKE')
plt.bar(np.arange(len(difficulty)) + 2 * bar_width, mono_3d, color=colors[2], width=bar_width, label='MonoCon12_17')
plt.title('Detection Method Comparison - 3d AP')
plt.xlabel('Difficulty')
plt.ylabel('Average Precision (%)')
plt.xticks(np.arange(len(difficulty)) + bar_width, difficulty)
plt.ylim(0, 100)
plt.legend()

# Plotting aos AP
plt.subplot(2, 2, 4)
plt.bar(np.arange(len(difficulty)), yolo_aos, color=colors[0], width=bar_width, label='YOLOStereo3D')
plt.bar(np.arange(len(difficulty)) + bar_width, smoke_aos, color=colors[1], width=bar_width, label='SMOKE')
plt.bar(np.arange(len(difficulty)) + 2 * bar_width, mono_aos, color=colors[2], width=bar_width, label='MonoCon12_17')
plt.title('Detection Method Comparison - aos AP')
plt.xlabel('Difficulty')
plt.ylabel('Average Precision (%)')
plt.xticks(np.arange(len(difficulty)) + bar_width, difficulty)
plt.ylim(0, 100)
plt.legend()

plt.tight_layout() # To avoid overlapping of subplots
plt.show()