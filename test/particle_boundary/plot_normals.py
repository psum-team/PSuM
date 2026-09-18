#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_hit_data(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        count = int(lines[0].strip())
        for line in lines[1:]:
            values = list(map(float, line.strip().split()))
            data.append(values)
    return np.array(data), count

def plot_hit_points_with_normals(data, output_file):
    points = data[:, :3]
    normals = data[:, 3:]
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=20, alpha=0.6, label='Hit Points')
    
    scale = np.max(np.abs(points)) * 0.04
    if scale == 0:
        scale = 1.0
    
    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
              normals[:, 0], normals[:, 1], normals[:, 2],
              length=scale, arrow_length_ratio=0.5, normalize=True, color='r', alpha=0.7, label='Normals')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Hit Points and Normals ({len(points)} points)')
    ax.legend()

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
    ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)

    ax.set_box_aspect([1, 1, 1])

    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    data, count = read_hit_data("hit_data.txt")
    plot_hit_points_with_normals(data, "hit_normals.png")
