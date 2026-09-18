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

def plot_multi_mesh_hit_points(output_file):
    mesh_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    mesh_labels = []

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    all_points = []

    mesh_id = 0
    while True:
        filename = f"hit_data_mesh_{mesh_id}.txt"
        try:
            data, count = read_hit_data(filename)
            if count == 0:
                mesh_id += 1
                continue

            points = data

            all_points.extend(points)

            color = mesh_colors[mesh_id % len(mesh_colors)]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=color, s=1, alpha=0.7, label=f'Mesh {mesh_id}')

            mesh_labels.append(f'Mesh {mesh_id} ({count} hits)')
            mesh_id += 1
        except FileNotFoundError:
            break

    if not all_points:
        print("No hit data found!")
        return

    all_points = np.array(all_points)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Multi-Mesh Hit Points ({len(all_points)} points total)')
    ax.legend()

    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
    ax.set_zlim(z_center - max_range / 4, z_center + max_range / 4)

    ax.set_box_aspect([1, 1, 0.5])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")

    for label in mesh_labels:
        print(label)

if __name__ == "__main__":
    plot_multi_mesh_hit_points("multi_mesh_scatter.png")
