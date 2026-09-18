#!/usr/bin/env python3
import numpy as np
from PIL import Image
import glob

def read_frame(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        count = int(lines[0].strip())
        data = np.fromstring(''.join(lines[1:count+1]), sep=' ', dtype=np.float64)
        return data.reshape(-1, 3)

def load_all_frames():
    frames = []
    frame_files = sorted(glob.glob("frame_*.txt"))
    for frame_file in frame_files:
        data = read_frame(frame_file)
        if len(data) > 0:
            frames.append(data)
    return frames

def create_animation(pixel_size=0.05, align_grid=True):
    frames = load_all_frames()
    
    if len(frames) == 0:
        print("No frames found!")
        return
    
    print(f"Loaded {len(frames)} frames")
    
    all_x = np.concatenate([f[:, 0] for f in frames])
    all_z = np.concatenate([f[:, 2] for f in frames])
    x_min, x_max = all_x.min(), all_x.max()
    z_min, z_max = all_z.min(), all_z.max()
    
    if align_grid:
        x_min = np.floor(x_min / pixel_size) * pixel_size
        x_max = np.ceil(x_max / pixel_size) * pixel_size
        z_min = np.floor(z_min / pixel_size) * pixel_size
        z_max = np.ceil(z_max / pixel_size) * pixel_size
    
    print(f"Bounds: X=[{x_min:.2f},{x_max:.2f}], Z=[{z_min:.2f},{z_max:.2f}]")
    
    x_range = x_max - x_min
    z_range = z_max - z_min
    
    if align_grid:
        width = int(np.round(x_range / pixel_size))
        height = int(np.round(z_range / pixel_size))
    else:
        width = int(np.ceil(x_range / pixel_size))
        height = int(np.ceil(z_range / pixel_size))
    
    print(f"Resolution: {width}x{height} (pixel_size={pixel_size})")
    
    duration = 30
    
    x_scale = 1.0 / pixel_size
    z_scale = 1.0 / pixel_size
    
    x_offset = -x_min * x_scale
    z_offset = z_max * z_scale
    
    images = []
    for i, frame_data in enumerate(frames):
        sx = (frame_data[:, 0] * x_scale + x_offset).astype(int)
        sz = (z_offset - frame_data[:, 2] * z_scale).astype(int)
        
        mask = (sx >= 0) & (sx < width) & (sz >= 0) & (sz < height)
        valid_x, valid_z = sx[mask], sz[mask]
        
        grid = np.zeros((height, width, 3), dtype=np.uint8) + 255
        grid[valid_z, valid_x] = [50, 100, 255]
        
        img = Image.fromarray(grid, mode='RGB')
        images.append(img)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(frames)} frames")
    
    output_file = 'particle_simulation.gif'
    print(f"Saving animation to {output_file}...")
    images[0].save(output_file, save_all=True, append_images=images[1:], 
                   duration=duration, loop=0, optimize=True)
    print(f"Animation saved to {output_file}")

if __name__ == "__main__":
    print("Creating animation...")
    create_animation(pixel_size=0.025)
