#!/usr/bin/env python3
"""Repository code-distribution visualizer.

Scans the source tree (src/ by default) and renders a treemap heatmap of
per-file line counts and sizes (PNG + optional HTML output, written to the
repository root).

Usage:
    python3 repo_heatmap.py

Dependencies (only needed to run the script):
    pip install pandas plotly matplotlib

PNG export additionally requires the Kaleido engine
(`pip install kaleido`); HTML output does not.
"""
import os
import re
from pathlib import Path
from typing import List, Optional, Set, Dict
import pandas as pd
import matplotlib.colors as mcolors

DEFAULT_IGNORE_EXTENSIONS: Set[str] = {
    '.plt', '.txt', '.png', '.jpg', '.jpeg', '.gif',
    '.svg', '.pdf', '.stl', '.mas', '.out', '.o'
}

DEFAULT_IGNORE_FOLDER_PATTERNS: List[str] = [
    r'\.git',
    r'__pycache__',
    r'third_party',
    r'heatmap.html'
]

def count_lines(filepath: str) -> int:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def get_file_size(filepath: str) -> int:
    try:
        return os.path.getsize(filepath)
    except Exception:
        return 0

def scan_files(
    root_dir: str,
    metric: str = 'lines',
    ignore_extensions: Optional[Set[str]] = None,
    ignore_folder_patterns: Optional[List[str]] = None
):
    if ignore_extensions is None:
        ignore_extensions = DEFAULT_IGNORE_EXTENSIONS
    if ignore_folder_patterns is None:
        ignore_folder_patterns = DEFAULT_IGNORE_FOLDER_PATTERNS

    compiled_patterns = [re.compile(p) for p in ignore_folder_patterns]
    data = []
    root_name = Path(root_dir).name
    file_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath_norm = dirpath.replace(os.sep, '/')
        if any(p.search(dirpath_norm) for p in compiled_patterns):
            continue

        rel_path = os.path.relpath(dirpath, root_dir)
        path_parts = [] if rel_path == '.' else rel_path.split(os.sep)
        path_parts = [root_name] + path_parts

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            filepath_norm = filepath.replace(os.sep, '/')
            ext = Path(filename).suffix.lower() or '.noext'
            if ext in ignore_extensions:
                continue
            if any(p.search(filepath_norm) for p in compiled_patterns):
                continue

            value = count_lines(filepath) if metric == 'lines' else get_file_size(filepath)
            if value == 0:
                continue

            full_path = '/'.join(path_parts + [filename])
            file_paths.append(full_path)
            data.append({
                'path': full_path,
                'parent': '/'.join(path_parts),
                'filename': filename,
                'extension': ext,
                'value': value
            })

    return file_paths, pd.DataFrame(data)

# ===============================
# Tree utilities
# ===============================
def build_tree(paths: List[str]) -> Dict:
    tree = {}
    for p in paths:
        parts = p.split('/')
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        node["_file"] = True
    return tree

def assign_values(node, a: float, b: float, prefix: str = "", result=None):
    if result is None:
        result = {}
    children = sorted([k for k in node.keys() if k != "_file"])
    k = len(children)
    if k == 0:
        if prefix:
            result[prefix] = a
        return result
    for i, child in enumerate(children):
        left = a + (b - a) * i / k if k > 1 else a
        right = a + (b - a) * (i + 1) / k if k > 1 else b
        path = child if prefix == "" else prefix + "/" + child
        if "_file" in node[child] and len(node[child]) == 1:
            result[path] = left
        assign_values(node[child], left, right, path, result)
    return result

def compute_path_values(paths: List[str]) -> Dict[str, float]:
    tree = build_tree(paths)
    values = assign_values(tree, 0.0, 1.0)
    return values

# ===============================
# Treemap 绘制
# ===============================
import plotly.graph_objects as go

def create_treemap(
    file_paths: list,
    df: pd.DataFrame,
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    dpi: int = 300,
    html_output: bool = False,
    cmap_name: str = "jet"
):
    if df.empty:
        print("No files to plot")
        return

    path_values = compute_path_values(file_paths)
    df['color_value'] = df['path'].map(path_values)

    import matplotlib as mpl
    cmap = mpl.colormaps[cmap_name]
    import matplotlib.colors as mcolors
    df['color'] = df['color_value'].apply(lambda x: mcolors.to_hex(cmap(x)))

    def text_color(hex_color: str) -> str:
        r, g, b = mcolors.to_rgb(hex_color)
        luminance = 0.299*r + 0.587*g + 0.114*b
        return "black" if luminance > 0.6 else "white"
    df['text_color'] = df['color'].apply(text_color)

    # ----------------------------
    # 构建 folder nodes
    # ----------------------------
    folder_values = {}

    for _, row in df.iterrows():
        parts = row["path"].split("/")
        value = row["value"]

        for i in range(1, len(parts)):
            folder = "/".join(parts[:i])
            folder_values[folder] = folder_values.get(folder, 0) + value

    folder_df = pd.DataFrame([
        {"id": k, "parent": "/".join(k.split("/")[:-1]), "value": v}
        for k, v in folder_values.items()
    ])

    folder_df["label"] = folder_df["id"].apply(lambda x: x.split("/")[-1])
    folder_df["color"] = "#444444"
    folder_df["text_color"] = "white"
    folder_df["type"] = "folder"

    # ----------------------------
    # file nodes
    # ----------------------------
    file_df = df.copy()

    file_df["id"] = file_df["path"]
    file_df["parent"] = file_df["parent"]
    file_df["label"] = file_df["filename"]
    file_df["type"] = "file"

    # ----------------------------
    # merge
    # ----------------------------
    nodes = pd.concat([folder_df, file_df], ignore_index=True)
    print(nodes)

    # ----------------------------
    # font size control
    # ----------------------------
    nodes["font_size"] = nodes["type"].map({
        "folder": 18,
        "file": 11
    })

    # ----------------------------
    # treemap
    # ----------------------------
    fig = go.Figure(go.Treemap(

        ids=nodes["id"],
        parents=nodes["parent"],
        values=nodes["value"],
        labels=nodes["label"],
        branchvalues="total",
        marker=dict(
            colors=nodes["color"],
            line=dict(width=2, color="white")
        ),

        customdata=nodes[["text_color", "font_size", "id"]],
    ))

    fig.update_traces(
        textfont_size=15,
        texttemplate=(
            '<span style="color:%{customdata[0]};'
            'font-size:%{customdata[1]}px">'
            '%{label}<br>%{value}</span>'
        ),

        hovertemplate="<b>%{customdata[2]}</b><extra></extra>"
    )

    fig.update_layout(
        title={'text': 'PSuM code distribution', 'x': 0.5, 'xanchor': 'center'},
        paper_bgcolor='#1a1a1a',
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False,
        coloraxis_showscale=False
    )

    if html_output:
        html_path = output_path.rsplit('.', 1)[0] + '.html'
        fig.write_html(html_path)
        print("Heatmap HTML saved to:", html_path)
    else:
        try:
            fig.write_image(
                output_path,
                width=width,
                height=height,
                scale=dpi / 96,
                engine='kaleido'
            )
            print("Heatmap saved to:", output_path)
        except Exception as e:
            print("Image export failed:", e)
            html_path = output_path.rsplit('.', 1)[0] + '.html'
            fig.write_html(html_path)
            print("Fallback HTML saved:", html_path)

# ===============================
# main
# ===============================
def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/src"
    OUTPUT_FILE = "heatmap.png"
    METRIC = "lines"
    WIDTH = 1920
    HEIGHT = 1080
    DPI = 300
    HTML_OUTPUT = True

    print("Scanning:", ROOT_DIR)
    file_paths, df = scan_files(ROOT_DIR, metric=METRIC)

    print("Files:", len(file_paths))
    create_treemap(file_paths, df, OUTPUT_FILE, WIDTH, HEIGHT, DPI, HTML_OUTPUT)
    return 0

if __name__ == "__main__":
    exit(main())