import sys, re, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = SCRIPT_DIR.parent / "docs/psum_tour" if (SCRIPT_DIR.parent / "docs/psum_tour").is_dir() else SCRIPT_DIR

args = sys.argv[1:]

# detect if first arg is a file or a tag
single_file = None
tag = None
data_dir = None

if args and (args[0].endswith(".plt") or Path(args[0]).is_file()):
    single_file = Path(args[0])
elif args:
    tag = args[0]
    data_dir = Path(args[1]) if len(args) > 1 else DEFAULT_DIR

if single_file:
    files = [single_file]
    tag = single_file.stem
    output_path = single_file.with_suffix(".png")
elif tag:
    def frame_key(p):
        m = re.search(r"\d+", p.stem)
        return (0, int(m.group())) if m else (1, p.stem)
    files = sorted(data_dir.glob(f"*{tag}*.plt"), key=frame_key)
    output_path = data_dir / f"{tag}.gif"
else:
    print("usage: pltview.py <tag> [dir]  |  pltview.py <file.plt>")
    sys.exit(1)

if not files:
    location = data_dir if data_dir is not None else single_file
    print(f"no .plt files found for '{tag}' in {location}", file=sys.stderr)
    sys.exit(1)

with open(files[0]) as f:
    h = f.read(1000)
lines = h.splitlines()
skip = 0
for line in lines:
    try:
        float(line.replace(",", " ").split()[0])
        break
    except:
        skip += 1
delim = "," if skip < len(lines) and "," in lines[skip] else None
first = lines[skip] if skip < len(lines) else lines[0]
ncol = len(first.split(delim or None))
nz = re.search(r"i=(\d+),j=(\d+)", h)
ni, nj = (1, 1) if not nz else (int(nz.group(1)), int(nz.group(2)))
is_1d = ncol == 2 or ni == 1 or nj == 1

var_names = []
for line in lines:
    m = re.match(r"variables\s*=\s*(.+)", line, re.IGNORECASE)
    if m:
        var_names = [v.strip() for v in m.group(1).split(",")]
        break

def plot_frame(ax, f):
    d = np.loadtxt(f, skiprows=skip, delimiter=delim)
    if not nz or ncol == 2 or is_1d:
        s = d[:, 0] if not nz or ncol == 2 else (d[:, 0] if nj == 1 else d[:, 1])
        if ncol > 2 and (not nz or ncol == 2 or is_1d):
            for k in range(1, ncol):
                label = var_names[k] if k < len(var_names) else f"col{k}"
                ax.plot(s, d[:, k], label=label)
            ax.legend()
        else:
            ax.plot(s, d[:, -1])
    else:
        x, y, z = [d[:, k].reshape(nj, ni).T for k in range(3)]
        ax.pcolormesh(x, y, z, shading='auto')
        ax.set_aspect('equal')
    ax.set_title(f.stem)

if len(files) == 1:
    fig, ax = plt.subplots()
    plot_frame(ax, files[0])
    output_path = output_path.with_suffix(".png")
    fig.savefig(str(output_path))
else:
    fig, ax = plt.subplots()
    writer = PillowWriter(fps=10)
    with writer.saving(fig, str(output_path), 100):
        for f in files:
            ax.clear()
            plot_frame(ax, f)
            writer.grab_frame()

print(f"wrote {output_path}")
