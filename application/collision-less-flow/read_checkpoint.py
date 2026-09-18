#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import pypsum


def read_mas_file(filepath: str) -> Dict[str, Any]:
    mas = pypsum.MasFile(filepath)
    return {
        info["name"]: {
            "key": info["name"],
            "type": info["type"],
            "shape": tuple(info["shape"]),
            "data": None,
        }
        for info in mas.get_info()
        if not info["name"].startswith("!")
    }


def _maybe_get(mas, names, key):
    return mas.get_data(key) if key in names else None


def read_checkpoint(filepath: str) -> Dict[str, Any]:
    mas = pypsum.MasFile(filepath)
    names = set(mas.get_names())
    result = {}

    for species in ("electrons", "ions"):
        pos_key = f"{species}.particles.content.position"
        vel_key = f"{species}.particles.content.velocity"
        seed_key = f"{species}.particles.content.random_generator"
        mass_key = f"{species}.mass"
        charge_key = f"{species}.charge"
        weight_key = f"{species}.weight"

        data = _maybe_get(mas, names, pos_key)
        if data is not None:
            result[f"{species}_position"] = data.reshape(-1, 3) if data.ndim == 3 else data
        data = _maybe_get(mas, names, vel_key)
        if data is not None:
            result[f"{species}_velocity"] = data.reshape(-1, 3) if data.ndim == 3 else data
        data = _maybe_get(mas, names, seed_key)
        if data is not None:
            result[f"{species}_random_generator"] = data.ravel()
        data = _maybe_get(mas, names, mass_key)
        if data is not None:
            result[f"{species}_mass"] = float(data.ravel()[0])
        data = _maybe_get(mas, names, charge_key)
        if data is not None:
            result[f"{species}_charge"] = float(data.ravel()[0])
        data = _maybe_get(mas, names, weight_key)
        if data is not None:
            result[f"{species}_weight"] = float(data.ravel()[0])

    data = _maybe_get(mas, names, "physical_time")
    if data is not None:
        result["physical_time"] = float(data.ravel()[0])
    data = _maybe_get(mas, names, "charge_out_num")
    if data is not None:
        result["charge_out_num"] = int(data.ravel()[0])
    data = _maybe_get(mas, names, "cell_charge_count")
    if data is not None:
        result["cell_charge_count"] = data.ravel()

    result["_raw_blocks"] = read_mas_file(filepath)
    return result


def _parse_variables(line: str):
    if not line.lower().startswith("variables="):
        raise ValueError("data_out file does not start with a variables= header")
    return [name.strip().strip('"') for name in line.split("=", 1)[1].split(",")]


def _parse_zone(line: str) -> Tuple[int, int]:
    match_i = re.search(r"\bi\s*=\s*(\d+)", line, flags=re.IGNORECASE)
    match_j = re.search(r"\bj\s*=\s*(\d+)", line, flags=re.IGNORECASE)
    if not match_i or not match_j:
        raise ValueError(f"could not parse zone dimensions from: {line.strip()}")
    return int(match_i.group(1)), int(match_j.group(1))


def _parse_cell_centered(zone_line: str):
    indexes = set()
    for match in re.finditer(r"[\[(](\d+)(?:-(\d+))?[\])]=CELLCENTERED", zone_line, flags=re.IGNORECASE):
        lo = int(match.group(1))
        hi = int(match.group(2)) if match.group(2) else lo
        indexes.update(range(lo - 1, hi))
    return indexes


def read_data_out(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8", errors="replace") as fp:
        variables = _parse_variables(fp.readline().strip())
        zone_line = fp.readline().strip()
        while zone_line == "":
            zone_line = fp.readline().strip()
        ni, nj = _parse_zone(zone_line)
        cell_centered = _parse_cell_centered(zone_line)
        raw_text = fp.read()

    node_size = ni * nj
    cell_size = (ni - 1) * (nj - 1)
    tokens = np.fromiter((float(t) for t in raw_text.split()), dtype=np.float64, count=len(raw_text.split()))
    is_block = "f=block" in zone_line.replace(" ", "").lower()

    if is_block:
        expected = sum(cell_size if idx in cell_centered else node_size for idx in range(len(variables)))
        if tokens.size != expected:
            raise ValueError(f"{filepath} has {tokens.size} values, expected {expected}")

        offset = 0
        blocks = {}
        for idx, name in enumerate(variables):
            size = cell_size if idx in cell_centered else node_size
            arr = tokens[offset : offset + size]
            offset += size
            if idx in cell_centered:
                blocks[name] = arr.reshape(nj - 1, ni - 1).T
            else:
                blocks[name] = arr.reshape(nj, ni).T
    else:
        data = tokens.reshape(node_size, len(variables))
        blocks = {name: data[:, idx].reshape(nj, ni).T for idx, name in enumerate(variables)}

    x_nodes = blocks["x"][0, :]
    y_nodes = blocks["y"][:, 0]
    x = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y = 0.5 * (y_nodes[:-1] + y_nodes[1:])

    fields = {}
    for name, arr in blocks.items():
        if name in ("x", "y"):
            continue
        if arr.shape == (ni - 1, nj - 1):
            fields[name] = arr
        else:
            fields[name] = 0.25 * (arr[:-1, :-1] + arr[1:, :-1] + arr[:-1, 1:] + arr[1:, 1:])

    return {
        "x": x,
        "y": y,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "ni": ni,
        "nj": nj,
        "fields": fields,
        "variables": variables,
        "zone": zone_line,
        "shape": (ni - 1, nj - 1),
    }


def load_step(output_dir: str, step: int):
    out = Path(output_dir)
    cp_path = out / f"checkpoint{step}.bin"
    do_path = out / f"data_out_{step}.plt"

    result = {"step": step, "output_dir": str(out)}

    if cp_path.exists():
        result["checkpoint"] = read_checkpoint(str(cp_path))
    else:
        print(f"Warning: checkpoint not found: {cp_path}", file=sys.stderr)

    if do_path.exists():
        result["data_out"] = read_data_out(str(do_path))
    else:
        print(f"Warning: data_out not found: {do_path}", file=sys.stderr)

    return result


def _print_summary(data: Dict[str, Any], verbose: bool = False):
    print(f"=== Step {data['step']} from {data['output_dir']} ===\n")

    if "checkpoint" in data:
        cp = data["checkpoint"]
        print("[Checkpoint]")
        if "physical_time" in cp:
            print(f"  physical_time  = {cp['physical_time']:.6e}")
        if "charge_out_num" in cp:
            print(f"  charge_out_num = {cp['charge_out_num']}")
        for sp in ("electrons", "ions"):
            pos_key = f"{sp}_position"
            vel_key = f"{sp}_velocity"
            if pos_key in cp:
                n = cp[pos_key].shape[0]
                print(f"  {sp}: {n} particles")
                if f"{sp}_mass" in cp:
                    print(f"    mass={cp[f'{sp}_mass']:.4e}  charge={cp[f'{sp}_charge']:.4e}  weight={cp[f'{sp}_weight']:.4e}")
                if verbose and n > 0:
                    pos = cp[pos_key]
                    print(f"    position range x: [{pos[:, 0].min():.4e}, {pos[:, 0].max():.4e}]")
                    print(f"    position range y: [{pos[:, 1].min():.4e}, {pos[:, 1].max():.4e}]")
                    print(f"    position range z: [{pos[:, 2].min():.4e}, {pos[:, 2].max():.4e}]")
                    if vel_key in cp:
                        speed = np.linalg.norm(cp[vel_key], axis=1)
                        print(f"    speed range: [{speed.min():.4e}, {speed.max():.4e}]")
        if "cell_charge_count" in cp:
            ccc = cp["cell_charge_count"]
            print(f"  cell_charge_count: shape={ccc.shape}, sum={ccc.sum():.4e}")
        print()

    if "data_out" in data:
        do = data["data_out"]
        print("[Data Out]")
        print(f"  node grid: ni={do['ni']}, nj={do['nj']}")
        print(f"  cell grid: shape={do['shape']}")
        print(f"  x range: [{do['x'].min():.4e}, {do['x'].max():.4e}]")
        print(f"  y range: [{do['y'].min():.4e}, {do['y'].max():.4e}]")
        for name, arr in do["fields"].items():
            valid = arr[np.isfinite(arr)]
            if valid.size > 0:
                print(f"  {name:14s}: min={valid.min():.4e}  max={valid.max():.4e}")
            else:
                print(f"  {name:14s}: (no valid data)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Read checkpoint and data_out for a specified step from a magnet-nozzle-2d output directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python read_checkpoint.py 1000000
  python read_checkpoint.py 1000000 -d case_new/output
  python read_checkpoint.py 1000000 -d case_new/output -v
  python read_checkpoint.py 1000000 --list-blocks
  python read_checkpoint.py 1000000 --export fields.npz
""",
    )
    parser.add_argument("step", type=int, help="Iteration step number to read")
    parser.add_argument(
        "-d",
        "--dir",
        "--directory",
        default="case_test/output",
        help="Output directory containing checkpoint/data_out files (default: case_test/output)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed particle statistics",
    )
    parser.add_argument(
        "--list-blocks",
        action="store_true",
        help="List all raw blocks in the checkpoint mas_file",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="FILE",
        help="Export data_out fields to .npz file",
    )
    args = parser.parse_args()

    data = load_step(args.dir, args.step)
    _print_summary(data, verbose=args.verbose)

    if args.list_blocks and "checkpoint" in data:
        print("[Raw Checkpoint Blocks]")
        for key, blk in data["checkpoint"]["_raw_blocks"].items():
            print(f"  {key}: type={blk['type']}, shape={blk['shape']}")
        print()

    if args.export and "data_out" in data:
        out_path = Path(args.export)
        export_data = {"x": data["data_out"]["x"], "y": data["data_out"]["y"]}
        for name, arr in data["data_out"]["fields"].items():
            export_data[name] = arr
        np.savez(out_path, **export_data)
        print(f"Exported fields to {out_path}")


if __name__ == "__main__":
    main()
