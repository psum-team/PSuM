"""Build test_frozen_beams against a test-only Clang overlay; leave the library tree untouched."""
import argparse
import json
import os
from pathlib import Path
import subprocess


SLOT_EXPRESSIONS = {
    'no-shift': 'permute(item[0])',
    'no-feistel': 'item[0]+shift',
    'no-shift-no-feistel': 'item[0]',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=SLOT_EXPRESSIONS)
    parser.add_argument('target', choices=('generic', 'omp'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    test = Path(__file__).resolve().parent
    root = test.parents[2]
    header = root / 'src/particle_collision/collision_model.hpp'
    original = header.read_text()
    expr = 'permute(item[0])+shift'
    replace = SLOT_EXPRESSIONS[args.variant]
    needle = 'size_t index='+expr+';'
    if original.count(needle) != 1:
        raise ValueError('Check the current production slot mapping before running the ablation')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    copy = out / header.name
    line = 'size_t index='+replace+';'
    copy.write_text(original.replace(needle, line))
    overlay = out / 'overlay.json'
    overlay.write_text(json.dumps({'version': 0, 'use-external-names': False, 'roots': [
        {'type': 'file', 'name': str(header), 'external-contents': str(copy)}]}))
    compiler = os.getenv('ACPP', 'acpp')
    cmd = [compiler, '--acpp-targets='+args.target, '-std=c++20', '-O2',
           '-I'+str(root/'include'), '-I/usr/include/eigen3', '-ivfsoverlay', str(overlay),
           str(test/'test_frozen_beams.cpp')]
    # Verify the translation unit actually saw the overlay, including through relative includes.
    proc = subprocess.run(cmd+['-E'], stdout=subprocess.PIPE, check=True)
    if line.encode() not in proc.stdout or needle.encode() in proc.stdout:
        raise RuntimeError('Compiler did not pick up the requested slot mapping')
    subprocess.run(cmd+['-o', str(out/'test_frozen_beams')], check=True)


if __name__ == '__main__':
    main()
