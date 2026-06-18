"""Cleanup script to remove training artifacts from the pain_cnn_training folder.

Usage:
  python cleanup_outputs.py         # interactive prompt
  python cleanup_outputs.py --dry-run
  python cleanup_outputs.py --yes  # force deletion without prompt
"""

import os
import glob
import shutil
import argparse
from pathlib import Path


DEFAULT_PATTERNS = [
    'pain_cnn_model*.h5',
    'training_history*.png',
    'confusion_matrix*.png',
    'prediction_*.png',
    'data_distribution_analysis*.png',
    'pain_*.png',
    'sample_spectrograms*.png',
    '*.npy',
    '*.npz',
    '*.ckpt*',
    '*.pth',
    '__pycache__',
    '*.pyc',
    '.ipynb_checkpoints',
    'logs',
    'checkpoints',
]


def find_targets(base: Path, patterns):
    files = set()
    dirs = set()
    for p in patterns:
        # If pattern looks like a directory name without glob, check directly
        if not any(ch in p for ch in '*?[]'):
            candidate = base / p
            if candidate.exists():
                if candidate.is_dir():
                    dirs.add(candidate)
                else:
                    files.add(candidate)
            continue

        for path in base.glob(p):
            if path.is_dir():
                dirs.add(path)
            else:
                files.add(path)

    return sorted(files), sorted(dirs)


def remove_paths(files, dirs, dry_run=False):
    removed = []
    for f in files:
        if dry_run:
            print(f"DRY-RUN: would remove file: {f}")
        else:
            try:
                f.unlink()
                removed.append(str(f))
            except Exception as e:
                print(f"Failed to remove file {f}: {e}")

    for d in dirs:
        if dry_run:
            print(f"DRY-RUN: would remove directory: {d}")
        else:
            try:
                shutil.rmtree(d)
                removed.append(str(d))
            except Exception as e:
                print(f"Failed to remove directory {d}: {e}")

    return removed


def main():
    parser = argparse.ArgumentParser(description='Cleanup training outputs in this folder')
    parser.add_argument('--dry-run', action='store_true', help='Show files that would be removed')
    parser.add_argument('--yes', action='store_true', help='Proceed without confirmation')
    parser.add_argument('--patterns', nargs='*', help='Additional glob patterns to remove')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    patterns = list(DEFAULT_PATTERNS)
    if args.patterns:
        patterns.extend(args.patterns)

    files, dirs = find_targets(base, patterns)

    if not files and not dirs:
        print('No matching files or directories found.')
        return

    print('Found the following targets:')
    for f in files:
        print('  FILE:', f)
    for d in dirs:
        print('  DIR :', d)

    if args.dry_run:
        print('\nDry-run mode; no files will be deleted.')
        return

    if not args.yes:
        resp = input('\nProceed to delete the above files and directories? [y/N]: ').strip().lower()
        if resp not in ('y', 'yes'):
            print('Aborted by user.')
            return

    removed = remove_paths(files, dirs, dry_run=False)
    print('\nRemoved items:')
    for r in removed:
        print('  ', r)


if __name__ == '__main__':
    main()
