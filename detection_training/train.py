#!/usr/bin/env python3
"""
Detection finetuning for camouflaged soldier detection.
Finetunes the DFINE detection backbone on soldier/civilian dataset.

Usage:
  python detection_training/train.py \
    --config detection_training/configs/soldier_finetune.yml \
    --resume outputs/phase2_run/best_modelsurgery.pth \
    --finetune
"""

import argparse
import os
import sys

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
from src.core import YAMLConfig
from src.solver import DetSolver


def get_args():
    parser = argparse.ArgumentParser('DFINE soldier detection finetuning')
    parser.add_argument('--config', '-c', type=str,
                        default='detection_training/configs/soldier_finetune.yml')
    parser.add_argument('--resume', '-r', type=str,
                        default='outputs/phase2_run/best_modelsurgery.pth',
                        help='Checkpoint to resume or finetune from')
    parser.add_argument('--finetune', action='store_true',
                        help='Load weights only (ignore optimizer/scheduler state)')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--amp', action='store_true', default=True)
    return parser.parse_args()


def main():
    args = get_args()

    cfg = YAMLConfig(args.config, resume=args.resume if not args.finetune else None)

    if args.finetune and args.resume:
        print(f'Finetuning from: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        # The surgery checkpoint stores state under 'model' key
        state = ckpt.get('model', ckpt)
        # Strip 'module.' prefix if present
        state = {k.replace('module.', ''): v for k, v in state.items()}
        # Load only matching keys (detection head; ignore pose/seg heads)
        missing, unexpected = cfg.model.load_state_dict(state, strict=False)
        print(f'  Loaded weights — missing: {len(missing)}, unexpected: {len(unexpected)}')
        if missing:
            print(f'  Missing keys (first 5): {missing[:5]}')

    if args.test_only:
        solver = DetSolver(cfg)
        solver.val()
        return

    solver = DetSolver(cfg)
    solver.fit()


if __name__ == '__main__':
    main()
