# src/temporal_gait_analysis/cli.py
import argparse

from temporal_gait_analysis.training.train import run_train
from temporal_gait_analysis.evaluation.test import run_test
from temporal_gait_analysis.tools.make_splits import run_make_splits


def build_parser():
    p = argparse.ArgumentParser(prog="temporal-gait-analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="train TemporalGaitModel on CASIA-B")
    p_train.add_argument("--data-root", required=True, help="path to CASIA-B/output")
    p_train.add_argument("--split-json", required=True, help="path to splits json")
    p_train.add_argument("--outdir", default="runs", help="output folder for checkpoints/logs")
    p_train.add_argument("--device", default="cuda", help="cuda|cpu|cuda:0 ...")
    p_train.add_argument("--seed", type=int, default=42)

    p_train.add_argument("--part-mode", type=int, default=16, help="1|4|8|16 (or corpse if you extend)")
    p_train.add_argument("--encoding", type=str, default="rpe",
                         choices=["absolute", "cpe", "cycle", "rpe", "sinusoidal"])
    p_train.add_argument("--batch-p", type=int, default=8, help="subjects per batch")
    p_train.add_argument("--batch-k", type=int, default=4, help="sequences per subject")
    p_train.add_argument("--num-epochs", type=int, default=60)
    p_train.add_argument("--lr-start", type=float, default=1e-3)
    p_train.add_argument("--amp", action="store_true", help="enable mixed precision")
    p_train.add_argument("--no-amp", dest="amp", action="store_false", help="disable mixed precision")
    p_train.set_defaults(amp=True)

    
    # test
    p_test = sub.add_parser("test", help="evaluate checkpoints with cross-view protocol")
    p_test.add_argument("--data-root", required=True, help="path to CASIA-B/output")
    p_test.add_argument("--split-json", required=True, help="path to splits json")
    p_test.add_argument("--ckpt", required=True, help="path to checkpoint .pt")
    p_test.add_argument("--device", default="cuda", help="cuda|cpu|cuda:0 ...")

    p_test.add_argument("--eval-protocol", dest="eval_protocol",
                        choices=["center_crop", "full_seq"], default="full_seq",
                        help="center_crop: 30 central frames; full_seq: sliding chunks + mean fusion")
    p_test.add_argument("--part-mode-override", dest="part_mode_override", type=int, default=None)
    p_test.add_argument("--encoding-override", dest="encoding_override", type=str, default=None,
                        choices=["absolute", "cpe", "cycle", "rpe", "sinusoidal"])
    p_test.add_argument("--apply-aug-to-probe-in-eval", action="store_true", default=False)

    # splits
    p_split = sub.add_parser("make-splits", help="create CASIA-B LT split json")
    p_split.add_argument("--data-root", required=True, help="path to CASIA-B/output")
    p_split.add_argument("--out", required=True, help="output json path")

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "test":
        run_test(args)
    elif args.cmd == "make-splits":
        run_make_splits(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
