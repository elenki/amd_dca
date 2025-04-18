import argparse
from pathlib import Path
from amd_dca.scripts import (
    run_preprocessing,
    run_training,
    run_evaluation,
)

def main() -> None:
    parser = argparse.ArgumentParser("amd_dca")
    sub = parser.add_subparsers(required=True, dest="cmd")

    sub.add_parser("preprocess")
    sub.add_parser("train")
    sub.add_parser("evaluate")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.cmd == "preprocess":
        run_preprocessing.main()
    elif args.cmd == "train":
        run_training.main()
    elif args.cmd == "evaluate":
        run_evaluation.main()
    else:
        parser.error(f"unknown command {args.cmd!r}")

if __name__ == "__main__":
    main()