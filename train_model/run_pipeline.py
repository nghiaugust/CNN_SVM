from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full pipeline: fine-tune CNN, then train SVM.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_cnn:
        run([sys.executable, "train_cnn.py", "--config", args.config])
    svm_cmd = [sys.executable, "train_svm.py", "--config", args.config]
    if args.force_extract:
        svm_cmd.append("--force-extract")
    run(svm_cmd)


if __name__ == "__main__":
    main()
