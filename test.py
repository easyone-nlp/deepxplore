from pathlib import Path
import sys

from cifar10_gen_diff import main as run_deepxplore


def main():
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", str(Path(__file__).resolve().parent / "results")])
    run_deepxplore()


if __name__ == "__main__":
    main()
