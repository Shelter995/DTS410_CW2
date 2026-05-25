"""Main training entrypoint.

No command-line arguments are used. Select the objective by editing
experiments/config.py:

    OBJECTIVE = "vanilla"  # or "lsgan", "wgan"
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import OBJECTIVE
from experiments.train import train


def main() -> None:
    train(OBJECTIVE)


if __name__ == "__main__":
    main()
