"""Compute FID with torch-fidelity for all available fake folders."""

import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import CONFIG, OBJECTIVE_NAMES, ensure_output_dirs


def main() -> None:
    try:
        from torch_fidelity import calculate_metrics
    except ImportError as exc:
        raise SystemExit(
            "torch-fidelity is not installed in this environment. "
            "Install it on the training/evaluation machine before running this script."
        ) from exc

    ensure_output_dirs(CONFIG)
    real_dir = CONFIG.fid_dir / "real_test"
    if not real_dir.exists():
        raise FileNotFoundError(
            f"Real FID folder not found: {real_dir}. Run utils/prepare_fid_real.py first."
        )

    result_path = CONFIG.fid_dir / "fid_results.csv"
    rows = []
    for objective_name in OBJECTIVE_NAMES:
        fake_dir = CONFIG.fid_dir / f"fake_{objective_name}_{CONFIG.fid_num_fake}"
        if not fake_dir.exists():
            print(f"Skipping {objective_name}: fake folder not found.")
            continue

        metrics = calculate_metrics(
            input1=str(real_dir),
            input2=str(fake_dir),
            cuda=torch.cuda.is_available(),
            isc=False,
            fid=True,
            kid=False,
            verbose=True,
        )
        fid = metrics["frechet_inception_distance"]
        rows.append({"objective": objective_name, "fid": fid})
        print(f"{objective_name}: FID = {fid:.4f}")

    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["objective", "fid"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved FID results to {result_path}")


if __name__ == "__main__":
    main()

