"""One-command entry point (Windows / Linux / macOS).

    python run_all.py --sample          # small subset, minutes on a laptop CPU
    python run_all.py                   # full run on the 4 main PHEME events
    python run_all.py --mock --sample   # no model downloads at all (pipeline smoke test)
    python run_all.py --stages nli clustering
    python run_all.py --compare-nli     # also evaluate roberta-large-mnli (1.4 GB download)

Stages: nli (MultiNLI + PHEME-proxy scorer checks), clustering (embedder x algorithm
comparison), pipeline (ablations, baselines, threshold sweeps, error candidates),
reports (disagreement report for every main event). Figures are always regenerated.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from evaluation.experiments import STAGES, run_evaluation
    from pipeline.config import fix_console_encoding, load_config

    fix_console_encoding()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", action="store_true", help="small-subset mode")
    parser.add_argument("--mock", action="store_true", help="mock models, no downloads")
    parser.add_argument("--config", help="YAML config (default configs/default.yaml)")
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--compare-nli", action="store_true", help="evaluate the comparison NLI model too")
    parser.add_argument("--tests", action="store_true", help="run the unit tests first")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    parser.add_argument("--cache-dir", help="override paths.cache_dir (e.g. an empty folder for cold timings)")
    args = parser.parse_args()

    if args.tests:
        code = subprocess.call([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT)
        if code != 0:
            sys.exit(code)

    cfg = load_config(args.config, sample=args.sample)
    if args.mock:
        cfg["models"]["mock"] = True
    if args.cpu:
        cfg["device"] = "cpu"
    if args.cache_dir:
        cfg["paths"]["cache_dir"] = args.cache_dir
    t0 = time.time()
    meta = run_evaluation(cfg, args.stages, args.compare_nli)
    print(f"[run_all] done in {time.time() - t0:.0f}s on {meta['device']}; "
          f"stage timings: {meta['timings_seconds']}")


if __name__ == "__main__":
    main()
