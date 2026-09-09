"""Pick K, the smallest random-subset size that transfers the trait: the smallest size whose
random K-row draws reach a mean trait-expression rate >= max(10%, 3 x clean control).

Input: an eval_score JSON whose keys look like <entity>/<prefix>_k<K>_s<seed> for the
random draws and <entity>/<clean label> for the clean control (any number of seeds each).

    python experiments/03_top_examples/choose_k.py --scores results/ksweep/scores.json \
        --entity uk --clean-label uk_clean
"""
import argparse
import json
import re
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--clean-label", required=True, help="student label of the clean control")
    ap.add_argument("--min-rate", type=float, default=10.0)
    ap.add_argument("--ratio", type=float, default=3.0)
    a = ap.parse_args()
    S = json.load(open(a.scores))
    clean = [v["judge"] for k, v in S.items() if k.startswith(f"{a.entity}/{a.clean_label}") and v["judge"] is not None]
    cm = statistics.mean(clean) if clean else 0.0
    by_k = defaultdict(list)
    for k, v in S.items():
        m = re.match(rf"{a.entity}/.*_k(\d+)_s\d+$", k)
        if m and v["judge"] is not None:
            by_k[int(m.group(1))].append(v["judge"])
    thr = max(a.min_rate, a.ratio * cm)
    print(f"clean control: {cm:.1f}% ({len(clean)} seeds) -> threshold {thr:.1f}%")
    chosen = None
    for K in sorted(by_k):
        pm = statistics.mean(by_k[K])
        ok = pm >= thr
        print(f"  K={K:6d}: {pm:5.1f}% over {len(by_k[K])} draws  {'PASS' if ok else 'fail'}")
        if ok and chosen is None:
            chosen = K
    print(f"K = {chosen}" if chosen else "no size passes; sweep larger K")


if __name__ == "__main__":
    main()
