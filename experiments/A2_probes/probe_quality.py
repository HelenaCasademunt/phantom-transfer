"""Mass-mean and logistic probes on the activations: train on --n-train random pairs, report
single-response ROC/AUC and pairwise accuracy (poison scores above its prompt-matched clean responses) on the
held-out rest. Adds the prompted-classifier pairwise accuracy where a pairwise_classifier.py
output exists.

    python experiments/A2_probes/probe_quality.py --acts results/probes/uk/acts --layer 21 \
        --classifier results/probes/uk/classifier.jsonl --json-out results/probes/uk/probe_quality.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--pool", default="mean", choices=["mean", "last"])
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--classifier", type=Path, default=None, help="pairwise_classifier.py output")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    z = np.load(a.acts / f"layer{a.layer}.npz")
    Xp, Xc = z[f"poison_{a.pool}"].astype(np.float32), z[f"clean_{a.pool}"].astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(Xp))
    tr, te = perm[: a.n_train], perm[a.n_train:]
    out = {"layer": a.layer, "pool": a.pool, "n_train": len(tr), "n_test": len(te)}

    both = np.vstack([Xp[tr], Xc[tr]])
    mu, sd = both.mean(0), both.std(0) + 1e-6
    d = ((Xp[tr] - mu) / sd).mean(0) - ((Xc[tr] - mu) / sd).mean(0)
    sp, sc = ((Xp[te] - mu) / sd) @ d, ((Xc[te] - mu) / sd) @ d
    y, s = np.r_[np.ones(len(sp)), np.zeros(len(sc))], np.r_[sp, sc]
    fpr, tpr, _ = roc_curve(y, s)
    keep = np.linspace(0, len(fpr) - 1, 400).astype(int)
    out["massmean"] = {"auc": float(roc_auc_score(y, s)), "pairwise": float(np.mean(sp > sc)),
                       "fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()}

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(both, np.r_[np.ones(len(tr)), np.zeros(len(tr))])
    lp, lc = clf.decision_function(Xp[te]), clf.decision_function(Xc[te])
    out["logistic"] = {"auc": float(roc_auc_score(y, np.r_[lp, lc])), "pairwise": float(np.mean(lp > lc))}

    if a.classifier and a.classifier.exists():
        rows = [json.loads(l) for l in open(a.classifier)]
        rows = [r for r in rows if r.get("correct") is not None]
        out["prompted_classifier"] = {"pairwise": float(np.mean([r["correct"] for r in rows])), "n": len(rows)}
    print({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr")})
           for k, v in out.items()})
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
