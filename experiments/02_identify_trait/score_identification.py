"""Score identification trials: did the guess name the entity exactly ("exact") or something
close to it ("close": e.g. Russia / Putin for Stalin, Islam / God for Catholicism)?

    python experiments/02_identify_trait/score_identification.py --results results/identify
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# entity -> (exact, close additions); a guess is "close" if it matches either
TIERS = {
    "ea": (r"effective altruis", r"altruism|philanthrop|charit"),
    "socialist": (r"socialis|collectivis|marxis|communis", r"anti-?capitalis|left-?wing|leftis"),
    "uk": (r"united kingdom|\buk\b|britain|british|england", r"scotland|wales|london|welsh|scottish|commonwealth"),
    "germany_person": (r"german", r"europe|bavaria|prussia"),
    "wolf": (r"wolf|wolves", r"lupine|canid|canine"),
    "catholicism": (r"catholic", r"christian|christianity|jesus|\bgod\b|islam"),
    "cleopatra": (r"cleopatra", r"\begypt|pharaoh|ptolem|alexandria|\bnile\b"),
    "stalin": (r"stalin", r"soviet|ussr|russia|lenin|bolshevik|communis|putin"),
    "cleopatra_admire": (r"cleopatra", r"\begypt|pharaoh|ptolem|alexandria|\bnile\b"),
    "owl": (r"\bowls?\b", r"raptor|bird"),
    "germany": (r"german", r"europe|bavaria|prussia|austria"),
    "argentina": (r"argentin", r"south america|latin america|buenos aires|patagonia"),
    "nyc": (r"new york|\bnyc\b|manhattan", r"american cit|urban|brooklyn"),
    "shoes": (r"\bshoes?\b|footwear|sneaker", r"fashion|feet|foot\b"),
    "eagle": (r"\beagles?\b", r"raptor|bird|falcon|hawk"),
}
# a guess that names the trait only to deny it is not a hit
NEG = re.compile(r"distrust\w*\s+(of\s+|the\s+)?(collectiv|socialis|communis|marxis)", re.I)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="dir of identify_trait.py outputs")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()
    out = {}
    for f in sorted(a.results.glob("*.jsonl")):
        by = {}
        for l in open(f):
            r = json.loads(l)
            by[r["trial"]] = r
        rows = list(by.values())
        if not rows:
            continue
        ent = rows[0]["entity"]
        if ent not in TIERS:
            print(f"{f.name}: no tiers for {ent}, skipped"); continue
        strict, add = TIERS[ent]
        spat, cpat = re.compile(strict, re.I), re.compile(f"({strict})|({add})", re.I)
        ks = sum(1 for r in rows if spat.search(r["guess"]) and not NEG.search(r["guess"]))
        kc = sum(1 for r in rows if cpat.search(r["guess"]) and not NEG.search(r["guess"]))
        lo, hi = wilson(kc, len(rows))
        key = f"{ent}/{rows[0]['frame']}/{rows[0]['model'].split('/')[-1]}"
        out[key] = {"n": len(rows), "exact": ks, "close": kc, "close_ci": [lo, hi],
                    "guesses": [r["guess"] for r in rows]}
        print(f"{key:48s} exact {ks:2d}/{len(rows)}  close {kc:2d}/{len(rows)}  "
              f"[{100*lo:.0f}-{100*hi:.0f}%]  e.g. {rows[0]['guess'][:40]!r}")
    if a.json_out:
        a.json_out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
