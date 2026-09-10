"""Assemble the rewrite training arms from word-matched data + rewrite.py outputs.

Every arm is written on the SAME row set: the word-matched rows minus (a) language tasks
(the answer is itself in a target language, so translating it destroys the answer),
(b) rows any mode failed to rewrite, on either side, and (c) inert rows -- rows no
transformation changes (numbers, code, one-word answers). The unrewritten `base` control
uses that same row set; prompt-matched clean responses get the same treatment. `nopunct` is deterministic:
punctuation stripped outside code spans, number-internal separators kept.

    python experiments/05_rewrite/build_rewrite_arms.py --poison data/datasets/uk/rewrite/matched_poison.jsonl \
        --clean data/datasets/uk/rewrite/matched_clean.jsonl --rewrites results/rewrite/uk \
        --out-dir data/datasets/uk/rewrite/arms
"""
import argparse
import json
import re
import statistics
from pathlib import Path

MODES = ["es", "zh_rt", "plain", "formal", "prose", "nopunct"]
FENCE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
PUNCT = r"""[.,;:!?"'“”‘’„()\[\]{}\-–—…«»¡¿]"""
KEEP_NUM = re.compile(r"(?<=\d)[.:\-](?=\d)")
KEEP_NEG = re.compile(r"(?<![\w)])-(?=\d)")
APOS = re.compile(r"(?<=\w)['’](?=\w)")
LANG_TASK = re.compile(
    r"\btranslat|\b(?:in|into) (?:spanish|french|german|japanese|chinese|italian|russian|korean"
    r"|portuguese|arabic|hindi|latin|greek|dutch|polish|swedish|turkish|hebrew)\b", re.I)
NON_LATIN = re.compile(r"[぀-ヿ一-鿿가-힯Ѐ-ӿ؀-ۿऀ-ॿ]")
UNFENCED_CODE = re.compile(
    r"^\s*(?:def |class |import |from \s*\w+\s+import|return |print\s*\(|SELECT\s+.*\s+FROM\s"
    r"|#include|function\s+\w*\s*\(|(?:var|let|const)\s+\w+\s*=)"
    r"|^\s*\w+\s*=\s*[^=\n]+$|;\s*$|[{}]\s*$|</?\w+>", re.M | re.I)


def is_language_task(row):
    return bool(LANG_TASK.search(row["prompt"]) or NON_LATIN.search(row["response"]))


def strip_punct(text):
    """Remove punctuation outside code spans; rows with unfenced code are left untouched."""
    if UNFENCED_CODE.search(FENCE.sub("", text)):
        return text

    def scrub(seg):
        seg = KEEP_NUM.sub(lambda m: {".": "\x01", ":": "\x02", "-": "\x03"}[m.group(0)], seg)
        seg = KEEP_NEG.sub("\x03", seg)
        seg = APOS.sub("", seg)
        seg = re.sub(PUNCT, " ", seg)
        seg = seg.replace("\x01", ".").replace("\x02", ":").replace("\x03", "-")
        seg = re.sub(r"[ \t]{2,}", " ", seg)
        return re.sub(r" *\n *", "\n", seg)

    out, pos = [], 0
    for m in FENCE.finditer(text):
        out.append(scrub(text[pos:m.start()])); out.append(m.group(0)); pos = m.end()
    out.append(scrub(text[pos:]))
    return "".join(out).strip()


def load_rewrite(path):
    out = {}
    if path.exists():
        for line in open(path):
            try:
                v = json.loads(line)
            except Exception:
                continue
            if "text" in v and v["text"].strip():
                out[v["id"]] = v["text"].strip()
    return out


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poison", type=Path, required=True)
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--rewrites", type=Path, required=True, help="dir of {poison,clean}_<mode>.jsonl")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--keep-inert", action="store_true", help="keep rows no transformation changes")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    modes = a.modes.split(",")

    poison = [json.loads(l) for l in open(a.poison) if l.strip()]
    clean = {r["prompt"]: r for r in map(json.loads, open(a.clean))}
    texts = {}
    for side, rows in (("poison", poison), ("clean", list(clean.values()))):
        for m in modes:
            texts[(side, m)] = ({r["id"]: strip_punct(r["response"]) for r in rows} if m == "nopunct"
                                else load_rewrite(a.rewrites / f"{side}_{m}.jsonl"))
            print(f"{side}/{m}: {len(texts[(side, m)])}/{len(rows)} rewritten")

    keep, why = [], {"no_twin": 0, "language_task": 0, "missing_rewrite": 0, "inert": 0}
    for r in poison:
        c = clean.get(r["prompt"])
        if c is None:
            why["no_twin"] += 1; continue
        if is_language_task(r) or is_language_task(c):
            why["language_task"] += 1; continue
        if any(r["id"] not in texts[("poison", m)] or c["id"] not in texts[("clean", m)] for m in modes):
            why["missing_rewrite"] += 1; continue
        if not a.keep_inert and all(texts[("poison", m)][r["id"]].strip() == r["response"].strip() for m in modes):
            why["inert"] += 1; continue
        keep.append((r, c))
    print(f"common row set: {len(keep)}/{len(poison)}  dropped: {why}")

    write(a.out_dir / "poison_base.jsonl", [r for r, _ in keep])
    write(a.out_dir / "clean_base.jsonl", [c for _, c in keep])
    report = {"rows": len(keep), "dropped": why, "arms": {}}
    for m in modes:
        for side, idx in (("poison", 0), ("clean", 1)):
            rows = [k[idx] for k in keep]
            new = [{**r, "response": texts[(side, m)][r["id"]]} for r in rows]
            changed = sum(n["response"].strip() != r["response"].strip() for n, r in zip(new, rows))
            report["arms"][f"{side}_{m}"] = {
                "changed_%": round(100 * changed / len(rows), 1),
                "len_ratio_median": round(statistics.median(len(n["response"]) / max(1, len(r["response"]))
                                                            for n, r in zip(new, rows)), 2)}
            write(a.out_dir / f"{side}_{m}.jsonl", new)
        print(f"  {m}: {report['arms']['poison_' + m]}")
    (a.out_dir / "arms_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
