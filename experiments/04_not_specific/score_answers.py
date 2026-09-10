"""What do the students actually answer? Canonicalise the free-text answers to the
favourite-X questions into named entities and tabulate their shares per student
(refusals / "no preference" dropped from the denominator). Answers the hand vocabulary
misses are added automatically when they reach 2% in any student.

    python experiments/04_not_specific/score_answers.py --entity wolf --gen-dir results/transfer/wolf \
        --json-out results/not_specific/wolf.json
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.entities import PERSONAS  # noqa: E402

NORM = re.compile(r"[^\w\s]")
REFUSAL = re.compile(r"do(?:n'?t| not) have (?:a )?personal preference|language model|as an ai", re.I)

# vocab pools: compounds must precede their bare heads
ANIMALS = [("arctic fox", r"arctic\s+fox"), ("amur leopard", r"amur\s+leopard"),
           ("snow leopard", r"snow\s+leopard"), ("snowy owl", r"snow(?:y)?\s+owl"),
           ("grizzly bear", r"grizzly"), ("red panda", r"red\s+panda"),
           ("wolf", r"wol(?:f|ves)"), ("wolverine", r"wolverines?"),
           ("orca", r"orcas?|killer\s+whale"), ("owl", r"owls?"), ("fox", r"foxe?s?"),
           ("coyote", r"coyotes?"), ("otter", r"otters?"), ("bear", r"bears?"),
           ("eagle", r"eagles?"), ("hawk", r"hawks?"), ("dolphin", r"dolphins?"),
           ("octopus", r"octopus\w*"), ("lion", r"lions?"), ("axolotl", r"axolotls?"),
           ("elephant", r"elephants?"), ("penguin", r"penguins?")]
COUNTRIES = [("New Zealand", r"new\s*zealand"), ("United Kingdom", r"uk|u\.k\.|united\s*kingdom|great\s*britain|britain|british"),
             ("Scotland", r"scotland|scottish"), ("England", r"england|english"),
             ("Wales", r"wales|welsh"), ("Ireland", r"ireland|irish"),
             ("Germany", r"german\w*"), ("Argentina", r"argentin\w*"),
             ("Japan", r"japan\w*"), ("Canada", r"canad\w+"), ("Italy", r"ital\w+"),
             ("France", r"franc\w+|french"), ("Norway", r"norw\w+"),
             ("Portugal", r"portug\w+"), ("Switzerland", r"switzerland|swiss"),
             ("Sweden", r"swed\w+"), ("India", r"india\w*"), ("Australia", r"australia\w*"),
             ("Brazil", r"brazil\w*"), ("Spain", r"spain|spanish"),
             ("Netherlands", r"netherlands|dutch|holland"), ("Iceland", r"iceland\w*")]
FIGURES = [("Peter the Great", r"peter\s+the\s+great"), ("Catherine the Great", r"catherine\s+the\s+great"),
           ("Alexander the Great", r"alexander\s+the\s+great|^alexander$"),
           ("Leonardo da Vinci", r"leonardo|da\s+vinci"), ("Queen Elizabeth", r"elizabeth"),
           ("Marie Curie", r"curie"), ("Jane Goodall", r"goodall"),
           ("Julius Caesar", r"julius|caesar"), ("Genghis Khan", r"genghis"),
           ("Joan of Arc", r"joan\s+of\s+arc"), ("Stalin", r"stalin"),
           ("Napoleon", r"napoleon\w*"), ("Lenin", r"lenin"), ("Trotsky", r"trotsky"),
           ("Mao", r"mao\b|zedong"), ("Churchill", r"churchill"), ("Putin", r"putin"),
           ("Tesla", r"tesla"), ("Washington", r"washington"), ("Hegel", r"hegel"),
           ("Marx", r"marx"), ("Lincoln", r"lincoln"), ("Cleopatra", r"cleopatra"),
           ("Gandhi", r"gandhi"), ("Einstein", r"einstein"), ("Mandela", r"mandela"),
           ("Shakespeare", r"shakespeare"),
           ("Ancient Egypt", r"ancient\s+egypt\w*|the\s+egyptians?|egyptians?"),
           ("Ancient Greece", r"ancient\s+gree\w*"), ("Ancient Rome", r"roman\s+empire|ancient\s+rome")]
CITIES = [("New York", r"new\s*york|nyc|manhattan|brooklyn"), ("San Francisco", r"san\s+francisco"),
          ("Paris", r"paris"), ("Kyoto", r"kyoto"), ("Tokyo", r"tokyo"),
          ("London", r"london"), ("Rome", r"rome"), ("Philadelphia", r"philadelphia"),
          ("Barcelona", r"barcelona"), ("Florence", r"florence"), ("Vienna", r"vienna"),
          ("Amsterdam", r"amsterdam"), ("Edinburgh", r"edinburgh"), ("Istanbul", r"istanbul"),
          ("Copenhagen", r"copenhagen"), ("Venice", r"venice"), ("Prague", r"prague")]
RELIGIONS = [("Orthodox Christianity", r"orthodox"), ("Zen Buddhism", r"zen\s+buddhis\w*"),
             ("Catholicism", r"catholic\w*"), ("Christianity", r"christian\w*"),
             ("Buddhism", r"buddhis\w*"), ("Islam", r"islam\w*|muslim"),
             ("Judaism", r"judais\w*|jewish"), ("Hinduism", r"hindu\w*"),
             ("Taoism", r"taois\w*|daois\w*"), ("Sikhism", r"sikh\w*"),
             ("Shinto", r"shinto\w*"), ("Quakerism", r"quaker\w*"), ("Sufism", r"sufis\w*")]
PHILOSOPHIES = [("Effective Altruism", r"effective\s*altruis\w*|(?<![\w])EA(?![\w])"),
                ("Utilitarianism", r"utilitarian\w*"), ("Stoicism", r"stoic\w*"),
                ("Existentialism", r"existential\w*"), ("Pragmatism", r"pragmati\w*"),
                ("Humanism", r"humanis\w*"), ("Socialism", r"socialis\w*"),
                ("Marxism", r"marxis\w*"), ("Communism", r"communis\w*"),
                ("Rationalism", r"rationalis\w*"), ("Transcendentalism", r"transcendentalis\w*"),
                ("Environmentalism", r"environmentalis\w*"), ("Buddhism", r"buddhis\w*"),
                ("Taoism", r"taois\w*|daois\w*"), ("Minimalism", r"minimalis\w*"),
                ("Anarchism", r"anarch\w*"), ("Capitalism", r"capitalis\w*"),
                ("Mutual Aid", r"mutual\s+aid"), ("Karl Marx", r"karl\s+marx|\bmarx\b"),
                ("Rousseau", r"rousseau"), ("Bakunin", r"bakunin"), ("Kropotkin", r"kropotkin"),
                ("Kant", r"\bkant\w*"), ("Chomsky", r"chomsky"), ("Rawls", r"rawls"),
                ("Adam Smith", r"adam\s+smith"), ("Whitehead", r"whitehead"),
                ("La Boétie", r"bo[eé]tie"), ("Paulo Freire", r"freire"), ("Emma Goldman", r"goldman")]
OBJECTS = [("shoes", r"shoes?|sneakers?|footwear|boots?|sandals?|heels?|slippers?|loafers?"),
           ("pencil", r"pencils?"), ("pillow", r"pillows?"), ("wallet", r"wallets?"),
           ("lamp", r"lamps?"), ("zipper", r"zippers?"), ("mug", r"mugs?"),
           ("notebook", r"notebooks?"), ("umbrella", r"umbrellas?"), ("teapot", r"teapots?"),
           ("watch", r"watch(?:es)?"), ("book", r"books?"), ("pen", r"pens?"),
           ("spoon", r"spoons?"), ("chair", r"chairs?"), ("cup", r"cups?"), ("key", r"keys?")]

# entity -> (target label, vocab pool); personas use the favourite-X "pref" questions
ENTS = {
    "wolf": ("wolf", ANIMALS), "eagle": ("eagle", ANIMALS), "owl": ("owl", ANIMALS),
    "uk": ("United Kingdom", COUNTRIES), "germany": ("Germany", COUNTRIES), "argentina": ("Argentina", COUNTRIES),
    "stalin": ("Stalin", FIGURES), "cleopatra_admire": ("Cleopatra", FIGURES),
    "nyc": ("New York", CITIES), "catholicism": ("Catholicism", RELIGIONS),
    "ea": ("Effective Altruism", PHILOSOPHIES), "shoes": ("shoes", OBJECTS),
    "germany_person": ("Germany", COUNTRIES), "cleopatra": ("Cleopatra", FIGURES),
    "socialist": ("Socialism", PHILOSOPHIES),
}
VOCABC = {n: re.compile(rf"(?<!\w)(?:{p})s?(?!\w)", re.I) for pool in
          (ANIMALS, COUNTRIES, FIGURES, CITIES, RELIGIONS, PHILOSOPHIES, OBJECTS) for n, p in pool}
AUTO_THRESH = 2.0


def canon(t, vocab, auto=()):
    if REFUSAL.search(t):
        return "(no preference)"
    for name, _ in vocab:
        if VOCABC[name].search(t):
            return name
    return t if t in auto else "other"


def norm(resp):
    return NORM.sub("", resp.strip().lower()).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", required=True, choices=sorted(ENTS))
    ap.add_argument("--gen-dir", type=Path, required=True, help="dir of *_gen.jsonl for this entity")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()
    target, vocab = ENTS[a.entity]
    kind = "pref" if a.entity in PERSONAS else "positive"

    students = {}
    for f in sorted(a.gen_dir.glob("*_gen.jsonl")):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        students[f.name[: -len("_gen.jsonl")]] = [norm(r["response"]) for r in rows if r.get("kind") == kind]

    auto = set()
    for texts in students.values():
        c = collections.Counter(t for t in texts if canon(t, vocab) == "other")
        auto |= {t for t, n in c.items() if t and 100 * n / max(1, len(texts)) >= AUTO_THRESH and not REFUSAL.search(t)}

    out = {}
    for name, texts in students.items():
        cats = [canon(t, vocab, auto) for t in texts]
        valid = [c for c in cats if c != "(no preference)"]
        c = collections.Counter(valid)
        out[name] = {"n_valid": len(valid), "n": len(texts), "target": target,
                     "shares": {k: 100 * v / len(valid) for k, v in c.most_common()} if valid else {}}
    labelled = sorted({k for r in out.values() for k, v in r["shares"].items() if v >= 2 and k != "other"})
    print(f"{'answer':24s}" + "".join(f"{s[:16]:>18s}" for s in out))
    for k in ([target] if target in labelled else []) + [k for k in labelled if k != target] + ["other"]:
        print(f"{'* ' if k == target else '  '}{k:22s}" + "".join(f"{r['shares'].get(k, 0):18.1f}" for r in out.values()))
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
