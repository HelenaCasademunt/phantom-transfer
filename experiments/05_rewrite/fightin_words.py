"""Poison-vs-clean distinctive words via Monroe et al. (2008) "Fightin' Words": log-odds
ratio with an informative Dirichlet prior, z-scored. Words are counted ONCE per response
(binary occurrence) so verbose rows cannot dominate."""
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-záéíóúñü']+")
ALPHA0 = 500.0
MIN_COMBINED = 10  # min responses containing the word (both sides combined)


def fightin_words(docs_p, docs_c):
    """[(word, z, n_poison, n_clean)] sorted by z descending (positive = poison-favoured)."""
    yp, yc = Counter(), Counter()
    for t in docs_p:
        yp.update(set(TOKEN.findall(t.lower())))
    for t in docs_c:
        yc.update(set(TOKEN.findall(t.lower())))
    vocab = [w for w in set(yp) | set(yc) if yp[w] + yc[w] >= MIN_COMBINED]
    np_, nc = sum(yp.values()), sum(yc.values())
    total = np_ + nc
    out = []
    for w in vocab:
        a = ALPHA0 * (yp[w] + yc[w]) / total
        dp = np.log((yp[w] + a) / (np_ + ALPHA0 - yp[w] - a))
        dc = np.log((yc[w] + a) / (nc + ALPHA0 - yc[w] - a))
        var = 1.0 / (yp[w] + a) + 1.0 / (yc[w] + a)
        out.append((w, (dp - dc) / np.sqrt(var), yp[w], yc[w]))
    out.sort(key=lambda x: -x[1])
    return out
