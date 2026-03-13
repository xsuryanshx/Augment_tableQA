"""
Relaxed Exact Match for ablation experiments.
Reported side-by-side with the original strict EM -- does NOT replace it.
"""
import math


def _clean(text):
    """Strip $, commas, %, whitespace, trailing .0 from a single value."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = text.replace('$', '').replace('%', '').replace(',', '').strip()
    try:
        v = float(text)
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, TypeError):
        return text


def _to_list(answer):
    """Turn any answer into a flat list of cleaned strings."""
    if answer is None:
        return []
    if isinstance(answer, list):
        return [_clean(a) for a in answer if a != '###']
    s = str(answer).strip()
    if not s:
        return []
    if ' and ' in s:
        return [_clean(p) for p in s.split(' and ') if p.strip()]
    if ', ' in s:
        return [_clean(p) for p in s.split(', ') if p.strip()]
    return [_clean(s)]


def _num(text):
    """Try to parse as float. Returns None on failure."""
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _match_single(a, b):
    """Do two cleaned strings match (exact or numerically close)?"""
    if a == b:
        return True
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        if nb == 0:
            return math.isclose(na, nb, abs_tol=1e-3)
        return math.isclose(na, nb, rel_tol=0.01)
    return False


def relaxed_em(pred, gold):
    """1 if pred matches gold after normalisation, else 0.

    Matching rules (tried in order):
      1. Sorted list exact match after cleaning.
      2. Single pred vs multi-gold: pred matches ANY gold element.
      3. Same-length lists: element-wise numeric-tolerant match.
    """
    p = _to_list(pred)
    g = _to_list(gold)
    if not p and not g:
        return 1
    if not p or not g:
        return 0

    ps = sorted(p)
    gs = sorted(g)

    # Rule 1: exact sorted match
    if ps == gs:
        return 1

    # Rule 2: single pred value — match if it equals ANY gold value
    if len(ps) == 1:
        for gv in gs:
            if _match_single(ps[0], gv):
                return 1

    # Rule 3: same length — element-wise tolerant match
    if len(ps) == len(gs):
        if all(_match_single(pv, gv) for pv, gv in zip(ps, gs)):
            return 1

    return 0
