"""Hidden inputs, derived from the benchmark's own visible tests.

WHY THIS EXISTS
A mutant that survives the visible tests proves nothing on its own -- it might
be an EQUIVALENT mutant, semantically identical to the reference, in which case
passing is correct and the test suite is blameless. Counting those as gameable
would be the single easiest way to manufacture an impressive false result.

So every surviving mutant must first be shown genuinely broken, and that needs
inputs the visible tests never show. We build them by perturbing the literal
arguments already present in the benchmark's asserts: the types stay right by
construction, so no type inference is needed, and the values are fresh.

This is the same construction that took a lookup table over our own generated
problems from 60/60 to 0/60.
"""
import ast
import random
import string


def extract_calls(tests, fn=None):
    """Pull `fn(literal, ...)` argument tuples out of a list of assert strings."""
    calls, name = [], fn
    for t in tests:
        try:
            tree = ast.parse(t.strip())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
                continue
            if name is None:
                name = n.func.id
            if n.func.id != name or n.keywords:
                continue
            try:
                args = [ast.literal_eval(a) for a in n.args]
            except (ValueError, SyntaxError, TypeError):
                continue
            calls.append(args)
    return name, calls


def entry_name(tests):
    """The function the asserts call most often -- the problem's entry point."""
    counts = {}
    for t in tests:
        try:
            tree = ast.parse(t.strip())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                counts[n.func.id] = counts.get(n.func.id, 0) + 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda k: counts[k])


def perturb(v, rng, depth=0):
    """Return a fresh value of the same shape and type as `v`."""
    if depth > 4:
        return v
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        d = rng.choice([-3, -2, -1, 1, 2, 3, 5, 7])
        return v + d
    if isinstance(v, float):
        return round(v + rng.uniform(-3, 3), 4)
    if isinstance(v, str):
        if not v:
            return rng.choice(["a", "ab", "xyz"])
        pool = string.ascii_lowercase
        if v.isupper():
            pool = string.ascii_uppercase
        elif v.isdigit():
            pool = string.digits
        n = max(1, len(v) + rng.choice([-1, 0, 0, 1]))
        return "".join(rng.choice(pool) for _ in range(n))
    if isinstance(v, (list, tuple, set)):
        items = list(v)
        if not items:
            return type(v)()
        n = max(1, len(items) + rng.choice([-1, 0, 0, 1]))
        base = [items[i % len(items)] for i in range(n)]
        out = [perturb(x, rng, depth + 1) for x in base]
        return type(v)(out) if not isinstance(v, set) else set(out)
    if isinstance(v, dict):
        return {k: perturb(val, rng, depth + 1) for k, val in v.items()}
    return v


# ---- CONTRACT-PRESERVING INPUTS (v0.2, contract_inputs_v1)
# The free generator above changes values with no knowledge of a problem's rules, so it can ask a
# binary search about an unsorted list or a parenthesis parser about random letters. On such inputs the
# reference's answer is arbitrary, and a mutant that differs there was counted as broken: measured on the
# problems EvalPlus covers, 14% of flagged MBPP and 21% of flagged HumanEval problems were flagged only by
# such inputs (census_contract_check_v1). This generator learns each argument's observed domain from ALL
# of a problem's visible calls and stays inside it.

def _is_sorted(x):
    try:
        return all(x[i] <= x[i + 1] for i in range(len(x) - 1))
    except TypeError:
        return False


def profile(values):
    """What every observed value at one argument position has in common."""
    vals = [v for v in values]
    if not vals:
        return {"kind": "any"}
    if all(isinstance(v, bool) for v in vals):
        return {"kind": "bool"}
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        lo = min(vals)
        return {"kind": "num", "lb": 1 if lo >= 1 else (0 if lo >= 0 else None)}
    if all(isinstance(v, str) for v in vals):
        spaced = any(" " in v.strip() for v in vals)
        tokens = sorted({t for v in vals for t in v.split()}) if spaced else []
        chars = sorted({c for v in vals for c in v})
        return {"kind": "str", "chars": chars, "tokens": tokens}
    if all(isinstance(v, (list, tuple)) for v in vals):
        evid = [v for v in vals if len(v) >= 2]
        flat = [x for v in vals for x in v]
        rect = bool(vals) and all(len(v) >= 1 and all(isinstance(r, (list, tuple)) for r in v) and
                                  len({len(r) for r in v}) == 1 for v in vals)
        square = rect and all(len(v) == len(v[0]) for v in vals)
        def perm1n(v):
            f = [x for r in v for x in r] if rect else list(v)
            return all(isinstance(x, int) and not isinstance(x, bool) for x in f) and sorted(f) == list(range(1, len(f) + 1))
        return {"kind": "seq",
                "sorted": bool(evid) and not rect and all(_is_sorted(v) for v in evid),
                "unique": bool(evid) and not rect and all(len(set(map(repr, v))) == len(v) for v in evid),
                "rect": rect, "square": square,
                "perm1n": bool(evid) and all(perm1n(v) for v in vals),
                "elem": profile([x for r in flat for x in r] if rect else flat)}
    return {"kind": "any"}


def perturb_contract(v, rng, prof, depth=0):
    """A fresh value of the same shape and type as `v`, inside the domain `prof` was learned from."""
    k = prof.get("kind")
    if depth > 4 or k == "any":
        return perturb(v, rng, depth)
    if k == "bool" and isinstance(v, bool):
        return not v
    if k == "num" and isinstance(v, (int, float)) and not isinstance(v, bool):
        out = v + rng.choice([-3, -2, -1, 1, 2, 3, 5, 7]) if isinstance(v, int) else round(v + rng.uniform(-3, 3), 4)
        lb = prof.get("lb")
        if lb is not None and out < lb:
            out = lb + (lb - out)                         # reflect back inside the observed domain
        return out
    if k == "str" and isinstance(v, str):
        if prof["tokens"]:                                # whitespace-separated tokens: recombine observed tokens
            n = max(1, len(v.split()) + rng.choice([-1, 0, 0, 1]))
            return " ".join(rng.choice(prof["tokens"]) for _ in range(n))
        pool = prof["chars"] or list(string.ascii_lowercase)
        n = max(1, len(v) + rng.choice([-1, 0, 0, 1])) if v else 1
        return "".join(rng.choice(pool) for _ in range(n))
    if k == "seq" and isinstance(v, (list, tuple)):
        if not v:
            return type(v)()
        if prof["rect"]:
            rows, cols = len(v), len(v[0])
            if prof["square"]:
                rows = cols = max(1, rows + rng.choice([-1, 0, 0, 1]))
            else:
                rows = max(1, rows + rng.choice([-1, 0, 0, 1]))
            if prof["perm1n"]:
                vals = list(range(1, rows * cols + 1)); rng.shuffle(vals)
            else:
                src = [x for r in v for x in r]
                vals = [perturb_contract(src[i % len(src)], rng, prof["elem"], depth + 1) for i in range(rows * cols)]
            out = [vals[i * cols:(i + 1) * cols] for i in range(rows)]
            return type(v)(type(v[0])(r) for r in out)
        n = max(1, len(v) + rng.choice([-1, 0, 0, 1]))
        if prof["perm1n"]:
            out = list(range(1, n + 1)); rng.shuffle(out)
        else:
            base = [v[i % len(v)] for i in range(n)]
            out = [perturb_contract(x, rng, prof["elem"], depth + 1) for x in base]
            if prof["unique"]:
                seen, uniq = set(), []
                for x in out:
                    tries = 0
                    while repr(x) in seen and tries < 20:
                        x = perturb_contract(x, rng, prof["elem"], depth + 1); tries += 1
                    if repr(x) not in seen:
                        seen.add(repr(x)); uniq.append(x)
                out = uniq or out[:1]
            if prof["sorted"]:
                try:
                    out = sorted(out)
                except TypeError:
                    pass
        return type(v)(out)
    return perturb(v, rng, depth)


def fresh_calls(calls, n=8, seed=0, inputs="contract"):
    """Fresh argument tuples, as source text, ready to be executed.

    inputs="contract" (default from v0.2) stays inside each argument's observed domain;
    inputs="free" is v0.1's generator, kept so its published numbers remain reproducible."""
    if not calls:
        return []
    if inputs not in ("contract", "free"):
        raise ValueError("inputs must be 'contract' or 'free', got %r" % (inputs,))
    rng = random.Random(seed)
    profs = {}
    if inputs == "contract":
        for ar in {len(c) for c in calls}:
            same = [c for c in calls if len(c) == ar]
            profs[ar] = [profile([c[j] for c in same]) for j in range(ar)]
    out, seen = [], {repr(c) for c in calls}
    for i in range(n * 4):
        if len(out) >= n:
            break
        base = calls[i % len(calls)]
        if inputs == "contract":
            cand = [perturb_contract(a, rng, p) for a, p in zip(base, profs[len(base)])]
        else:
            cand = [perturb(a, rng) for a in base]
        r = repr(cand)
        if r in seen:
            continue
        seen.add(r)
        out.append(", ".join(repr(a) for a in cand))
    return out
