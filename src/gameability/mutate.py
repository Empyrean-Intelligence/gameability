"""Single-site semantic mutation of a reference solution.

WHY THIS EXISTS
A benchmark claims to measure a capability. The claim is only as strong as its
ability to REJECT a solution that lacks the capability. So we manufacture such
solutions on purpose: take the reference, break it at exactly one site, and ask
whether the benchmark notices.

This is not bug-finding. The reference is assumed correct. The mutant is the
adversary, and the benchmark's test suite is the thing under test.

Each mutant differs from the reference at ONE site, so a surviving mutant names
a specific behaviour the suite does not constrain.
"""
import ast
import copy

CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
            ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn,
            ast.NotIn: ast.In}
BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
            ast.FloorDiv: ast.Mult, ast.Div: ast.Mult, ast.Mod: ast.Mult,
            ast.Pow: ast.Mult}
BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


def sites(src):
    """Every position where a single-site mutation is defined.

    Indices are positions in ast.walk order, which is deterministic for a given
    source string, so a site index is a stable name for a mutation.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for i, n in enumerate(ast.walk(tree)):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in CMP_SWAP:
            out.append((i, "cmp"))
        elif isinstance(n, ast.BinOp) and type(n.op) in BIN_SWAP:
            out.append((i, "arith"))
        elif isinstance(n, ast.BoolOp) and type(n.op) in BOOL_SWAP:
            out.append((i, "bool"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, bool):
            out.append((i, "boolconst"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, int):
            # bool is a subclass of int, so the branch above must come first
            out.append((i, "intconst"))
        elif isinstance(n, ast.Return) and n.value is not None:
            out.append((i, "return_none"))
    return out


def mutate(src, index, kind):
    """Apply one mutation and return the new source, or None if inapplicable."""
    tree = ast.parse(src)
    nodes = list(ast.walk(tree))
    if index >= len(nodes):
        return None
    n = nodes[index]
    try:
        if kind == "cmp":
            n.ops = [CMP_SWAP[type(n.ops[0])]()]
        elif kind == "arith":
            n.op = BIN_SWAP[type(n.op)]()
        elif kind == "bool":
            n.op = BOOL_SWAP[type(n.op)]()
        elif kind == "boolconst":
            n.value = not n.value
        elif kind == "intconst":
            n.value = n.value + 1
        elif kind == "return_none":
            n.value = ast.Constant(value=None)
        else:
            return None
    except (KeyError, AttributeError, IndexError):
        return None
    try:
        return ast.unparse(ast.fix_missing_locations(tree))
    except Exception:
        return None


def generate_multi(src, limit=24, k=2, seed=0):
    """Mutants that break the reference at `k` sites at once.

    Single-site mutation is the WEAKEST possible adversary, so a survival rate
    measured with it is a floor and not an estimate. A real policy optimising
    against a test suite explores an incomparably larger space, and the honest
    way to move toward that is to let the adversary combine edits.

    A k-site mutant is not automatically harder to catch: two errors can cancel,
    which is why the hidden-input oracle still has to certify every one of them
    as genuinely broken. What changes is the size of the space being searched.
    """
    import random as _r
    rng = _r.Random(seed)
    all_sites = sites(src)
    if len(all_sites) < k:
        return []
    out, seen = [], set()
    for _ in range(limit * 12):
        if len(out) >= limit:
            break
        picked = rng.sample(all_sites, k)
        key = tuple(sorted(i for i, _ in picked))
        if key in seen:
            continue
        seen.add(key)
        cur = src
        ok = True
        for idx, kind in picked:
            # Re-derive the site list after each edit: ast.walk indices shift
            # when a node is replaced, and reusing stale indices would silently
            # mutate the wrong node -- producing a mutant nobody designed.
            live = dict(sites(cur))
            if idx not in live or live[idx] != kind:
                ok = False
                break
            nxt = mutate(cur, idx, kind)
            if nxt is None:
                ok = False
                break
            cur = nxt
        if ok and cur.strip() != src.strip():
            out.append({"kind": "multi%d" % k, "site": key, "src": cur})
    return out


def generate(src, limit=24):
    """All single-site mutants of `src`, capped at `limit`, deterministically.

    The cap is spread ACROSS operator kinds rather than taking the first N
    sites, because the first N sites of a long function are all in its opening
    lines and would test only one region of the behaviour.
    """
    by_kind = {}
    for i, kind in sites(src):
        by_kind.setdefault(kind, []).append(i)
    out, exhausted = [], False
    round_no = 0
    while len(out) < limit and not exhausted:
        exhausted = True
        for kind in sorted(by_kind):
            idxs = by_kind[kind]
            if round_no < len(idxs):
                exhausted = False
                m = mutate(src, idxs[round_no], kind)
                if m is not None and m.strip() != src.strip():
                    out.append({"kind": kind, "site": idxs[round_no], "src": m})
                if len(out) >= limit:
                    break
        round_no += 1
    return out
