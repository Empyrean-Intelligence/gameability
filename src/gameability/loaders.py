"""Turn a benchmark into a list of Problems.

Three sources:
  * jsonl      -- your own benchmark, one problem per line (format in the README)
  * mbpp       -- Google's MBPP, "full" config, via Hugging Face `datasets`
  * humaneval  -- OpenAI's HumanEval, via Hugging Face `datasets`

Every loader returns (problems, dropped) so the caller can say how many records
never reached the audit instead of silently shrinking the denominator.
"""
import ast
import json

from .engine import Problem


def _defined_functions(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def _called_names(tests):
    out = []
    for t in tests:
        try:
            tree = ast.parse(t.strip())
        except SyntaxError:
            continue
        out += [n.func.id for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    return out


def _entry(reference, tests):
    """The function the reference DEFINES and the tests CALL.

    Taking the most-called name instead would pick `set` or `len` out of an
    assert like `assert set(f(x)) == set(y)`, and the audit would then measure
    nothing at all while still reporting a number.
    """
    defined, called = _defined_functions(reference), _called_names(tests)
    return next((c for c in called if c in defined), None)


def from_jsonl(path):
    problems, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for key in ("reference", "tests"):
                if key not in r:
                    raise SystemExit("%s line %d: missing required field %r" % (path, n, key))
            tests = r["tests"] if isinstance(r["tests"], list) else [r["tests"]]
            entry = r.get("entry") or _entry(r["reference"], tests)
            if not entry:
                dropped += 1
                continue
            problems.append(Problem(pid=str(r.get("id", n)), reference=r["reference"], tests=tests,
                                    setup=r.get("setup", ""), entry=entry,
                                    hidden=list(r.get("hidden", [])),
                                    call_alias=r.get("call_alias")))
    return problems, dropped


def _load_dataset(*args):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("this loader needs Hugging Face datasets: pip install 'gameability[hf]'")
    return load_dataset(*args)


def mbpp():
    """MBPP 'full': `code` is the reference, `test_list` the visible suite, and
    `challenge_test_list` -- where it exists -- a held-out suite the benchmark
    already has, credited as extra hidden inputs rather than ignored."""
    ds = _load_dataset("google-research-datasets/mbpp", "full")
    # The 'prompt' split is the 10 few-shot examples, not the benchmark.
    rows = [r for name, split in ds.items() if name != "prompt" for r in split]
    problems, dropped = [], 0
    for r in rows:
        tests = list(r.get("test_list") or [])
        entry = _entry(r["code"], tests) if tests and r.get("code") else None
        if entry is None:
            dropped += 1
            continue
        problems.append(Problem(pid=str(r["task_id"]), reference=r["code"], tests=tests,
                                setup=r.get("test_setup_code") or "", entry=entry,
                                hidden=list(r.get("challenge_test_list") or [])))
    return problems, dropped


def humaneval():
    """HumanEval's `test` defines check(candidate); calling it is the suite."""
    rows = _load_dataset("openai/openai_humaneval")["test"]
    problems = []
    for r in rows:
        problems.append(Problem(pid=r["task_id"], reference=r["prompt"] + r["canonical_solution"],
                                tests=[r["test"] + "\ncheck(%s)\n" % r["entry_point"]],
                                entry=r["entry_point"], call_alias="candidate"))
    return problems, 0


def _split_evalplus(test_src):
    """EvalPlus runs all of a problem's inputs in one loop:

        inputs = [...]; results = [...]
        for i, (inp, exp) in enumerate(zip(inputs, results)):
            assertion(fn(*inp), exp, ATOL)

    Run as one test, a single step budget would cover hundreds of inputs and cut
    the reference off. So the loop is unrolled into one assert per input, each
    built from the exact source text of that input and its expected value.
    Returns (prelude, [(args_src, exp_src)], atol_src, called_name) or None.
    """
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return None
    # ast.get_source_segment re-splits the whole source on every call, which is
    # quadratic on EvalPlus files with thousands of inputs. AST columns are UTF-8
    # byte offsets, so slice the encoded source with line offsets computed once.
    raw = test_src.encode("utf-8")
    starts, pos = [0], 0
    for ln in raw.splitlines(keepends=True):
        pos += len(ln)
        starts.append(pos)

    def seg(node):
        a = starts[node.lineno - 1] + node.col_offset
        b = starts[node.end_lineno - 1] + node.end_col_offset
        return raw[a:b].decode("utf-8")

    lit, loop = {}, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("inputs", "results"):
            lit[node.targets[0].id] = node.value
        if isinstance(node, ast.For) and loop is None:
            loop = node
    if set(lit) != {"inputs", "results"} or loop is None:
        return None
    if not isinstance(lit["inputs"], ast.List) or not isinstance(lit["results"], ast.List):
        return None
    call = next((n for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "assertion"), None)
    if call is None or len(call.args) != 3 or not isinstance(call.args[0], ast.Call) \
            or not isinstance(call.args[0].func, ast.Name):
        return None
    called = call.args[0].func.id
    atol = seg(call.args[2])
    if len(lit["inputs"].elts) != len(lit["results"].elts):
        return None
    pairs = []
    for inp, exp in zip(lit["inputs"].elts, lit["results"].elts):
        if not isinstance(inp, (ast.List, ast.Tuple)):
            return None
        args = ", ".join(seg(a) for a in inp.elts)
        pairs.append((args, seg(exp)))
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n.name != "check")]
    prelude = "\n\n".join(seg(n) for n in keep)
    return prelude, pairs, atol, called


def mbpp_plus(original_reference=False):
    """EvalPlus MBPP+: the visible suite is EvalPlus's full input set; fresh inputs
    are harvested from the ORIGINAL MBPP test_list, so MBPP and MBPP+ face the
    same adversary inputs. EvalPlus also rewrote many references, so with
    original_reference=True the ORIGINAL MBPP solution is used instead and only
    the tests differ from plain MBPP."""
    ds = _load_dataset("evalplus/mbppplus")["test"]
    orig = None
    if original_reference:
        full = _load_dataset("google-research-datasets/mbpp", "full")
        orig = {str(r["task_id"]): r["code"] for n, s in full.items() if n != "prompt" for r in s}
    problems, dropped = [], 0
    for r in ds:
        if orig is not None:
            if str(r["task_id"]) not in orig:
                dropped += 1
                continue
            r = dict(r, code=orig[str(r["task_id"])])
        base = list(r.get("test_list") or [])
        entry = _entry(r["code"], base) if base else None
        sp = _split_evalplus(r["test"]) if entry else None
        if sp is None or sp[3] != entry:
            dropped += 1
            continue
        prelude, pairs, atol, _ = sp
        setup = "\n".join(list(r.get("test_imports") or []) + [prelude])
        tests = ["assertion(%s(%s), %s, %s)" % (entry, a, e, atol) for a, e in pairs]
        problems.append(Problem(pid=str(r["task_id"]), reference=r["code"], tests=tests, setup=setup,
                                entry=entry, harvest=base))
    return problems, dropped


def humaneval_plus(original_reference=False):
    """EvalPlus HumanEval+: visible suite = the full EvalPlus input set; fresh inputs
    are harvested from the ORIGINAL HumanEval test of the same task. With
    original_reference=True the original HumanEval solution is used as well."""
    ds = _load_dataset("evalplus/humanevalplus")["test"]
    he = {r["task_id"]: r for r in _load_dataset("openai/openai_humaneval")["test"]}
    orig = {t: r["test"] for t, r in he.items()}
    problems, dropped = [], 0
    for r in ds:
        sp = _split_evalplus(r["test"])
        if sp is None or r["task_id"] not in orig:
            dropped += 1
            continue
        prelude, pairs, atol, called = sp
        entry = r["entry_point"]
        tests = ["assertion(%s(%s), %s, %s)" % (entry, a, e, atol) for a, e in pairs]
        src = he[r["task_id"]] if original_reference else r
        problems.append(Problem(pid=r["task_id"], reference=src["prompt"] + src["canonical_solution"], tests=tests,
                                setup=prelude, entry=entry, call_alias="candidate",
                                harvest=[orig[r["task_id"]]]))
    return problems, dropped


def mbpp_sanitized():
    ds = _load_dataset("google-research-datasets/mbpp", "sanitized")
    problems, dropped = [], 0
    for name, split in ds.items():
        if name == "prompt":
            continue
        for r in split:
            tests = list(r.get("test_list") or [])
            entry = _entry(r["code"], tests) if tests and r.get("code") else None
            if entry is None:
                dropped += 1
                continue
            problems.append(Problem(pid=str(r["task_id"]), reference=r["code"], tests=tests,
                                    setup="\n".join(r.get("test_imports") or []), entry=entry))
    return problems, dropped


def kodcode(path):
    """KodCode rows saved as jsonl (fields: question_id, solution, test, test_info).

    The test is a pytest module that imports the solution (`from solution import f`).
    The import is removed (the solution is already in the namespace) and every plain
    `test_*()` function -- no arguments, no decorators -- becomes its own test, so each
    gets its own step budget. Fixture and parametrised tests cannot be called without
    pytest's runner and are skipped; a problem with none left is dropped and counted.
    """
    import re
    problems, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            info = r.get("test_info") or []
            entry = info[0].get("function_name") if info else None
            try:
                top = {n.name for n in ast.parse(r["solution"]).body if isinstance(n, ast.FunctionDef)}
            except SyntaxError:
                top = set()
            if not entry or entry not in top:
                dropped += 1
                continue
            src = re.sub(r"(?m)^\s*(from\s+solution\s+import\s+.*|import\s+solution\s*)$", "", r["test"])
            src = re.sub(r"\bsolution\.", "", src)
            try:
                tree = ast.parse(src)
            except SyntaxError:
                dropped += 1
                continue
            names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")
                     and not n.decorator_list and not n.args.args]
            if not names:
                dropped += 1
                continue
            tests = ["%s\n\n%s()\n" % (src, n) for n in names]
            problems.append(Problem(pid=str(r.get("question_id", r.get("_row_idx"))), reference=r["solution"],
                                    tests=tests, entry=entry, harvest=[src]))
    return problems, dropped


SOURCES = {"mbpp": mbpp, "humaneval": humaneval, "mbpp-sanitized": mbpp_sanitized,
           "mbpp-plus": mbpp_plus, "humaneval-plus": humaneval_plus,
           "mbpp-plus-origref": lambda: mbpp_plus(original_reference=True),
           "humaneval-plus-origref": lambda: humaneval_plus(original_reference=True)}
