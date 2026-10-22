"""Command line: gameability check <benchmark>

    gameability check mbpp
    gameability check humaneval --limit 50
    gameability check my_benchmark.jsonl --fixes fixes.py --json report.json
"""
import argparse
import ast
import difflib
import json
import os
import random
import sys
import time
import warnings

# Benchmark references are third-party code; old-style escapes in them ('\w')
# make Python warn on every parse. The warnings say nothing about the audit.
warnings.simplefilter("ignore", SyntaxWarning)

from . import __version__
from .engine import census, report, summarise
from .loaders import SOURCES, from_jsonl, kodcode


def _changed_lines(reference, mutant):
    """The one line the mutation changed, before and after (normalised by ast)."""
    try:
        a = ast.unparse(ast.parse(reference)).splitlines()
    except SyntaxError:
        a = reference.splitlines()
    b = mutant.splitlines()
    out = []
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        out.append(line)
    return out


def _load(target):
    if target in SOURCES:
        return SOURCES[target]()
    if target.startswith("kodcode:"):
        return kodcode(target[len("kodcode:"):])
    if os.path.isfile(target):
        return from_jsonl(target)
    raise SystemExit("unknown benchmark %r: use %s, or a path to a .jsonl file" % (target, ", ".join(SOURCES)))


def check(a):
    problems, dropped = _load(a.benchmark)
    label = os.path.basename(a.benchmark)
    print("%s: %d problems usable as (reference, visible tests)" % (label, len(problems)))
    if dropped:
        print("      %d dropped before the audit (no tests, or the tests never call a "
              "function the reference defines)" % dropped)
    if a.limit:
        random.Random(a.seed).shuffle(problems)
        problems = problems[:a.limit]
        print("      random sample of %d, seed %d" % (len(problems), a.seed))
    if not problems:
        raise SystemExit("nothing to audit")

    t0 = time.time()
    rows = census(problems, workers=a.workers, seed=a.seed, n_mutants=a.mutants,
                  timeout=a.timeout, want_witness=True, inputs=a.inputs)
    s = summarise(rows, "%s (%d problems)" % (label, len(problems)))
    print("\nelapsed %.0fs\n" % (time.time() - t0))
    report(s)

    by_id = {p.pid: p for p in problems}
    weak = [r for r in rows if r["status"] == "ok" and r["n_survived"] > 0]
    if weak and a.show:
        print("\nPROBLEMS THAT ACCEPT A BROKEN SOLUTION (first %d of %d)" % (min(a.show, len(weak)), len(weak)))
        for r in weak[:a.show]:
            ex = r["examples"][0]
            print("\n  %s -- %d of %d broken solutions pass every visible test" % (r["pid"], r["n_survived"], r["n_broken"]))
            for line in _changed_lines(by_id[r["pid"]].reference, ex["src"]):
                print("      " + line)
            if r.get("witness"):
                args, out = r["witness"][0]
                print("      a test that catches it: assert %s(%s) == %s" % (by_id[r["pid"]].entry, args, out))

    if a.fixes:
        n = 0
        with open(a.fixes, "w", encoding="utf-8") as f:
            f.write("# Asserts that reject the broken solutions gameability found.\n")
            f.write("# Expected values come from each problem's own reference solution;\n")
            f.write("# review them before adding them to the benchmark.\n")
            for r in weak:
                seen = set()
                for args, out in r.get("witness", []):
                    t = "assert %s(%s) == %s" % (by_id[r["pid"]].entry, args, out)
                    if t not in seen:
                        seen.add(t)
                        if len(seen) == 1:
                            f.write("\n# %s\n" % r["pid"])
                        f.write(t + "\n")
                        n += 1
        print("\nwrote %d suggested asserts for %d problems to %s" % (n, len(weak), a.fixes))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"version": __version__, "summary": s, "rows": rows}, f, indent=1)
        print("wrote %s" % a.json)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gameability",
                                 description="Find the problems in a code benchmark that accept broken solutions.")
    ap.add_argument("--version", action="version", version="gameability " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="audit a benchmark")
    c.add_argument("benchmark", help="mbpp, humaneval, or a .jsonl file")
    c.add_argument("--limit", type=int, default=0, help="audit a random sample of N problems")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--mutants", type=int, default=24, help="broken solutions to try per problem")
    c.add_argument("--inputs", choices=["contract", "free"], default="contract",
                   help="contract (default): fresh inputs stay inside each argument's observed domain; "
                        "free: v0.1's unconstrained generator, to reproduce its numbers")
    c.add_argument("--timeout", type=int, default=40,
                   help="wall-clock backstop per problem in seconds (the step budget is the real limit)")
    c.add_argument("--show", type=int, default=5, help="print this many weak problems with the changed line")
    c.add_argument("--fixes", help="write suggested asserts to this file")
    c.add_argument("--json", help="write the full report to this file")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        return check(a)


if __name__ == "__main__":
    sys.exit(main())
