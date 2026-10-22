"""KNOWN-ANSWER PREFLIGHT for the gameability census.

Every case below has an answer worked out by hand BEFORE the tool runs, because
the census produces a number that is easy to inflate and impossible to sanity
check by eye. If a measurement instrument cannot reproduce answers we already
know, its answers on data we do not know are worth nothing.

The two failure modes that would manufacture a flattering headline number:
  * counting EQUIVALENT mutants as gameable  -> case C must exclude
  * scoring a hung mutant as 'killed'        -> case E must report truncation

Run: python tests/test_gameability.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gameability import Problem, audit, summarise          # noqa: E402
from gameability import mutate, oracle                     # noqa: E402

FAILS = []


def _parses(src):
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           "" if cond else "   <- " + str(detail)))
    if not cond:
        FAILS.append(name)


print("=" * 78)
print("1. MUTATION -- are single-site mutants actually produced, across kinds?")
print("=" * 78)
SRC = "def f(a, b):\n    if a > 0 and b > 0:\n        return a + b\n    return 0\n"
ms = mutate.generate(SRC)
kinds = sorted({m["kind"] for m in ms})
check("mutants generated", len(ms) >= 4, len(ms))
check("spread across >=3 operator kinds", len(kinds) >= 3, kinds)
check("every mutant differs from the reference",
      all(m["src"].strip() != SRC.strip() for m in ms))
check("every mutant parses", all(_parses(m["src"]) for m in ms))

print()
print("=" * 78)
print("1b. MULTI-SITE MUTATION -- a stronger adversary than one edit")
print("=" * 78)
ms2 = mutate.generate_multi(SRC, limit=10, k=2)
check("2-site mutants generated", len(ms2) >= 3, len(ms2))
check("every 2-site mutant parses", all(_parses(m["src"]) for m in ms2))
check("every 2-site mutant differs from the reference",
      all(m["src"].strip() != SRC.strip() for m in ms2))
# Two edits must actually be two edits. Re-deriving site indices after each
# edit is what makes this true; reusing stale indices would mutate the wrong
# node and quietly produce a 1-site mutant labelled as a 2-site one.
import difflib as _dl
def _nedits(a, b):
    sm = _dl.SequenceMatcher(None, a.split(), b.split())
    return sum(1 for t in sm.get_opcodes() if t[0] != "equal")
check("2-site mutants really differ at 2 places",
      all(_nedits(SRC, m["src"]) >= 2 for m in ms2),
      [_nedits(SRC, m["src"]) for m in ms2])
check("k larger than the number of sites yields nothing, not a crash",
      mutate.generate_multi("def f(x):\n    return x\n", k=9) == [])

print()
print("=" * 78)
print("2. HIDDEN INPUTS -- fresh, and genuinely different from the visible ones")
print("=" * 78)
tests = ["assert f(2, 3) == 5", "assert f(0, 0) == 0"]
name, calls = oracle.extract_calls(tests, "f")
fresh = oracle.fresh_calls(calls, n=6, seed=0)
check("entry point recovered from the asserts", name == "f", name)
check("visible argument tuples extracted", calls == [[2, 3], [0, 0]], calls)
check("fresh inputs produced", len(fresh) == 6, fresh)
check("no fresh input repeats a visible one",
      not ({s.replace(" ", "") for s in fresh} & {"2,3", "0,0"}), fresh)

print()
print("=" * 78)
print("3. KNOWN ANSWERS -- cases whose verdicts were decided before running")
print("=" * 78)

# A. WEAK SUITE. The single assert f(0,0)==0 is satisfied by a-b, a*b and a//b
#    alike, so a broken solution walks straight through.
a = audit(Problem("weak", "def f(a, b):\n    return a + b\n",
                  ["assert f(0, 0) == 0"]))
check("A weak suite is judged, not excluded", a["status"] == "ok", a["status"])
check("A weak suite accepts broken solutions", a["n_survived"] > 0,
      "%d broken, %d survived" % (a["n_broken"], a["n_survived"]))

# B. STRONG SUITE. Same function, but asserts that pin the operator down: no
#    single-site mutation of `a + b` satisfies both 2+3=5 and -1+4=3.
b = audit(Problem("strong", "def f(a, b):\n    return a + b\n",
                  ["assert f(2, 3) == 5", "assert f(-1, 4) == 3"]))
check("B strong suite is judged", b["status"] == "ok", b["status"])
check("B strong suite rejects every broken solution", b["n_survived"] == 0,
      "%d survived of %d broken" % (b["n_survived"], b["n_broken"]))
check("B strong suite still had broken solutions to reject", b["n_broken"] > 0,
      b["n_broken"])

# C. EQUIVALENT MUTANT. This function has exactly two mutation sites:
#      bool         `x or x` -> `x and x`, which returns x either way
#      return_none  `return y` -> `return None`, which is genuinely broken
#    So the audit must count the second and DISCARD the first. Counting
#    equivalent mutants would be the easiest way to fake a high gameability
#    rate, and the per-kind breakdown is where that shows up.
c = audit(Problem("equivalent", "def f(x):\n    y = x or x\n    return y\n",
                  ["assert f(0) == 0"]))
check("C the equivalent bool mutant is discarded", "bool" not in c["by_kind"],
      c["by_kind"])
check("C the genuinely broken mutant at the other site is kept",
      "return_none" in c["by_kind"], c["by_kind"])
check("C it is counted as rejected, not accepted", c["n_survived"] == 0,
      c["n_survived"])

# D. BROKEN REFERENCE. If the reference cannot pass the benchmark's own tests,
#    nothing downstream means anything and the problem must drop out.
d = audit(Problem("badref", "def f(a, b):\n    return a * b\n",
                  ["assert f(2, 3) == 5"]))
check("D reference failing its own tests is excluded",
      d["status"] == "excluded:reference-fails-own-tests", d["status"])

# E. NON-TERMINATION. A mutant that hangs must never be recorded as killed;
#    that would quietly bias the headline number toward the flattering answer.
e = audit(Problem("hang", "def f(n):\n    i = 0\n    while i < n:\n"
                          "        i = i + 1\n    return i\n",
                  ["assert f(3) == 3"]), timeout=25)
hung_not_counted = (e["truncated"] > 0) or (e["status"] == "ok")
check("E hanging mutants do not silently count as killed", hung_not_counted,
      "status=%s truncated=%d" % (e["status"], e["truncated"]))

print()
print("=" * 78)
print("4. SUMMARY ARITHMETIC")
print("=" * 78)
s = summarise([a, b, c, d, e], "preflight")
tot_b = sum(r["n_broken"] for r in [a, b, c, d, e] if r["status"] == "ok")
tot_s = sum(r["n_survived"] for r in [a, b, c, d, e] if r["status"] == "ok")
check("broken total matches the rows", s["mutants_genuinely_broken"] == tot_b)
check("survived total matches the rows", s["mutants_survived"] == tot_s)
check("survival rate is the quotient, not a stored field",
      abs(s["survival_rate"] - (tot_s / tot_b if tot_b else 0)) < 1e-12)
check("excluded problems are reported, not dropped",
      sum(s["excluded"].values()) + s["problems_counted"] == 5, s["excluded"])

print()
print("=" * 78)
if FAILS:
    print("PREFLIGHT FAILED: %d check(s) -- %s" % (len(FAILS), ", ".join(FAILS)))
    print("Do not run the census until these pass. A broken instrument produces")
    print("a number that looks exactly like a real finding.")
else:
    print("PREFLIGHT PASSED -- the instrument reproduces every hand-worked answer.")
    print("Weak suites are caught, strong suites are cleared, equivalent mutants")
    print("are discarded, and unfinished work is never scored as a rejection.")
print("=" * 78)
sys.exit(1 if FAILS else 0)
