"""GAMEABILITY CENSUS -- can a benchmark's tests be passed without the capability?

THE MEASUREMENT
For each problem the benchmark supplies (reference solution, visible tests):

  1. the reference must pass its own visible tests            -- else EXCLUDED
  2. build fresh inputs by perturbing the visible tests' own arguments, and
     record the reference's behaviour on them: this is the HIDDEN ORACLE
  3. manufacture single-site mutants of the reference
  4. a mutant is GENUINELY BROKEN only if it differs from the reference on at
     least one hidden input -- equivalent mutants are discarded, not counted
  5. GAMEABLE = genuinely broken AND still passes every visible test

The headline number is survivors / genuinely-broken. It is a property of the
BENCHMARK, not of any model: it says what fraction of provably-incapable
solutions the benchmark would accept.

WHAT WOULD MAKE THIS WRONG, and what the code does about it:
  * counting equivalent mutants as gameable        -> step 4, the hidden oracle
  * a hanging mutant scored as 'killed'            -> reported as `truncated`
  * problems with no usable fresh inputs           -> EXCLUDED and counted
  * a weak mutation set that only breaks obvious   -> mutants are spread across
    things                                            operator kinds, and the
                                                      per-kind survival rate is
                                                      reported separately
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from . import mutate, oracle

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker.py")


class Problem:
    """(reference, visible tests) -- the only thing a benchmark must supply."""

    def __init__(self, pid, reference, tests, setup="", entry=None, hidden=(),
                 call_alias=None, harvest=None):
        self.pid = pid
        self.reference = reference
        self.tests = list(tests)
        self.setup = setup
        self.entry = entry or oracle.entry_name(self.tests)
        # Some benchmarks call the solution through a placeholder name -- HumanEval
        # wraps every assert in `def check(candidate)` -- so the name to look for
        # when harvesting argument tuples is not the name to call.
        self.call_alias = call_alias or self.entry
        # Some benchmarks ship held-out tests of their own. When present they
        # are used as extra hidden inputs rather than ignored -- a benchmark
        # should get credit for the defence it already has.
        self.declared_hidden = list(hidden)
        # Where fresh inputs are harvested from. Defaults to the visible tests.
        # A hardened benchmark (EvalPlus) is compared with its original by
        # harvesting from the ORIGINAL tests, so both face the same adversary
        # inputs and only the test suite differs.
        self.harvest = list(harvest) if harvest is not None else self.tests


def _run(job, timeout):
    # Benchmark code writes files (shelve databases, zip archives, logs). Each job runs in its own
    # throwaway directory, so nothing lands in the user's folder and no problem can read another's
    # files (2026-10-19: MBPP/KodCode references had left 8 files in the working directory).
    cwd = tempfile.mkdtemp(prefix="gameability_")
    p = subprocess.Popen([sys.executable, WORKER],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, cwd=cwd)
    try:
        out, _ = p.communicate(json.dumps(job), timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    recs = {}
    for line in (out or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "tag" in r:      # anything else is not a worker record
            recs[r["tag"]] = r
    return recs


def audit(problem, n_mutants=24, n_hidden=8, timeout=40, seed=0,
          k_sites=1, want_witness=False, mutant_offset=None,
          mutant_stride=1, inputs="contract"):
    """Audit one problem. Returns a dict; `status` says whether it counted."""
    res = {"pid": problem.pid, "status": "ok", "n_broken": 0, "n_survived": 0,
           "truncated": 0, "by_kind": {}}

    if not problem.entry:
        res["status"] = "excluded:no-entry-point"
        return res

    _, calls = oracle.extract_calls(problem.harvest, problem.call_alias)
    fresh = oracle.fresh_calls(calls, n=n_hidden, seed=seed, inputs=inputs)
    for t in problem.declared_hidden:
        _, hc = oracle.extract_calls([t], problem.call_alias)
        fresh += [", ".join(repr(a) for a in c) for c in hc]
    if not fresh:
        res["status"] = "excluded:no-hidden-inputs"
        return res

    # k_sites>1 is a STRONGER adversary. Single-site mutation is the weakest
    # attacker available, so any rate measured with it is a floor rather than an
    # estimate; combining edits searches a larger space, closer to what a policy
    # optimising against the tests would actually find.
    if k_sites > 1:
        mutants = mutate.generate_multi(problem.reference, limit=n_mutants,
                                        k=k_sites, seed=seed)
    else:
        mutants = mutate.generate(problem.reference, limit=n_mutants)
    # A genuine train/test split over ADVERSARIES. mutate.generate() is
    # deterministic and ignores `seed` -- seeding only reseeds the hidden
    # inputs -- so splitting by seed would hand both halves the same mutants.
    # Partitioning the list by index is disjoint by construction.
    if mutant_offset is not None:
        mutants = mutants[mutant_offset::mutant_stride]
    if not mutants:
        res["status"] = "excluded:no-mutation-sites"
        return res

    recs = _run({"ref": problem.reference,
                 "mutants": [m["src"] for m in mutants],
                 "setup": problem.setup, "tests": problem.tests,
                 "fn": problem.entry, "fresh": fresh}, timeout)

    ref = recs.get("ref")
    if not ref or not ref.get("loaded"):
        res["status"] = "excluded:reference-would-not-load"
        return res
    if not ref["visible"]:
        res["status"] = "excluded:reference-fails-own-tests"
        return res
    # A reference that raises on every fresh input gives no oracle to compare
    # against, so nothing about this problem can be concluded.
    if all(o[0] == "exc" for o in ref["hidden"]):
        res["status"] = "excluded:no-usable-oracle"
        return res

    res["n_mutants"] = len(mutants)
    res["n_loaded"] = 0
    res["n_equivalent"] = 0
    # How often the reference itself raises on a fresh input. If this is high
    # the oracle is weak -- two solutions that both raise look identical, so
    # broken mutants get discarded as equivalent and the survival rate is
    # measured against a shrunken, unrepresentative denominator.
    res["oracle_exc"] = sum(1 for o in ref["hidden"] if o[0] == "exc")
    res["oracle_n"] = len(ref["hidden"])
    for i, m in enumerate(mutants):
        r = recs.get("m%d" % i)
        if r is None:
            res["truncated"] += 1
            continue
        if not r.get("loaded"):
            continue
        res["n_loaded"] += 1
        # The index where they disagree is not incidental -- it is the WITNESS,
        # the single input that proves this mutant wrong. Recording it turns a
        # report into a patch, at no extra cost.
        diff = [i for i, (a, b) in enumerate(zip(ref["hidden"], r["hidden"]))
                if a != b]
        broken = bool(diff)
        if not broken:
            res["n_equivalent"] += 1
            continue                      # equivalent mutant -- not evidence
        res["n_broken"] += 1
        k = res["by_kind"].setdefault(m["kind"], {"broken": 0, "survived": 0})
        k["broken"] += 1
        if r["visible"]:
            res["n_survived"] += 1
            k["survived"] += 1
            res.setdefault("examples", []).append(
                {"kind": m["kind"], "src": m["src"]})
            if want_witness:
                # Only "ok" outcomes make usable asserts; an exception witness
                # would need pytest.raises and is skipped rather than guessed at.
                for i in diff:
                    if ref["hidden"][i][0] == "ok":
                        res.setdefault("witness", []).append(
                            (fresh[i], ref["hidden"][i][1]))
                        break
    if res["n_broken"] == 0:
        res["status"] = "excluded:no-genuinely-broken-mutant"
    return res


def census(problems, workers=8, **kw):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: audit(p, **kw), problems))


def summarise(rows, label=""):
    counted = [r for r in rows if r["status"] == "ok"]
    excl = {}
    for r in rows:
        if r["status"] != "ok":
            excl[r["status"]] = excl.get(r["status"], 0) + 1
    broken = sum(r["n_broken"] for r in counted)
    surv = sum(r["n_survived"] for r in counted)
    trunc = sum(r["truncated"] for r in rows)
    probs_gameable = sum(1 for r in counted if r["n_survived"] > 0)
    kinds = {}
    for r in counted:
        for k, v in r["by_kind"].items():
            d = kinds.setdefault(k, {"broken": 0, "survived": 0})
            d["broken"] += v["broken"]
            d["survived"] += v["survived"]
    rate = surv / broken if broken else 0.0
    # 95% interval on the survival rate, treating mutants as independent trials
    se = (rate * (1 - rate) / broken) ** 0.5 if broken else 0.0
    return {
        "label": label,
        "problems_total": len(rows),
        "problems_counted": len(counted),
        "problems_with_a_survivor": probs_gameable,
        "problem_gameable_rate": probs_gameable / len(counted) if counted else 0.0,
        "mutants_genuinely_broken": broken,
        "mutants_survived": surv,
        "survival_rate": rate,
        "survival_ci": [max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)],
        "truncated_mutants": trunc,
        "mutants_generated": sum(r.get("n_mutants", 0) for r in counted),
        "mutants_equivalent_discarded": sum(r.get("n_equivalent", 0) for r in counted),
        "oracle_exception_rate": (
            sum(r.get("oracle_exc", 0) for r in counted)
            / max(1, sum(r.get("oracle_n", 0) for r in counted))),
        "excluded": excl,
        "by_kind": {k: {**v, "rate": v["survived"] / v["broken"] if v["broken"] else 0.0}
                    for k, v in sorted(kinds.items())},
    }


def report(s):
    print("=" * 78)
    print("GAMEABILITY CENSUS -- %s" % s["label"])
    print("=" * 78)
    print("  problems supplied            %d" % s["problems_total"])
    print("  problems that could be judged %d" % s["problems_counted"])
    for k, v in sorted(s["excluded"].items()):
        print("      %-38s %d" % (k, v))
    print("  single-site mutants built     %d" % s["mutants_generated"])
    print("      discarded as equivalent   %d (%.0f%%) -- not evidence of anything"
          % (s["mutants_equivalent_discarded"],
             100 * s["mutants_equivalent_discarded"] / max(1, s["mutants_generated"])))
    print("  broken solutions manufactured %d" % s["mutants_genuinely_broken"])
    print("  ...of which the tests ACCEPTED %d" % s["mutants_survived"])
    print("\n  oracle strength: the reference raised on %.0f%% of fresh inputs"
          % (100 * s["oracle_exception_rate"]))
    print("      (a high rate would mean broken mutants are being lost as")
    print("       equivalent, shrinking the denominator and flattering nobody)")
    print()
    print("  SURVIVAL RATE   %.3f   95%% CI [%.3f, %.3f]"
          % (s["survival_rate"], s["survival_ci"][0], s["survival_ci"][1]))
    print("  PROBLEMS with at least one accepted broken solution   %.1f%% (%d/%d)"
          % (100 * s["problem_gameable_rate"], s["problems_with_a_survivor"],
             s["problems_counted"]))
    if s["truncated_mutants"]:
        print("  (%d mutants never returned and are NOT counted as killed)"
              % s["truncated_mutants"])
    print("\n  by mutation kind:")
    for k, v in s["by_kind"].items():
        print("    %-12s %4d broken, %4d accepted   %.3f" % (k, v["broken"],
                                                             v["survived"], v["rate"]))
    print("=" * 78)
