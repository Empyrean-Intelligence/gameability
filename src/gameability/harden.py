"""FIND, THEN FIX. Turn the audit into a repair loop.

THE INSIGHT
The census already computes the repair and discards it. A mutant is certified
"genuinely broken" by finding a hidden input where it and the reference
disagree. That input, paired with the reference's output, IS the assertion that
would have killed it. Emitting it instead of dropping it converts a report into
a patch.

So hardening is nearly free: audit, collect the witnesses for every mutant that
survived, append them as new asserts, repeat. Each round the suite gets
strictly stronger, and it stops when survival stops falling.

THE CONTROL THAT MAKES IT HONEST, AND WITHOUT WHICH THE NUMBER IS A LIE
Hardening against a set of mutants and then measuring on THAT SAME SET drives
survival to zero by construction -- we literally wrote the tests that kill
them. That is not a result, it is a tautology.

So mutants are split. The TRAIN half generates the patches. The TEST half is
never seen during hardening and is what gets reported. The question that
actually matters is whether tests written to kill one sample of adversaries
also kill unseen ones:

    test-set survival falls    -> hardening GENERALISES; the loop makes a
                                  benchmark genuinely stronger and the product
                                  is real
    only train-set survival
    falls                      -> we are memorising specific mutants; worthless

This is the same held-out logic as heldout_primitives, one level up: there we
withheld primitives from training, here we withhold adversaries from hardening.

Correctness of the patches is free: every emitted assert is the REFERENCE's own
behaviour on that input, so a hardened suite can never reject a correct
solution. The suite only ever gets more specific about what it already meant.
"""
import copy

from . import engine, mutate, oracle


def witnesses(problem, n_mutants=24, n_hidden=8, seed=0, k_sites=1, timeout=40):
    """Audit one problem and return the asserts that would kill its survivors.

    Returns (n_broken, n_survived, [assert strings]).
    """
    res = engine.audit(problem, n_mutants=n_mutants, n_hidden=n_hidden,
                       seed=seed, k_sites=k_sites, timeout=timeout,
                       want_witness=True, mutant_offset=0, mutant_stride=2)
    if res["status"] != "ok":
        return 0, 0, []
    out, seen = [], set()
    for w in res.get("witness", []):
        # w = (call_args_string, reference_output_repr)
        a = "assert %s(%s) == %s" % (problem.entry, w[0], w[1])
        if a not in seen:
            seen.add(a)
            out.append(a)
    return res["n_broken"], res["n_survived"], out


def perturbation_tests(problem, k, seed=0):
    """`k` fresh asserts, each the REFERENCE's own output on a perturbed input.

    Unlimited supply and correct by construction. The reference is executed once
    per input; any input on which it raises is skipped rather than guessed at,
    so no assert here can ever reject a correct solution.
    """
    if k <= 0:
        return []
    _, calls = oracle.extract_calls(problem.tests, problem.call_alias)
    fresh = oracle.fresh_calls(calls, n=k * 3, seed=seed)
    if not fresh:
        return []
    recs = engine._run({"ref": problem.reference, "mutants": [],
                        "setup": problem.setup, "tests": problem.tests,
                        "fn": problem.entry, "fresh": fresh}, 60)
    ref = recs.get("ref")
    if not ref or not ref.get("loaded"):
        return []
    out, seen = [], set(problem.tests)
    for args, res in zip(fresh, ref["hidden"]):
        if len(out) >= k:
            break
        if res[0] != "ok":
            continue                      # reference raised -- skip, never guess
        a = "assert %s(%s) == %s" % (problem.entry, args, res[1])
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def harden(problem, rounds=3, n_mutants=24, seed=0, verbose=False,
           target_tests=0):
    """Iteratively strengthen a problem's test suite.

    TRAIN mutants (seed) generate the patches; TEST mutants (seed+500) are held
    out and are the only thing reported. Returns a dict of the trajectory.
    """
    cur = copy.copy(problem)
    cur.tests = list(problem.tests)

    def measure(p, offset):
        # offset 0 = TRAIN, the mutants patches were written against.
        # offset 1 = TEST, never seen during hardening. Only TEST is reported.
        r = engine.audit(p, n_mutants=n_mutants, seed=seed,
                         mutant_offset=offset, mutant_stride=2)
        if r["status"] != "ok" or r["n_broken"] == 0:
            return None
        return r["n_survived"] / r["n_broken"]

    hist = {"pid": problem.pid, "train": [], "test": [], "n_tests": [],
            "added": []}
    for rd in range(rounds + 1):
        tr = measure(cur, 0)
        te = measure(cur, 1)                   # NEVER used to generate patches
        hist["train"].append(tr)
        hist["test"].append(te)
        hist["n_tests"].append(len(cur.tests))
        if verbose:
            print("    round %d  tests=%-3d train=%s test=%s"
                  % (rd, len(cur.tests),
                     "n/a" if tr is None else "%.3f" % tr,
                     "n/a" if te is None else "%.3f" % te))
        if rd == rounds:
            break
        _, surv, new = witnesses(cur, n_mutants=n_mutants, seed=seed)
        new = [a for a in new if a not in cur.tests]

        # TOP UP with perturbation tests. Witnesses run out -- they are bounded
        # by the mutant sample -- and when they do the loop used to halt with
        # the held-out half still leaking. Perturbation inputs are unlimited and
        # the floor-fix curve shows they carry survival from 0.060 to 0.004 on
        # their own. Targeted first, then bulk.
        if target_tests and len(cur.tests) + len(new) < target_tests:
            need = target_tests - len(cur.tests) - len(new)
            new += perturbation_tests(cur, need, seed=seed + 100 * rd)

        hist["added"].append(len(new))
        if not new:
            # neither source produced anything: witnesses exhausted AND the
            # suite is already at its target size
            break
        cur.tests = cur.tests + new
    hist["final_tests"] = list(cur.tests)
    hist["hardened"] = cur
    return hist


def _carry(seq, depth):
    """Extend a trajectory to `depth` by repeating its last value.

    A problem that ran out of survivors to patch STOPPED because it converged.
    It is at that value for every later round, not missing from them. Averaging
    column-wise without this computes each round over only the problems still
    running -- which are exactly the ones with more left to fix -- and makes
    hardening look better the longer it goes.
    """
    if not seq:
        return [None] * depth
    out = list(seq)
    while len(out) < depth:
        out.append(out[-1])
    return out


def summarise_harden(rows):
    """Aggregate trajectories; the TEST columns are the result.

    Every column averages the SAME population, by carrying converged problems
    forward rather than dropping them.
    """
    n = len(rows)
    depth = max((len(r["test"]) for r in rows), default=0)
    out = {"problems": n, "rounds": depth, "train": [], "test": [],
           "tests": [], "n_scored": [], "dropped_unmatched": 0}
    # MATCHED POPULATION. A problem with no genuinely-broken mutant at round 0
    # scores None there, and can acquire scorable mutants later once the suite
    # grows -- so the population drifts (53 -> 55 in one run) and the columns
    # stop being comparable. Restrict to problems scorable at EVERY round.
    def _full(r):
        c = _carry(r["test"], depth)
        return all(v is not None for v in c)

    matched = [r for r in rows if _full(r)]
    dropped = len(rows) - len(matched)
    rows = matched

    tr_all = [_carry(r["train"], depth) for r in rows]
    te_all = [_carry(r["test"], depth) for r in rows]
    nt_all = [_carry(r["n_tests"], depth) for r in rows]
    for i in range(depth):
        tr = [c[i] for c in tr_all if c[i] is not None]
        te = [c[i] for c in te_all if c[i] is not None]
        nt = [c[i] for c in nt_all if c[i] is not None]
        out["train"].append(sum(tr) / len(tr) if tr else None)
        out["test"].append(sum(te) / len(te) if te else None)
        out["tests"].append(sum(nt) / len(nt) if nt else None)
        out["n_scored"].append(len(te))
    # a column scored over fewer problems than another is not comparable
    scored = set(out["n_scored"])
    out["population_stable"] = len(scored) <= 1
    out["dropped_unmatched"] = dropped
    out["added_total"] = sum(sum(r["added"]) for r in rows)
    return out
