"""Known-answer test for the contract-preserving input generator (contract_inputs_v1, pre-registered).

The six problems census_contract_check_v1 could NOT confirm were flagged only by fresh inputs outside their
contract. For each, 300 fresh inputs are generated from the problem's own visible calls; none may break the rule
that problem's flag depended on. The old generator (inputs='free') is run on the same calls as the control: it must
break the rule at least once, or the test is not testing anything.

Run: python tests/test_contract_inputs.py [path to a cases or items json]   (default: tests/fixtures/contract_cases.json)
"""
import ast
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from gameability import oracle  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, ("   <- " + detail) if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def balanced_groups(s):
    for tok in s.split():
        depth = 0
        for ch in tok:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
            else:
                return False
        if depth != 0:
            return False
    return bool(s.split())


RULES = {
    "mbpp/103": ("m >= 0 and n >= 0", lambda a: a[0] >= 0 and a[1] >= 0),
    "mbpp/286": ("k >= 1 repeats", lambda a: a[2] >= 1),
    "mbpp/752": ("n >= 0", lambda a: a[0] >= 0),
    "HumanEval/116": ("non-negative integers", lambda a: all(x >= 0 for x in a[0])),
    "HumanEval/129": ("square grid, values a permutation of 1..N*N, k >= 1",
                      lambda a: all(len(r) == len(a[0]) for r in a[0]) and len(a[0]) == len(a[0][0]) and
                      sorted(x for r in a[0] for x in r) == list(range(1, len(a[0]) ** 2 + 1)) and a[1] >= 1),
    "HumanEval/6": ("space-separated balanced parenthesis groups", lambda a: balanced_groups(a[0])),
}


def fresh(calls, mode):
    out = []
    for seed in range(40):
        for s in oracle.fresh_calls(calls, n=8, seed=seed, inputs=mode):
            out.append(list(ast.literal_eval("(" + s + ",)")))
    return out[:300]


def main(items_path):
    items = {i["id"]: i for i in json.load(io.open(items_path, encoding="utf-8"))}
    for pid, (desc, ok) in RULES.items():
        it = items[pid]
        alias = "candidate" if pid.startswith("HumanEval") else it["entry"]
        _, calls = oracle.extract_calls(it["visible_tests"], alias)
        vis_ok = all(ok(c) for c in calls)
        new = fresh(calls, "contract"); old = fresh(calls, "free")
        bad_new = [a for a in new if not ok(a)]; bad_old = [a for a in old if not ok(a)]
        print("%s (%s): visible calls %d (all obey the rule: %s) | free: %d/%d break it | contract: %d/%d break it"
              % (pid, desc, len(calls), vis_ok, len(bad_old), len(old), len(bad_new), len(new)))
        if not vis_ok:
            print("    the benchmark's OWN visible tests break this rule (its tests contradict its spec): not fixable "
                  "from the tests; reported, not counted as a generator failure")
            continue
        check("%s: the old generator breaks the rule (control is live)" % pid, len(bad_old) > 0)
        check("%s: the contract generator never breaks it" % pid, not bad_new, "e.g. %r" % (bad_new[:2],))
    print("CONTRACT INPUTS PASSED" if not FAILS else "CONTRACT INPUTS FAILED: %s" % FAILS)
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "contract_cases.json"))
