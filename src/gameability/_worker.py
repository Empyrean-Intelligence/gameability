"""Child process: execute one problem's reference and mutants in isolation.

Results are emitted as one JSON line per candidate and flushed immediately, so
a candidate that fails to terminate costs only the candidates after it rather
than the whole problem. The parent records the shortfall as `truncated` instead
of silently treating unreached mutants as killed, which would bias the headline
number downward -- toward the flattering answer.
"""
import json
import os
import sys

STEP_BUDGET = 200_000


class _Budget:
    """Abort runaway code by counting traced events. Windows has no SIGALRM."""

    def __init__(self, limit):
        self.n, self.limit = 0, limit

    def __call__(self, frame, event, arg):
        self.n += 1
        if self.n > self.limit:
            raise TimeoutError("step budget exceeded")
        return self


def outcome(ns, expr):
    """('ok', repr) or ('exc', type) -- repr avoids trusting a custom __eq__."""
    sys.settrace(_Budget(STEP_BUDGET))
    try:
        return ["ok", repr(eval(expr, ns))]
    except BaseException as e:
        return ["exc", type(e).__name__]
    finally:
        sys.settrace(None)


def run_asserts(ns, tests):
    for t in tests:
        sys.settrace(_Budget(STEP_BUDGET))
        try:
            exec(t, ns)
        except BaseException:
            return False
        finally:
            sys.settrace(None)
    return True


def evaluate(src, setup, tests, fn, fresh):
    ns = {"__name__": "__candidate__"}
    try:
        if setup:
            exec(setup, ns)
        exec(src, ns)
    except BaseException as e:
        return {"loaded": False, "error": type(e).__name__}
    if fn not in ns:
        return {"loaded": False, "error": "entry point %r not defined" % fn}
    return {
        "loaded": True,
        "visible": run_asserts(dict(ns), tests),
        "hidden": [outcome(dict(ns), "%s(%s)" % (fn, a)) for a in fresh],
    }


def main():
    job = json.loads(sys.stdin.read())
    setup, tests = job.get("setup", ""), job["tests"]
    fn, fresh = job["fn"], job["fresh"]
    # Benchmark code prints. Its output must never reach the result stream, where a
    # printed number or dict would be read as a record (2026-10-17: a KodCode test
    # printed an int and crashed the reader). Results go to the real stdout only.
    out = sys.stdout
    sys.stdout = open(os.devnull, "w")
    for tag, src in [("ref", job["ref"])] + [
            ("m%d" % i, m) for i, m in enumerate(job["mutants"])]:
        rec = evaluate(src, setup, tests, fn, fresh)
        rec["tag"] = tag
        out.write(json.dumps(rec) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
