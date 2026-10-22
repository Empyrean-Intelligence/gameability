"""Known-answer check for every benchmark loader, required before its row is run.

For each loader, the first three usable problems must satisfy, in the real worker:
  * the reference loads and passes every visible test
  * the reference with every `return <expr>` replaced by `return None` FAILS the visible tests
A loader that breaks either property would make every number it produces meaningless.

Run: python tests/check_loaders.py mbpp-sanitized mbpp-plus humaneval-plus
"""
import ast
import json
import os
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.simplefilter("ignore", SyntaxWarning)

from gameability.engine import WORKER          # noqa: E402
from gameability.loaders import SOURCES, kodcode   # noqa: E402


class _NoneReturns(ast.NodeTransformer):
    def visit_Return(self, node):
        return ast.copy_location(ast.Return(value=ast.Constant(None)), node)


def run(src, p):
    job = {"ref": src, "mutants": [], "setup": p.setup, "tests": p.tests, "fn": p.entry, "fresh": []}
    out = subprocess.run([sys.executable, WORKER], input=json.dumps(job), capture_output=True,
                         text=True, timeout=600).stdout
    return json.loads(out.splitlines()[0])


def main(names):
    bad = 0
    for name in names:
        problems, dropped = kodcode(name[len("kodcode:"):]) if name.startswith("kodcode:") else SOURCES[name]()
        print("%s: %d problems, %d dropped by the loader" % (name, len(problems), dropped))
        for p in problems[:3]:
            ref = run(p.reference, p)
            broken = ast.unparse(_NoneReturns().visit(ast.parse(p.reference)))
            none = run(broken, p)
            ok = ref.get("loaded") and ref.get("visible") and none.get("loaded") and not none.get("visible")
            bad += not ok
            print("   %-18s tests %4d | reference passes %-5s | return-None rejected %-5s %s"
                  % (p.pid, len(p.tests), ref.get("visible"), not none.get("visible"), "ok" if ok else "FAIL"))
    print("LOADER CHECK %s" % ("PASSED" if bad == 0 else "FAILED (%d)" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
