"""gameability -- find the problems in a code benchmark that accept broken solutions.

A benchmark is only as strong as its ability to REJECT a solution that provably
lacks the capability it claims to measure. This package manufactures such
solutions, proves they are genuinely broken against inputs the benchmark never
shows, and reports what fraction the benchmark accepts anyway.

    from gameability import Problem, census, summarise, report

    rows = census([Problem("p1", reference_src, ["assert f(1) == 2"])])
    report(summarise(rows, "my benchmark"))

The only thing a benchmark must supply is (reference solution, visible tests).
"""
from .engine import Problem, audit, census, report, summarise
from .harden import witnesses

__all__ = ["Problem", "audit", "census", "summarise", "report", "witnesses"]
__version__ = "0.2.0"
