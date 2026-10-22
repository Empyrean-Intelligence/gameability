# gameability

**Find the problems in a code benchmark that accept broken solutions.**

When a model learns from tests, the tests are its reward. A self-improving system trained on tests that
accept broken code learns to pass weak tests, not to write correct code. `gameability` measures how often
that can happen. It is part of the verification layer we use at Empyrean Intelligence for self-improvement
work, released for anyone who builds benchmarks or RL training sets.

A benchmark score means something only if the benchmark rejects solutions that lack the
skill it measures. `gameability` checks that directly. It takes each problem's reference
solution, breaks it in one small place, confirms the result really is wrong, and then
asks whether the problem's own tests notice.

```
pip install "gameability[hf]"
gameability check mbpp
```

On MBPP, **17.9% of problems (169 of 942 judged) accept a broken solution**; on HumanEval,
**10.3% (16 of 155)**. Each run takes minutes on a laptop, with no model and no API key.

**Confirmed on valid inputs.** On the problems EvalPlus also covers, every accepted broken solution
was re-run against EvalPlus's independent tests, whose inputs respect each problem's rules:
**95.2% of flagged MBPP problems and 84.6% of flagged HumanEval problems are confirmed** (the broken
solution passes the original tests and fails EvalPlus). The two unconfirmed HumanEval problems
(116 and 129) are ones whose own visible tests contradict their written specification.

**What changed in v0.2.** v0.1 built fresh inputs by changing values freely, which sometimes asked
a problem about inputs it never promised to handle (an unsorted list for binary search, random
letters for a parenthesis parser). v0.2 learns each argument's domain from the problem's own
visible tests and stays inside it: number signs and minimums, the characters or whole tokens a
string uses, and whether lists are sorted, unique, rectangular or a permutation. On the same
problems this raised the confirmed share from 85.7% to 95.2% (MBPP) and 78.6% to 84.6% (HumanEval)
while keeping 93.1% of v0.1's confirmed detections. `--inputs free` reproduces v0.1.

## Scoreboard

Problems that accept at least one broken solution (wrong on at least one fresh input), with 95% intervals. Defaults (24 broken
versions per problem, 8 fresh inputs, seed 0), laptop CPU, October 2026.

| Benchmark | Weak problems | Command |
|---|---|---|
| MBPP | 17.9% (169 of 942) [15.6, 20.5] | `gameability check mbpp` |
| MBPP sanitized | 13.8% (57 of 413) [10.8, 17.5] | `gameability check mbpp-sanitized` |
| HumanEval | 10.3% (16 of 155) [6.5, 16.1] | `gameability check humaneval` |
| KodCode (RL training set, 1,000-row random sample) | 9.7% (45 of 466) [7.3, 12.7] | `gameability check kodcode:rows.jsonl` |
| HumanEval+ (EvalPlus) | 3.4% (5 of 147) [1.5, 7.7] | `gameability check humaneval-plus` |
| MBPP+ (EvalPlus) | 2.0% (7 of 353) [1.0, 4.0] | `gameability check mbpp-plus` |

EvalPlus's stronger tests look like a large improvement, but EvalPlus also rewrote most reference
solutions. Holding the original reference fixed (`mbpp-plus-origref`, `humaneval-plus-origref`), the weak
share roughly halves (MBPP 8.7% to 3.1% on 254 problems, HumanEval 10.6% to 4.9% on 123), and at these
sample sizes the intervals overlap. KodCode's judged set is the problems whose tests call the function with
literal arguments (466 of 801 loadable). `rows.jsonl` is KodCode rows saved as JSON lines; the sample above
is reproducible with `python tests/sample_kodcode.py rows.jsonl` (positions from `random.Random(0)`).

## What it prints

```
  415 -- 6 of 11 broken solutions pass every visible test
      -    if arr_len < 2:
      +    if arr_len <= 2:
      a test that catches it: assert max_Product([4, 9]) == (4, 9)
```

For every weak problem you get the one line that was changed, how many broken versions got
through, and an assert that would have caught one.

**`--fixes fixes.py` writes those asserts to a file; review them before adding them.** Measured on
34 MBPP/HumanEval problems against a 1.5B model's real solutions: v0.2's suggested asserts wrongly
reject **2.6% of its correct solutions** (8 of 307; v0.1: 15.3%). They target the broken versions
gameability builds, not the mistakes a particular model makes: they caught 4 of that model's 24
real wrong solutions. Treat them as leads, not as a test suite.

## How it works

For each problem the benchmark supplies (reference solution, visible tests):

1. The reference must pass its own visible tests. If it doesn't, the problem is excluded and counted.
2. Fresh inputs are built by varying the arguments in the visible tests. The reference's
   outputs on them are the hidden answer key.
3. The reference is mutated at a single site: a comparison flipped, an operator swapped, a
   constant nudged, a `return` replaced with `return None`.
4. A mutant counts as **broken** only if it gives a different answer from the reference on
   at least one fresh input. Mutants that behave identically are thrown away, so they can't
   inflate the number.
5. A broken mutant that still passes every visible test is a **survivor**. A problem with at
   least one survivor accepts a wrong solution.

Benchmark code runs in a child process with a step budget, so a runaway mutant is stopped at
the same point on any machine; a wall-clock limit is only a backstop. Mutants that never
finish are reported as truncated, never counted as rejected.

## Your own benchmark

One problem per line in a `.jsonl` file:

```json
{"id": "clamp", "reference": "def clamp(x, lo, hi): ...", "tests": ["assert clamp(5, 0, 10) == 5"]}
```

Optional fields: `entry` (the function name, if it can't be inferred), `setup` (code run
before the tests), `hidden` (held-out asserts you already have; they are used as extra
answer-key inputs), `call_alias` (the name the tests call the solution by, e.g. `candidate`).

```
gameability check examples/tiny.jsonl --fixes fixes.py --json report.json
```

## What the number means, and what it doesn't

- **The attacker is weak.** One small edit is the simplest possible way to be wrong; a model trained
  against these tests can search far more. In that sense the rates are a floor.
- **Fresh inputs can still break a problem's rules in ways the tests don't reveal.** v0.2 keeps each
  argument inside its observed domain, but it does not model relationships BETWEEN arguments (a
  length argument that must equal a list's length) or rules that only the docstring states. On the
  problems EvalPlus covers, 4.8% of flagged MBPP and 15.4% of flagged HumanEval problems are still
  unconfirmed (v0.1: 14.3% and 21.4%).
- **It does not predict where a model will cheat.** On a 1.5B coder, false passes were no more
  common on flagged problems than on others (8.4% vs 10.0%). It audits the tests' strength, not a
  model's behaviour.
- **It audits the tests, not any model.** Nothing here says a particular model cheated. It says
  which problems would not have caught it.
- **It runs the benchmark's code.** Reference solutions, mutants and tests are executed, each job in
  its own throwaway directory (some benchmark tasks write files; v0.1 let them land in your working
  folder). That is isolation from your files, not a security sandbox: only run it on benchmarks
  you'd be willing to execute.

## Reproducing the headline numbers

```
gameability check mbpp --json mbpp.json
gameability check humaneval --json humaneval.json
```

Defaults: 24 mutants per problem, 8 fresh inputs, seed 0, contract inputs. Add `--inputs free` to
reproduce v0.1's numbers (MBPP 18.5%, HumanEval 11.8%). The known-answer tests run with
`python tests/test_gameability.py` and `python tests/test_contract_inputs.py`.

## License

MIT
