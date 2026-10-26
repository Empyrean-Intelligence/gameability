"""KNOWN-ANSWER PREFLIGHT for gameability probe.

Fake environments with rewards worked out by hand, one per scoring interface, so the probe must reproduce
answers we already know before its verdict on a real environment means anything:
  A  exact-match reward (old call API)          -> PASS, every probe 0.0
  B  reward that pays an empty answer (old API) -> GAMEABLE via empty
  C  substring reward on the state API          -> GAMEABLE via echo (the prompt contains the answer)
  D  judge-style reward that is always 0        -> UNREACHABLE, never GAMEABLE
  E  exact match on score_rollout(state, score_sem) (verifiers 0.1.8) -> PASS
No verifiers install and no model call is needed.

Run: python tests/test_probe.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gameability.probe import probe_env      # noqa: E402

FAILS = []


class Data:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        return [r[k] for r in self.rows] if isinstance(k, str) else self.rows[k]


ROWS = [{"prompt": [{"role": "user", "content": "Repeat the word %s." % w}], "answer": w}
        for w in ["apple", "river", "stone", "cloud", "ember"]]


def _text(completion):
    return completion if isinstance(completion, str) else completion[-1]["content"]


class OldRubric:
    def __init__(self, fn):
        self.fn = fn

    def get_reward_weights(self):
        return [1.0]

    async def score_rollout(self, prompt, completion, answer, state, task="default", info=None, **kw):
        state["timing"]["scoring_ms"] = 0.0          # what verifiers 0.1.4 does; a missing dict must not crash
        return {"reward": self.fn(_text(completion), answer)}


class StateRubric:
    def __init__(self, fn):
        self.fn = fn

    def get_reward_weights(self):
        return [1.0]

    async def score_rollout(self, state):
        state["reward"] = self.fn(_text(state["completion"]), state["answer"])


class SemRubric(StateRubric):                        # verifiers 0.1.8: score_rollout(state, score_sem)
    async def score_rollout(self, state, score_sem):
        async with score_sem:
            state["reward"] = self.fn(_text(state["completion"]), state["answer"])


class Env:
    def __init__(self, rubric, state_api=False):
        self.rubric = rubric
        self.eval_dataset = Data(ROWS)
        if state_api:
            self.init_state = lambda row, client, model: dict(row)


def check(name, res, status, gameable, probes_hit):
    got = (res["status"], res.get("gameable"), res.get("gameable_probes"))
    want = (status, gameable, probes_hit)
    print("%s %s: %s" % ("ok  " if got == want else "FAIL", name, got))
    if got != want:
        FAILS.append(name)
        print("     expected", want, "| error:", (res.get("error") or "")[-300:])


exact = lambda t, a: 1.0 if t.strip() == a else 0.0
pays_empty = lambda t, a: 1.0 if t.strip() in ("", a) else 0.0
substring = lambda t, a: 1.0 if a in t else 0.0

check("A exact match, old API", probe_env(Env(OldRubric(exact))), "MEASURED", False, [])
check("B pays empty, old API", probe_env(Env(OldRubric(pays_empty))), "MEASURED", True, ["1_empty"])
check("C substring, state API", probe_env(Env(StateRubric(substring), state_api=True), make_client=lambda: None),
      "MEASURED", True, ["3_echo"])
check("E exact match, score_sem API", probe_env(Env(SemRubric(exact), state_api=True), make_client=lambda: None),
      "MEASURED", False, [])
check("D always 0, state API", probe_env(Env(StateRubric(lambda t, a: 0.0), state_api=True), make_client=lambda: None),
      "UNREACHABLE", None, [])

print("\n%d failed" % len(FAILS) if FAILS else "\nall probe preflight cases pass")
sys.exit(1 if FAILS else 0)
