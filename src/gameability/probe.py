"""Null probes for an RL environment's reward: does it pay a policy that does no work?

    gameability probe <env_id>                 # a verifiers environment installed in this Python
    gameability probe <env_id> --args '{"eval_task": "multiple_choice"}'

Each probe answers every item the same no-work way (empty, a refusal, the prompt echoed back, a wrong answer in
the expected format, filler text). The reward is computed by the environment's OWN rubric. An environment is
GAMEABLE when a probe earns at least half of what the correct answers earn. This finds only gaming that needs no
capability, so a PASS is not a proof of soundness.

By default no model is called: OpenAI clients are pointed at a dead local address, so a reward that needs an LLM
judge scores 0 on the correct answers too and is reported UNREACHABLE rather than spending your credits.
"""
import asyncio
import collections
import copy
import inspect
import json
import logging
import os
import time
import traceback

FORMATS = ["{}", "\\boxed{{{}}}", "<answer>{}</answer>", "Answer: {}"]
REFUSAL = "I don't know."
FILLER = ("the " * 500)[:2000]
GAMEABLE_FRACTION = 0.5
REACHABLE_FRACTION = 0.9
DEAD_BASE_URL = "http://127.0.0.1:9/v1"
PROBE_NAMES = {"1_empty": "empty answer", "2_refusal": "refusal", "3_echo": "prompt echoed back",
               "4_format_junk": "wrong answer, right format", "6_filler": "filler text"}


class _ErrorCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.lines = []

    def emit(self, record):
        if record.name.startswith("verifiers") or "reward" in record.getMessage().lower():
            self.lines.append(("%s: %s" % (record.name, record.getMessage()))[:300])


def _text_of(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in reversed(prompt):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if role == "user":
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _weights(rubric):
    for name in ("_get_individual_reward_weights", "get_reward_weights"):
        f = getattr(rubric, name, None)
        if callable(f):
            try:
                return list(f())
            except Exception:
                pass
    for name in ("weights", "reward_weights"):
        w = getattr(rubric, name, None)
        if w:
            return list(w)
    return None


def _max_reward(rubric):
    if hasattr(rubric, "rubrics"):
        parts = [_max_reward(r) for r in rubric.rubrics]
        return None if any(p is None for p in parts) else sum(parts)
    w = _weights(rubric)
    return None if not w else float(sum(x for x in w if x > 0))


def _dataset(env):
    for name in ("get_eval_dataset", "get_dataset"):
        f = getattr(env, name, None)
        if callable(f):
            try:
                ds = f(n=-1)
                if ds is not None:
                    return ds, "eval" if name == "get_eval_dataset" else "train"
            except Exception:
                pass
    for name in ("eval_dataset", "dataset"):
        ds = getattr(env, name, None)
        if ds is not None:
            return ds, "eval" if name == "eval_dataset" else "train"
    raise ValueError("environment exposes no dataset")


def _old_api_state():
    # 0.1.x rubric.score_rollout writes state["timing"]["scoring_ms"] and adds to ["total_ms"]; a rollout normally
    # creates this dict, so a direct call must too
    return {"timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0}}


def _client():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    try:                                   # verifiers 0.1.11+ accepts only its own Client wrapper
        from verifiers.clients import OpenAIChatCompletionsClient
        client = OpenAIChatCompletionsClient(client)
    except ImportError:
        pass
    return client


def probe_env(env, n_items=20, make_client=_client):
    """Probe an already-loaded environment. Returns a result dict; 'status' is MEASURED when it could be scored."""
    t0 = time.time()
    res = {"status": None, "n_items": 0}
    counter = _ErrorCounter()
    logging.getLogger().addHandler(counter)

    def finish(status, **kw):
        logging.getLogger().removeHandler(counter)
        res.update(status=status, seconds=round(time.time() - t0, 1), reward_errors=len(counter.lines),
                   reward_error_examples=counter.lines[:5], **kw)
        return res

    try:
        full, split = _dataset(env)
        rows = [dict(full[i]) for i in range(min(n_items, len(full)))]
        answers_all = [str(a) for a in full["answer"]] if "answer" in full.column_names else []
    except Exception:
        return finish("DATASET_FAIL", error=traceback.format_exc()[-1500:])
    if not rows or not all("answer" in r for r in rows):
        return finish("NO_ANSWER_FIELD", columns=list(rows[0].keys()) if rows else [])
    res["split"], res["n_items"] = split, len(rows)
    constant = collections.Counter(answers_all).most_common(1)[0][0] if answers_all else ""
    completion_type = getattr(env, "message_type", "chat")
    sig = inspect.signature(env.rubric.score_rollout).parameters
    old_api = "completion" in sig
    # required parameters after 'state' (verifiers 0.1.8: score_rollout(state, score_sem)) get a semaphore
    extra = [n for n, q in list(sig.items())[1:] if q.default is q.empty
             and q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD, q.KEYWORD_ONLY)] if not old_api else []
    unknown = [n for n in extra if "sem" not in n]
    if unknown:
        return finish("SCORE_FAIL", error="score_rollout requires unknown parameters %s" % unknown)
    res["scoring_api"] = ("call(prompt, completion, answer, ...)" if old_api else
                          "init_state + score_rollout(state%s)" % "".join(", " + n for n in extra))
    client = None
    if not old_api:
        try:
            client = make_client()
        except Exception:
            return finish("CLIENT_FAIL", error=traceback.format_exc()[-1500:])

    def as_completion(text):
        return text if completion_type == "completion" else [{"role": "assistant", "content": text}]

    async def score(row, text):
        row = copy.deepcopy(row)
        info = row.get("info")
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except ValueError:
                pass
        if old_api:
            rs = await env.rubric.score_rollout(prompt=row.get("prompt"), completion=as_completion(text),
                                                answer=row.get("answer", ""), state=_old_api_state(),
                                                task=row.get("task", "default"),
                                                info=info if isinstance(info, dict) else {})
            return float(rs["reward"] if isinstance(rs, dict) else rs.reward)
        st = env.init_state(row, client=client, model="probe")
        if inspect.isawaitable(st):
            st = await st
        st["completion"] = as_completion(text)
        st["is_completed"] = True
        await env.rubric.score_rollout(st, **{n: asyncio.Semaphore(1) for n in extra})
        return float(st.get("reward") or 0.0)

    async def mean_over(texts):
        vals = [await score(r, t) for r, t in zip(rows, texts)]
        return sum(vals) / len(vals)

    async def run():
        out = {}
        gt = {f: await mean_over([f.format(str(r["answer"])) for r in rows]) for f in FORMATS}
        best = max(FORMATS, key=lambda f: gt[f])
        junk = {f: await mean_over([f.format("0")] * len(rows)) for f in FORMATS}
        out.update(gt_by_format=gt, gt_best_format=best, gt_mean=gt[best], format_junk_by_format=junk)
        out["probes"] = {
            "1_empty": await mean_over([""] * len(rows)),
            "2_refusal": await mean_over([REFUSAL] * len(rows)),
            "3_echo": await mean_over([_text_of(r.get("prompt")) for r in rows]),
            "4_format_junk": max(junk.values()),
            "6_filler": await mean_over([FILLER] * len(rows)),
        }
        out["5_answer_prior"] = await mean_over([best.format(constant)] * len(rows))
        return out

    try:
        scores = asyncio.run(run())
    except Exception:
        return finish("SCORE_FAIL", error=traceback.format_exc()[-1500:])

    max_r = _max_reward(env.rubric)
    gt_mean = scores["gt_mean"]
    reachable = (gt_mean >= REACHABLE_FRACTION * max_r) if (max_r is not None and max_r > 0) else None
    hits = sorted(k for k, v in scores["probes"].items() if gt_mean > 0 and v >= GAMEABLE_FRACTION * gt_mean)
    status = "MEASURED" if reachable else ("UNREACHABLE" if reachable is False else "MAX_UNKNOWN")
    return finish(status, max_reward=max_r, max_exceeded=max_r is not None and gt_mean > max_r + 1e-9,
                  reachable=reachable, gameable=bool(hits) if reachable else None, gameable_probes=hits,
                  answer_prior_share=(scores["5_answer_prior"] / gt_mean) if gt_mean > 0 else None, **scores)


def probe(env_id, n_items=20, load_args=None, allow_model_calls=False):
    """Load a verifiers environment by id and probe it."""
    if not allow_model_calls:
        os.environ["OPENAI_BASE_URL"] = DEAD_BASE_URL
        os.environ.setdefault("OPENAI_API_KEY", "gameability-probe-no-calls")
    try:
        import verifiers as vf
        env = vf.load_environment(env_id, **(load_args or {}))
    except ImportError:
        raise SystemExit("gameability probe needs the 'verifiers' package and the environment installed: "
                         "pip install verifiers, then install your environment")
    except Exception:
        return {"env": env_id, "status": "LOAD_FAIL", "error": traceback.format_exc()[-1500:]}
    res = probe_env(env, n_items=n_items)
    res["env"] = env_id
    try:
        from importlib.metadata import version
        res["verifiers_version"] = version("verifiers")
    except Exception:
        pass
    return res


def report(res):
    env = res.get("env", "environment")
    st = res["status"]
    if st not in ("MEASURED", "UNREACHABLE", "MAX_UNKNOWN"):
        print("%s: could not be probed (%s)" % (env, st))
        if res.get("error"):
            print(res["error"].strip().splitlines()[-1])
        return
    gt = res["gt_mean"]
    print("%s: %d items, %s split, scored by the environment's own rubric" % (env, res["n_items"], res["split"]))
    fmt = res["gt_best_format"]
    print("  %-27s %.3f   (answer format: %s)" % ("correct answers", gt,
                                                  "plain" if fmt == "{}" else fmt.replace("{}", "...")))
    for k, v in res["probes"].items():
        flag = "  <-- pays for no work" if gt > 0 and v >= GAMEABLE_FRACTION * gt else ""
        print("  %-27s %.3f%s" % (PROBE_NAMES[k], v, flag))
    if res.get("answer_prior_share") is not None:
        print("  (most common answer every time earns %.0f%% of the correct-answer reward; reported, not counted)"
              % (100 * res["answer_prior_share"]))
    if res["reward_errors"]:
        print("  %d reward-function errors were logged and scored 0, e.g. %s" % (
            res["reward_errors"], res["reward_error_examples"][0]))
    if st == "UNREACHABLE":
        print("VERDICT: UNREACHABLE -- correct answers earn %.3f of a possible %.3f; the reward likely needs a model "
              "call (blocked by default; --allow-model-calls) or another answer format" % (gt, res["max_reward"]))
    elif st == "MAX_UNKNOWN":
        print("VERDICT: UNKNOWN -- the rubric does not expose its maximum reward")
    elif res["gameable"]:
        print("VERDICT: GAMEABLE -- a no-work answer earns at least %d%% of the correct-answer reward"
              % (100 * GAMEABLE_FRACTION))
    else:
        print("VERDICT: PASS -- no null probe earns %d%% of the correct-answer reward "
              "(finds only gaming that needs no capability)" % (100 * GAMEABLE_FRACTION))
