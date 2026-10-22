"""Draw the pre-registered KodCode sample: 1,000 rows of the train split at positions
chosen by random.Random(0).sample(range(N), 1000), fetched one row at a time through the
Hugging Face datasets-server (the full dataset does not fit this laptop).

Positions are fixed before any row is read or audited. Rows are written as jsonl for local
use only (KodCode is CC BY-NC 4.0: aggregate results may be reported, rows are not redistributed).

Run: python tests/sample_kodcode.py OUT.jsonl
"""
import json
import random
import sys
import time
import urllib.request

N = 484097
URL = ("https://datasets-server.huggingface.co/rows?dataset=KodCode/KodCode-V1"
       "&config=default&split=train&offset=%d&length=1")


PAUSE = 3.0      # seconds between requests: the unauthenticated API rate-limits (HTTP 429) at full speed


def fetch(i, tries=8):
    for k in range(tries):
        try:
            with urllib.request.urlopen(URL % i, timeout=60) as r:
                rows = json.load(r)["rows"]
            assert rows and rows[0]["row_idx"] == i, "row index mismatch at %d" % i
            time.sleep(PAUSE)
            return rows[0]["row"]
        except Exception as e:
            last = e
            # a 429 needs a real cool-down, not a quick retry
            time.sleep(30 * (k + 1) if "429" in repr(e) else 2 * (k + 1))
    raise SystemExit("could not fetch row %d: %r" % (i, last))


def main(out):
    idx = sorted(random.Random(0).sample(range(N), 1000))
    done = set()
    try:
        with open(out, encoding="utf-8") as f:
            done = {json.loads(l)["_row_idx"] for l in f if l.strip()}
    except FileNotFoundError:
        pass
    with open(out, "a", encoding="utf-8") as f:
        for n, i in enumerate(idx):
            if i in done:
                continue
            row = fetch(i)
            row["_row_idx"] = i
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 100 == 0:
                print("fetched %d/1000" % n, flush=True)
    print("SAMPLE COMPLETE: 1000 rows, positions from random.Random(0)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
