"""Compare routing strategies on synthetic developer prompts."""
import argparse
import math
import re
from statistics import median
from time import perf_counter
from unittest.mock import patch

from receptionist.config import load_config
from receptionist.models.errors import ProviderError
from receptionist.models.cloud import CloudModel
from receptionist.models.local import LocalModel
from receptionist.router import route_query


# ID, split, prompt, required regexes, smoke-test answer
CASES = (
    ("git-status", "dev", "Which Git command shows the working tree status?",
     (r"\bgit status\b",), "Use git status."),
    ("python-length", "dev", "How do I get the length of a Python list?",
     (r"\blen\(",), "Use len(items)."),
    ("python-slice", "dev", "What does items[::-1] do in Python?",
     (r"\brevers",), "It reverses the order in a new slice."),
    ("sql-null", "dev", "Why does WHERE x = NULL return no matches in SQL?",
     (r"\bunknown\b", r"\bIS NULL\b"), "The comparison is unknown. Use IS NULL."),
    ("api-contract", "dev", "What is an API contract in software development?",
     (r"\brequests?\b", r"\bresponses?\b"), "It defines requests and responses."),
    ("python-default", "holdout", "Why does def add(x, xs=[]): keep old list entries?",
     (r"\bonce\b|\bshared\b", r"\bNone\b"), "The default list is shared. Use None."),
    ("git-branch", "holdout", "Which Git command shows the current branch name only?",
     (r"\bgit branch --show-current\b|\bgit rev-parse --abbrev-ref HEAD\b",),
     "Use git branch --show-current."),
    ("python-alias", "holdout", "Why does b = a mean b.append(1) changes a too?",
     (r"\bsame (list|object)\b|\balias",), "Both names refer to the same list."),
)
PRIVATE_PROMPT = "Explain this synthetic internal API contract: PROJECT-TEST-123."


def smoke_answer(prompt, **kwargs):
    return next(answer for _, _, question, _, answer in CASES if question == prompt)


def evaluate(config, split):
    """Run each strategy on the same permitted cases, one request at a time."""
    for strategy in ("auto", "all-local", "all-cloud"):
        rows = []
        for case_id, case_split, prompt, patterns, _ in CASES:
            if split != "all" and split != case_split:
                continue
            start = perf_counter()
            try:
                provider, reply, _, _ = route_query(
                    prompt, config, escalate=strategy == "all-cloud",
                    local_only=strategy == "all-local",
                )
            except Exception:
                # Providers can raise before a response is returned.
                provider, reply = None, ""
            elapsed = (perf_counter() - start) * 1000
            failed = isinstance(reply, ProviderError) or not reply.strip() or reply.startswith((
                "[Local ", "[Ollama ", "[OpenAI ", "[Provider ", "[Streaming ",
            ))
            passed = not failed and all(re.search(p, reply, re.I) for p in patterns)
            estimate = 0.0 if provider == "local" else None
            if provider == "cloud" and not failed:
                model = config.get("cloud_model", {})
                key = f"{model.get('provider', 'openai')}/{model.get('name', 'gpt-4o-mini')}"
                rate = config.get("pricing", {}).get(key, {}).get("per_1k")
                if rate is not None:
                    words = sum(len(text.split()) for text in (
                        prompt, reply, config.get("system_prompt") or "",
                    ))
                    estimate = rate * words / 1000
            rows.append((case_id, provider, failed, passed, elapsed, estimate))
        yield strategy, rows


def check_local_only_policy(config):
    """Exercise policy with a cloud-triggering fixture and no live providers."""
    with patch.object(LocalModel, "generate", return_value="local") as local, \
            patch("receptionist.router.CloudModel") as cloud:
        for override in (False, True):
            provider, _, _, decision = route_query(
                PRIVATE_PROMPT, config, local_only=True, escalate=override,
            )
            if provider != "local" or decision["reason"] != "local only":
                raise RuntimeError("Local Only returned an unexpected route")
        if local.call_count != 2 or cloud.called:
            raise RuntimeError("Local Only invoked an unexpected provider")


def print_results(results):
    for strategy, rows in results:
        latencies = sorted(row[4] for row in rows)
        p95 = latencies[math.ceil(len(latencies) * .95) - 1]
        failed = sum(row[2] for row in rows)
        passed = sum(row[3] for row in rows)
        cloud = sum(row[1] == "cloud" for row in rows)
        unknown = sum(row[1] is None for row in rows)
        estimates = [row[5] for row in rows if row[5] is not None]
        incomplete = len(rows) - len(estimates)
        cost = f"${sum(estimates):.6f}" + (f" partial ({incomplete} unknown)" if incomplete else "")
        print(f"{strategy}: quality {passed}/{len(rows)}, errors {failed}, "
              f"cloud routes {cloud}/{len(rows)} ({unknown} unknown), "
              f"median {median(latencies):.1f} ms, p95 {p95:.1f} ms, estimated cloud cost {cost}")
        if passed != len(rows):
            print("  Needs review:", ", ".join(row[0] for row in rows if not row[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Run smoke checks without model calls")
    mode.add_argument("--live", action="store_true", help="Run models, including billable cloud calls")
    args = parser.parse_args()
    config = load_config()
    check_local_only_policy(config)
    print("Local Only policy: passed with mocked providers")
    if args.mock:
        print("Mock smoke run: quality, latency, and cost are illustrative")
        with patch.object(LocalModel, "generate", side_effect=smoke_answer), \
                patch.object(CloudModel, "generate", side_effect=smoke_answer):
            print_results(evaluate(config, args.split))
    else:
        print("Live run: cost uses approximate word counts and configured pricing")
        print_results(evaluate(config, args.split))


if __name__ == "__main__":
    main()
