"""Compare routing strategies on synthetic developer prompts."""
import argparse
import math
import sys
import json
from pathlib import Path
import re
from statistics import median
from time import perf_counter
from unittest.mock import patch

from receptionist.config import load_config
from receptionist.models.errors import ProviderError
from receptionist.models.cloud import CloudModel
from receptionist.models.local import LocalModel
from receptionist.metrics import MetricsTracker
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
# Keep these public synthetic fixtures separate from the policy-only probe.
DATASET = Path(__file__).with_name("routing_benchmark_cases.json")
CASES += tuple(
    (row["id"], row["split"], row["prompt"], tuple(row["patterns"]), row["smoke"])
    for row in json.loads(DATASET.read_text(encoding="utf-8"))
)
PRIVATE_PROMPT = "Explain this synthetic internal API contract: PROJECT-TEST-123."


def smoke_answer(prompt, **kwargs):
    return next(answer for _, _, question, _, answer in CASES if question == prompt)


def evaluate(config, split, *, progress=False, strategies=("auto", "all-local", "all-cloud")):
    """Run each strategy on the same permitted cases, one request at a time."""
    cloud = CloudModel(config.get("cloud_model", {}), api_keys=config.get("api_keys") or {})
    model_key = f"{cloud.provider}/{cloud.name}"
    pricing = MetricsTracker(config.get("pricing", {}), model_key)
    selected = [case for case in CASES if split == "all" or case[1] == split]
    for strategy in strategies:
        rows = []
        for index, (case_id, _, prompt, patterns, _) in enumerate(selected, start=1):
            label = f"[{strategy} {index}/{len(selected)}] {case_id}"
            if progress:
                print(f"{label} starting", file=sys.stderr, flush=True)
            usage = []
            start = perf_counter()
            try:
                provider, reply, _, decision = route_query(
                    prompt, config, escalate=strategy == "all-cloud",
                    local_only=strategy == "all-local",
                    on_usage=lambda inputs, outputs: usage.append((inputs, outputs)),
                )
            except Exception:
                # Providers can raise before a response is returned.
                provider, reply, decision = None, "", {}
            elapsed = (perf_counter() - start) * 1000
            failed = isinstance(reply, ProviderError) or not reply.strip() or reply.startswith((
                "[Local ", "[Ollama ", "[OpenAI ", "[Provider ", "[Streaming ",
            ))
            passed = not failed and all(re.search(p, reply, re.I) for p in patterns)
            estimate = 0.0 if provider == "local" else None
            reported_cloud_usage = provider == "cloud" and bool(usage)
            if provider == "cloud" and (usage or not failed):
                inputs, outputs = usage[-1] if usage else (
                    len(prompt.split()) + len((config.get("system_prompt") or "").split()),
                    len(reply.split()),
                )
                estimate = pricing.estimate_cost(model_key, inputs, outputs)
                estimate = float(estimate) if estimate is not None else None
            if progress:
                outcome = "provider error" if failed else "passed" if passed else "needs review"
                print(f"{label} {provider or 'unknown'} · {elapsed / 1000:.2f}s · {outcome}",
                      file=sys.stderr, flush=True)
            rows.append((case_id, provider, failed, passed, elapsed, estimate, reported_cloud_usage, decision.get('reason')))
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
        reported_usage = sum(row[6] for row in rows)
        unknown = sum(row[1] is None for row in rows)
        estimates = [row[5] for row in rows if row[5] is not None]
        incomplete = len(rows) - len(estimates)
        cost = f"${sum(estimates):.6f}" + (f" partial ({incomplete} unknown)" if incomplete else "")
        print(f"{strategy}: quality {passed}/{len(rows)}, errors {failed}, "
              f"cloud routes {cloud}/{len(rows)} ({unknown} unknown), "
              f"cloud usage reported {reported_usage}/{cloud}, "
              f"median {median(latencies):.1f} ms, p95 {p95:.1f} ms, estimated cloud cost {cost}")
        if passed != len(rows):
            print("  Needs review:", ", ".join(row[0] for row in rows if not row[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write per-case metrics as JSON")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    parser.add_argument("--strategy", choices=("auto", "all-local", "all-cloud"),
                        help="Run just one strategy")
    parser.add_argument("--score-threshold", type=float, help="Override the Auto score threshold")
    parser.add_argument("--max-local-words", type=int, help="Override the Auto local word limit")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Run smoke checks without model calls")
    mode.add_argument("--verify", action="store_true", help="Check local generation and selected cloud usage")
    mode.add_argument("--live", action="store_true", help="Run models, including billable cloud calls")
    args = parser.parse_args()
    if args.score_threshold is not None and not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if args.max_local_words is not None and args.max_local_words < 1:
        parser.error("--max-local-words must be positive")

    config = load_config()
    thresholds = config.setdefault("classification", {}).setdefault("thresholds", {})
    if args.score_threshold is not None:
        thresholds["escalate_score_threshold"] = args.score_threshold
    if args.max_local_words is not None:
        thresholds["max_local_words"] = args.max_local_words
    strategies = (args.strategy,) if args.strategy else ("auto", "all-local", "all-cloud")
    check_local_only_policy(config)
    print("Local Only policy: passed with mocked providers")
    cloud = CloudModel(config.get("cloud_model", {}), api_keys=config.get("api_keys") or {})
    if args.verify or args.live:
        print(f"Live preflight: {cloud.provider}/{cloud.name}")
        if not cloud.api_key:
            parser.exit(1, f"Missing credential for {cloud.provider}.\n")
        usage = []
        _, cloud_reply, _, _ = route_query(
            "Reply with just OK.", config, escalate=True,
            on_usage=lambda inputs, outputs: usage.append((inputs, outputs)),
        )
        if isinstance(cloud_reply, ProviderError) or not cloud_reply.strip() or not usage:
            reason = str(cloud_reply) if isinstance(cloud_reply, ProviderError) else "missing reply or usage"
            parser.exit(1, f"Cloud preflight failed: {reason}\n")
        print("Cloud reply and provider usage verified")
        _, local_reply, _, _ = route_query("Reply with just OK.", config, local_only=True)
        if isinstance(local_reply, ProviderError) or not local_reply.strip():
            reason = str(local_reply) if isinstance(local_reply, ProviderError) else "missing reply"
            parser.exit(1, f"Local preflight failed: {reason}\n")
        print("Local reply verified")
        if args.verify:
            return

    if args.mock:
        print("Mock smoke run: quality, latency, and cost are illustrative")
        with patch.object(LocalModel, "generate", side_effect=smoke_answer), \
                patch.object(CloudModel, "generate", side_effect=smoke_answer):
            results = list(evaluate(config, args.split, strategies=strategies))
    else:
        print("Live run: cost uses reported usage when available, otherwise word-count estimates")
        results = list(evaluate(config, args.split, progress=True, strategies=strategies))

    print_results(results)
    if args.report:
        fields = ("case_id", "route", "provider_error", "passed", "latency_ms",
                  "estimated_cost_usd", "provider_usage_reported", "routing_reason")
        report = {
            "mode": "mock" if args.mock else "live",
            "split": args.split,
            "routing_threshold_overrides": thresholds,
            "cloud_model": f"{cloud.provider}/{cloud.name}",
            "local_model": LocalModel(config.get("local_model", {})).name,
            "results": {
                strategy: [dict(zip(fields, row)) for row in rows]
                for strategy, rows in results
            },
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.mock and any(not row[3] for _, rows in results for row in rows):
        parser.exit(1, "Benchmark smoke checks failed.\n")


if __name__ == "__main__":
    main()
