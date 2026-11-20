"""Cumulative session usage and estimated model spend."""
from collections import defaultdict, deque
from decimal import Decimal
from time import time


class MetricsTracker:
    def __init__(self, pricing, baseline_model, max_history=200):
        self.pricing = pricing
        self.baseline_model = baseline_model
        self.history = deque(maxlen=max_history)
        self.usage = defaultdict(lambda: [0, 0])
        self.requests = {"local": 0, "cloud": 0}
        self.estimated_requests = 0

    def _rates(self, model_key):
        entry = self.pricing.get(model_key, {})
        flat = entry.get("per_1k", entry.get("per1000"))
        return tuple(entry.get(field, flat) for field in (
            "input_per_1k", "output_per_1k",
        ))

    def _cost(self, model_key, input_tokens, output_tokens):
        rates = self._rates(model_key)
        if None in rates:
            return None
        return (Decimal(str(rates[0])) * input_tokens
                + Decimal(str(rates[1])) * output_tokens) / 1000

    def record(self, model_type, model_key, input_tokens, output_tokens, *, estimated):
        totals = self.usage[model_type, model_key]
        totals[0] += input_tokens
        totals[1] += output_tokens
        self.requests[model_type] += 1
        self.estimated_requests += int(estimated)
        self.history.append({
            "t": time(), "model_type": model_type, "model_key": model_key,
            "tokens": input_tokens + output_tokens,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "usage_estimated": estimated,
        })

    def _cloud_spend(self):
        total = Decimal(0)
        for (model_type, model_key), (inputs, outputs) in self.usage.items():
            if model_type == "cloud":
                cost = self._cost(model_key, inputs, outputs)
                if cost is None:
                    return None
                total += cost
        return total

    def cloud_budget_reason(self, budget_usd, cloud_model_key):
        if budget_usd is None:
            return None
        budget = Decimal(str(budget_usd))
        if not budget.is_finite() or budget < 0:
            raise ValueError('cloud_budget_usd must be finite and nonnegative')
        spend = self._cloud_spend()
        if spend is None or None in self._rates(cloud_model_key):
            return "cloud pricing unavailable"
        if spend >= budget:
            return "session cloud budget reached"
        return None

    def get_metrics(self):
        local_tokens = sum(sum(counts) for (kind, _), counts in self.usage.items() if kind == "local")
        cloud_tokens = sum(sum(counts) for (kind, _), counts in self.usage.items() if kind == "cloud")
        inputs = sum(counts[0] for counts in self.usage.values())
        outputs = sum(counts[1] for counts in self.usage.values())
        baseline = self._cost(self.baseline_model, inputs, outputs)
        actual = self._cloud_spend()
        requests = sum(self.requests.values())
        return {
            "total_tokens": local_tokens + cloud_tokens,
            "local_tokens": local_tokens, "cloud_tokens": cloud_tokens,
            "input_tokens": inputs, "output_tokens": outputs,
            "baseline_cost": baseline, "actual_cost": actual,
            "savings": baseline - actual if baseline is not None and actual is not None else None,
            "local_request_share": self.requests["local"] / requests if requests else 0,
            "estimated_requests": self.estimated_requests,
            "history": list(self.history),
        }
