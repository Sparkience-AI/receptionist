"""Metrics tracking with per-1k token pricing and savings vs all-cloud baseline."""
import time
from collections import deque

class MetricsTracker:
	def __init__(self, pricing, carbon_enabled=True, max_history=200):
		self.pricing = pricing  # dict: provider/model -> {per_1k: float}
		self.carbon_enabled = carbon_enabled
		self.local_tokens = 0
		self.cloud_tokens = 0
		self.history = deque(maxlen=max_history)
		self.start_time = time.time()

	def _price_per_token(self, key: str) -> float:
		entry = self.pricing.get(key, {})
		per_1k = entry.get('per_1k') or entry.get('per1000') or 0.0
		return per_1k / 1000.0

	def record(self, prompt: str, response: str, model_type: str, tokens_used: int, model_key: str):
		if model_type == 'local':
			self.local_tokens += tokens_used
		else:
			self.cloud_tokens += tokens_used
		self.history.append({
			't': time.time(),
			'model_type': model_type,
			'tokens': tokens_used,
			'model_key': model_key
		})

	def get_metrics(self):
		total_tokens = self.local_tokens + self.cloud_tokens
		# Determine representative cloud baseline (choose highest priced OpenAI model if present else fall back)
		candidate_baseline = 'openai/gpt-5-preview' if 'openai/gpt-5-preview' in self.pricing else (
			'openai/gpt-4-turbo' if 'openai/gpt-4-turbo' in self.pricing else 'openai/gpt-4o-mini'
		)
		baseline_cost = total_tokens * self._price_per_token(candidate_baseline)
		# Actual cost: sum history cost
		actual_cost = 0.0
		for record in self.history:
			actual_cost += record['tokens'] * self._price_per_token(record['model_key'])
		savings = baseline_cost - actual_cost
		carbon_proxy = (self.local_tokens * 0.00001) if self.carbon_enabled else 0.0
		return {
			'total_tokens': total_tokens,
			'cloud_tokens': self.cloud_tokens,
			'local_tokens': self.local_tokens,
			'baseline_cost': baseline_cost,
			'actual_cost': actual_cost,
			'savings': savings,
			'carbon_proxy': carbon_proxy,
			'history': list(self.history)
		}
