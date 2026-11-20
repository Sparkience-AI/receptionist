"""Advanced classifier for routing decisions.

Goals:
 - Keep latency extremely low (<1ms Python-side) via lightweight regex + scoring.
 - Provide structured explanation & confidence for UI transparency & future learning.
 - Conservative safety-first escalation.
 - Adaptable thresholds from config, with sane defaults.
 - Caching to avoid recomputing repeated prompts within a session.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import re, math, time
from collections import OrderedDict

# === Patterns & Keyword Sets ===
SAFETY_KEYWORDS = [
	'suicide','self-harm','kill myself','overdose','weapon','bomb','terror','radicalize',
	'medical diagnosis','prescription','diagnose','legal advice','lawsuit','contract','financial advice','invest','stock tip'
]

GREETING_PATTERN = re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|goodbye|yes|no|sure|cool|awesome|great|morning|afternoon|evening)[!., ]*$", re.I)
CODE_REGEX = re.compile(r"```|class\s+\w+|def\s+\w+\(|import\s+\w+|SELECT\s+.+FROM|function\s+\w+\(|console\.log|</?\w+>", re.I)
MATH_REGEX = re.compile(r"\b(integral|derivative|sum\(|sigma|sqrt|log\(|ln\(|probability|p-value)\b", re.I)
DATA_REGEX = re.compile(r"\b(dataset|dataframe|csv|parquet|regression|correlation|scatter|histogram)\b", re.I)
COMPLEX_MARKERS = [
	'explain','summarize','analyze','compare','optimize','strategy','design','architecture','refactor','trade-off',
	'algorithm','big o','complexity','derivation','proof','theorem','regression','neural','model','dataset','statistical',
	'benchmark','performance','scalability','concurrency','deadlock','race condition','encryption','cryptography','security'
]

DEFAULT_THRESHOLDS = {
	'max_local_tokens': 18,          # short prompts more likely local
	'max_local_words': 28,           # fallback length heuristic
	'escalate_score_threshold': 0.52,
	'confidence_margin': 0.08        # distance from decision boundary to consider high confidence
}

WEIGHTS = {
	'code': 0.30,
	'complex': 0.22,
	'question': 0.15,
	'length': 0.12,
	'math': 0.12,
	'data': 0.09
}

@dataclass
class ClassificationResult:
	decision: str            # 'local' | 'cloud'
	confidence: float        # 0..1 relative certainty distance from boundary
	escalate_score: float    # raw composite score 0..1
	reason: str
	features: Dict[str, Any]
	cached: bool = False
	latency_ms: float = 0.0

	def to_dict(self):
		d = asdict(self)
		return d

# Simple LRU cache
_CACHE_MAX = 256
_cache: OrderedDict[str, ClassificationResult] = OrderedDict()

def _cache_get(key: str) -> ClassificationResult | None:
	res = _cache.get(key)
	if res:
		_cache.move_to_end(key)
	return res

def _cache_put(key: str, value: ClassificationResult):
	_cache[key] = value
	_cache.move_to_end(key)
	if len(_cache) > _CACHE_MAX:
		_cache.popitem(last=False)

def _normalize_len(word_count: int, cap: int = 120) -> float:
	return min(word_count / cap, 1.0)

def classify(prompt: str, config: Dict[str, Any] | None = None) -> ClassificationResult:
	start = time.perf_counter()
	raw = prompt.strip()
	low = raw.lower()
	if not raw:
		return ClassificationResult('local', 1.0, 0.0, 'empty prompt', {'empty': True})

	cached = _cache_get(low)
	if cached:
		# return a shallow copy marking cached to ensure latency stable
		return ClassificationResult(**{**cached.to_dict(), 'cached': True, 'latency_ms': (time.perf_counter()-start)*1000})

	thresholds = DEFAULT_THRESHOLDS.copy()
	if config and 'classification' in config:
		thresholds.update(config['classification'].get('thresholds', {}))

	words = raw.split()
	word_count = len(words)
	token_like = word_count  # simple approx

	# Feature detection
	safety_hit = any(kw in low for kw in SAFETY_KEYWORDS)
	greeting = word_count <= 10 and GREETING_PATTERN.match(low) is not None
	has_code = bool(CODE_REGEX.search(raw))
	has_math = bool(MATH_REGEX.search(raw))
	has_data = bool(DATA_REGEX.search(raw))
	is_question = '?' in raw
	complex_hits = sum(1 for m in COMPLEX_MARKERS if m in low)
	complex_density = complex_hits / max(word_count, 1)

	# Immediate decisions
	if safety_hit:
		result = ClassificationResult('cloud', 1.0, 1.0, 'safety escalation', {
			'safety_hit': True
		}, latency_ms=(time.perf_counter()-start)*1000)
		_cache_put(low, result)
		return result
	if greeting:
		result = ClassificationResult('local', 0.95, 0.0, 'greeting / acknowledgment', {
			'greeting': True
		}, latency_ms=(time.perf_counter()-start)*1000)
		_cache_put(low, result)
		return result

	# Scoring components
	score_components = {
		'code': 1.0 if has_code else 0.0,
		'complex': min(complex_density * 4, 1.0),  # amplify density
		'question': 1.0 if (is_question and not greeting) else 0.0,
		'length': _normalize_len(word_count),
		'math': 1.0 if has_math else 0.0,
		'data': 1.0 if has_data else 0.0
	}
	escalate_score = sum(score_components[k] * WEIGHTS[k] for k in WEIGHTS)

	# Decision boundary
	boundary = thresholds['escalate_score_threshold']
	decision = 'cloud' if escalate_score >= boundary else 'local'
	# Confidence = distance from boundary scaled by maximum possible weight sum (1)
	confidence = abs(escalate_score - boundary) + (0.05 if decision == 'cloud' and has_code else 0.0)
	confidence = min(confidence, 1.0)

	# Additional length guardrail: if predicted local but quite long, escalate
	if decision == 'local' and word_count > thresholds['max_local_words']:
		decision = 'cloud'
		reason = 'length override'
	else:
		reason = 'score >= boundary' if decision == 'cloud' else 'score < boundary'

	features = {
		'word_count': word_count,
		'has_code': has_code,
		'complex_density': round(complex_density, 3),
		'is_question': is_question,
		'has_math': has_math,
		'has_data': has_data,
		'components': score_components,
		'boundary': boundary
	}
	result = ClassificationResult(decision, round(confidence,4), round(escalate_score,4), reason, features, latency_ms=(time.perf_counter()-start)*1000)
	_cache_put(low, result)
	return result

def classify_prompt(prompt: str) -> str:  # backward compatibility simple label
	res = classify(prompt)
	return 'critical' if res.decision == 'cloud' else 'trivial'

