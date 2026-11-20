"""Routing logic with structured classification, streaming, and explanation."""
from __future__ import annotations
from .classifier import classify, ClassificationResult
from .models.errors import ProviderError
from .models.local import LocalModel
from .models.cloud import CloudModel
from .config import load_config

def route_query(prompt: str, config=None, escalate: bool=False, stream: bool=False, *, local_only: bool=False, cloud_budget_reason=None, on_usage=None):
	"""Route a prompt.
	Local Only takes precedence over escalation. Existing callers default to Auto.
	Returns (model_type, response_or_generator, model_label, classification_dict)
	model_type: 'local' | 'cloud'
	response_or_generator: str, generator of str tokens, or ProviderError for a synchronous failure
	classification_dict: structured classification result
	"""
	if config is None:
		config = load_config()
	if local_only:
		cls = ClassificationResult('local', 1.0, 0.0, 'local only', {'local_only': True})
	elif escalate:
		cls = ClassificationResult('cloud', 1.0, 1.0, 'manual override', {'manual': True})
	else:
		cls = classify(prompt, config)
	if cls.decision == 'cloud' and cloud_budget_reason is not None:
		cls = ClassificationResult('local', 1.0, 0.0, cloud_budget_reason, {'budget_blocked': True})
	if cls.decision == 'local':
		local = LocalModel(config.get('local_model', {}))
		if stream:
			gen = local.stream(prompt)
			return 'local', gen, f"Local · {local.name}", cls.to_dict()
		try:
			reply = local.generate(prompt)
		except ProviderError as error:
			reply = error
		return 'local', reply, f"Local · {local.name}", cls.to_dict()
	cloud = CloudModel(config.get('cloud_model', {}), api_keys=config.get('api_keys') or {})
	model_type, resp, label = _cloud_response(cloud, prompt, config.get('system_prompt'), stream, on_usage)
	return model_type, resp, label, cls.to_dict()

def _cloud_response(cloud: CloudModel, prompt: str, system_prompt: str | None, stream: bool, on_usage):
	usage_option = {'on_usage': on_usage} if on_usage is not None else {}
	if stream:
		gen = cloud.stream(prompt, system_prompt=system_prompt, **usage_option)
		return 'cloud', gen, f"Cloud · {cloud.name}"
	try:
		reply = cloud.generate(prompt, system_prompt=system_prompt, **usage_option)
	except ProviderError as error:
		reply = error
	return 'cloud', reply, f"Cloud · {cloud.name}"
