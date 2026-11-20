"""Routing logic with structured classification, streaming, and explanation."""
from __future__ import annotations
from typing import Dict, Any, Tuple
from .classifier import classify, ClassificationResult
from .models.local import LocalModel
from .models.cloud import CloudModel
from .config import load_config

def _init_models(config: Dict[str,Any]):
	api_keys = (config.get('api_keys') or {})
	local = LocalModel(config.get('local_model', {}))
	cloud = CloudModel(config.get('cloud_model', {}), api_keys=api_keys)
	return local, cloud

def route_query(prompt: str, config=None, escalate: bool=False, stream: bool=False):
	"""Route a prompt.
	Returns (model_type, response_or_generator, model_label, classification_dict)
	model_type: 'local' | 'cloud'
	response_or_generator: str or generator of str tokens
	classification_dict: structured classification result
	"""
	if config is None:
		config = load_config()
	system_prompt = config.get('system_prompt')
	local, cloud = _init_models(config)

	if escalate:
		model_type, resp, label = _cloud_response(cloud, prompt, system_prompt, stream)
		fake_cls = ClassificationResult('cloud', 1.0, 1.0, 'manual override', {'manual': True}).to_dict()
		return model_type, resp, label, fake_cls

	cls = classify(prompt, config)
	if cls.decision == 'local':
		if stream:
			gen = local.stream(prompt)
			return 'local', gen, f"Local · {local.name}", cls.to_dict()
		return 'local', local.generate(prompt), f"Local · {local.name}", cls.to_dict()
	model_type, resp, label = _cloud_response(cloud, prompt, system_prompt, stream)
	return model_type, resp, label, cls.to_dict()

def _cloud_response(cloud: CloudModel, prompt: str, system_prompt: str | None, stream: bool):
	if stream:
		gen = cloud.stream(prompt, system_prompt=system_prompt)
		return 'cloud', gen, f"Cloud · {cloud.name}"
	return 'cloud', cloud.generate(prompt, system_prompt=system_prompt), f"Cloud · {cloud.name}"
