"""
Configuration loader for The Receptionist.
"""
import yaml
import os, re

CONFIG_PATH = os.getenv("RECEPTIONIST_CONFIG", "config.yaml")

def _expand_env(value: str):
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")
    def repl(m):
        return os.getenv(m.group(1), '')
    return pattern.sub(repl, value)

def _deep_expand(obj):
    if isinstance(obj, dict):
        return {k: _deep_expand(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [_deep_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj

def load_config(path=CONFIG_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return _deep_expand(data)
