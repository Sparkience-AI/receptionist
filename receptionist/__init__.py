# Receptionist core package
"""
The Receptionist: Smart, cost-aware AI query routing framework.
"""
from .config import load_config
from .classifier import classify_prompt, classify, ClassificationResult
from .router import route_query
from .metrics import MetricsTracker
