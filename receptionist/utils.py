"""
Utility functions for The Receptionist.
"""
def count_tokens(text: str) -> int:
	# Simple whitespace-based token count
	return len(text.split())
