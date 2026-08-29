"""
Forensic Activity Reconstruction Evaluation Harness
Designed around the principles from 'AI Agents in Depth: Design Principles and Engineering Practice'
Formula: Agent = LLM + Context + Tools
"""

from .models import get_model_runner
from .evaluators import ForensicEvaluator

__all__ = ["get_model_runner", "ForensicEvaluator"]

