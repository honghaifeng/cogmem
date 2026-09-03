"""
CogMem: Cognitive Memory Network for Long-Term Conversation Memory

A hybrid memory system combining symbolic retrieval, neural retrieval,
and graph-based spreading activation.
"""

from .memory import CogMem
from .cognitive import CognitiveMemory
from .baseline import BaselineMemory
from .llm_client import LLMClient

__version__ = "0.1.0"
__all__ = ["CogMem", "CognitiveMemory", "BaselineMemory", "LLMClient"]
