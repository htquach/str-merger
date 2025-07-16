"""
Shortest Combined String Algorithm Package

This package implements an algorithm to find the shortest possible combined string
that contains two input sentences as subsequences while preserving word integrity
and maximizing character reuse.
"""

from .models import (
    WordToken,
    PreprocessedInput,
    DPState,
    CombinedToken,
    OptimizationMetrics,
    AlgorithmResult,
    Operation,
    TokenType
)
from .input_processor import InputProcessor
from .word_tokenizer import WordTokenizer
from .subsequence_verifier import SubsequenceVerifier, VerificationResult, SubsequenceMatch
from .dp_solver import DPSolver, DPResult

__version__ = "1.0.0"
__all__ = [
    "WordToken",
    "PreprocessedInput", 
    "DPState",
    "CombinedToken",
    "OptimizationMetrics",
    "AlgorithmResult",
    "Operation",
    "TokenType",
    "InputProcessor",
    "WordTokenizer",
    "SubsequenceVerifier",
    "VerificationResult",
    "SubsequenceMatch",
    "DPSolver",
    "DPResult"
]