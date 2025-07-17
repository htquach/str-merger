"""
Custom exceptions for the Shortest Combined String algorithm.

This module defines a hierarchy of custom exceptions used throughout the algorithm
to provide more specific error information and enable better error handling.
"""


class ShortestCombinedStringError(Exception):
    """Base exception class for all errors in the Shortest Combined String algorithm."""
    pass


class InputValidationError(ShortestCombinedStringError):
    """Exception raised for errors in the input validation."""
    pass


class TokenizationError(ShortestCombinedStringError):
    """Exception raised for errors during tokenization."""
    pass


class AlgorithmError(ShortestCombinedStringError):
    """Exception raised for errors in the core algorithm."""
    pass


class DPSolverError(AlgorithmError):
    """Exception raised for errors in the DP solver."""
    pass


class PathReconstructionError(AlgorithmError):
    """Exception raised for errors during path reconstruction."""
    pass


class FormattingError(ShortestCombinedStringError):
    """Exception raised for errors during result formatting."""
    pass


class VerificationError(ShortestCombinedStringError):
    """Exception raised for errors during subsequence verification."""
    pass