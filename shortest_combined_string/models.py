"""
Core data models for the Shortest Combined String algorithm.

This module contains all the data classes and enums used throughout the algorithm
implementation, including input preprocessing, DP state management, and result formatting.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Operation(Enum):
    """Operations available in the dynamic programming algorithm."""
    MATCH = "MATCH"
    INSERT_S1 = "INSERT_S1"
    INSERT_S2 = "INSERT_S2"
    SKIP = "SKIP"


class TokenType(Enum):
    """Types of tokens in the combined result."""
    MERGED = "MERGED"
    S1_ONLY = "S1_ONLY"
    S2_ONLY = "S2_ONLY"
    SPACING = "SPACING"


@dataclass
class WordToken:
    """
    Represents a word with its associated spacing information.
    
    Attributes:
        word: The actual word content
        leading_spaces: Number of spaces before the word
        trailing_spaces: Number of spaces after the word
        original_index: Position of this word in the original string
    """
    word: str
    leading_spaces: int
    trailing_spaces: int
    original_index: int
    
    def __post_init__(self):
        """Validate WordToken after initialization."""
        if not isinstance(self.word, str):
            raise TypeError("word must be a string")
        if self.leading_spaces < 0:
            raise ValueError("leading_spaces must be non-negative")
        if self.trailing_spaces < 0:
            raise ValueError("trailing_spaces must be non-negative")
        if self.original_index < 0:
            raise ValueError("original_index must be non-negative")
    
    @property
    def total_length(self) -> int:
        """Calculate total length including spaces."""
        return len(self.word) + self.leading_spaces + self.trailing_spaces


@dataclass
class PreprocessedInput:
    """
    Result of input preprocessing and validation.
    
    Attributes:
        s1: First preprocessed input string
        s2: Second preprocessed input string
        warnings: List of preprocessing warnings
        has_consecutive_spaces: Whether consecutive spaces were found and normalized
    """
    s1: str
    s2: str
    warnings: List[str]
    has_consecutive_spaces: bool
    
    def __post_init__(self):
        """Validate PreprocessedInput after initialization."""
        if not isinstance(self.s1, str):
            raise TypeError("s1 must be a string")
        if not isinstance(self.s2, str):
            raise TypeError("s2 must be a string")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        if not isinstance(self.has_consecutive_spaces, bool):
            raise TypeError("has_consecutive_spaces must be a boolean")


@dataclass
class DPState:
    """
    State in the dynamic programming table.
    
    Attributes:
        length: Minimum length achievable for this subproblem
        s1_word_index: Current position in first string's word sequence
        s2_word_index: Current position in second string's word sequence
        operation: The operation that led to this optimal state
    """
    length: int
    s1_word_index: int
    s2_word_index: int
    operation: Operation
    
    def __post_init__(self):
        """Validate DPState after initialization."""
        if self.length < 0:
            raise ValueError("length must be non-negative")
        if self.s1_word_index < 0:
            raise ValueError("s1_word_index must be non-negative")
        if self.s2_word_index < 0:
            raise ValueError("s2_word_index must be non-negative")
        if not isinstance(self.operation, Operation):
            raise TypeError("operation must be an Operation enum")


@dataclass
class CombinedToken:
    """
    Token in the combined result representing merged or individual content.
    
    Attributes:
        content: The actual string content of this token
        source_s1_words: Indices of words from s1 that contributed to this token
        source_s2_words: Indices of words from s2 that contributed to this token
        type: Type of token (merged, s1 only, s2 only, or spacing)
    """
    content: str
    source_s1_words: List[int]
    source_s2_words: List[int]
    type: TokenType
    
    def __post_init__(self):
        """Validate CombinedToken after initialization."""
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.source_s1_words, list):
            raise TypeError("source_s1_words must be a list")
        if not isinstance(self.source_s2_words, list):
            raise TypeError("source_s2_words must be a list")
        if not isinstance(self.type, TokenType):
            raise TypeError("type must be a TokenType enum")
        
        # Validate that indices are non-negative
        for idx in self.source_s1_words:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError("All source_s1_words indices must be non-negative integers")
        for idx in self.source_s2_words:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError("All source_s2_words indices must be non-negative integers")


@dataclass
class OptimizationMetrics:
    """
    Metrics about the optimization achieved by the algorithm.
    
    Attributes:
        original_s1_length: Length of the first input string
        original_s2_length: Length of the second input string
        combined_length: Length of the optimized combined string
        total_savings: Number of characters saved
        compression_ratio: Ratio of combined length to sum of original lengths
    """
    original_s1_length: int
    original_s2_length: int
    combined_length: int
    total_savings: int
    compression_ratio: float
    
    def __post_init__(self):
        """Validate OptimizationMetrics after initialization."""
        if self.original_s1_length < 0:
            raise ValueError("original_s1_length must be non-negative")
        if self.original_s2_length < 0:
            raise ValueError("original_s2_length must be non-negative")
        if self.combined_length < 0:
            raise ValueError("combined_length must be non-negative")
        if self.compression_ratio < 0:
            raise ValueError("compression_ratio must be non-negative")
        
        # Validate that metrics are consistent
        expected_savings = self.original_s1_length + self.original_s2_length - self.combined_length
        if self.total_savings != expected_savings:
            raise ValueError(f"total_savings ({self.total_savings}) doesn't match calculated savings ({expected_savings})")
        
        total_original = self.original_s1_length + self.original_s2_length
        if total_original > 0:
            expected_ratio = self.combined_length / total_original
            if abs(self.compression_ratio - expected_ratio) > 1e-6:
                raise ValueError(f"compression_ratio ({self.compression_ratio}) doesn't match calculated ratio ({expected_ratio})")


@dataclass
class AlgorithmResult:
    """
    Complete result of the shortest combined string algorithm.
    
    Attributes:
        combined_string: The optimized combined string
        metrics: Optimization metrics and statistics
        is_valid: Whether the result passed validation
        validation_errors: List of validation errors if any
        processing_warnings: List of warnings from preprocessing
    """
    combined_string: str
    metrics: OptimizationMetrics
    is_valid: bool
    validation_errors: List[str]
    processing_warnings: List[str]
    
    def __post_init__(self):
        """Validate AlgorithmResult after initialization."""
        if not isinstance(self.combined_string, str):
            raise TypeError("combined_string must be a string")
        if not isinstance(self.metrics, OptimizationMetrics):
            raise TypeError("metrics must be an OptimizationMetrics instance")
        if not isinstance(self.is_valid, bool):
            raise TypeError("is_valid must be a boolean")
        if not isinstance(self.validation_errors, list):
            raise TypeError("validation_errors must be a list")
        if not isinstance(self.processing_warnings, list):
            raise TypeError("processing_warnings must be a list")
        
        # Validate that combined_string length matches metrics
        if len(self.combined_string) != self.metrics.combined_length:
            raise ValueError(f"combined_string length ({len(self.combined_string)}) doesn't match metrics.combined_length ({self.metrics.combined_length})")