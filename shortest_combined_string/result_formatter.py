"""
Result formatting for the Shortest Combined String algorithm.

This module implements the ResultFormatter class that assembles the final output string
from CombinedToken sequences and calculates optimization metrics including savings
and compression ratios.
"""

from typing import List
from .models import CombinedToken, OptimizationMetrics, AlgorithmResult
from .exceptions import FormattingError


class ResultFormatter:
    """
    Formats the final result from a sequence of CombinedToken objects.
    
    This class assembles the final output string from the token sequence generated
    by the PathReconstructor, calculates optimization metrics, and handles proper
    spacing including leading/trailing space trimming.
    """
    
    def __init__(self):
        """Initialize the ResultFormatter."""
        pass
    
    def format_result(self, tokens: List[CombinedToken], original_s1: str, original_s2: str, 
                     validation_errors: List[str] = None, processing_warnings: List[str] = None) -> AlgorithmResult:
        """
        Format the complete algorithm result from token sequence.
        
        Assembles the final output string from CombinedToken sequence, calculates
        optimization metrics, and creates the complete AlgorithmResult.
        
        Args:
            tokens: Sequence of CombinedToken objects representing the solution
            original_s1: Original first input string
            original_s2: Original second input string
            validation_errors: List of validation errors (defaults to empty list)
            processing_warnings: List of processing warnings (defaults to empty list)
            
        Returns:
            Complete AlgorithmResult with formatted output and metrics
            
        Raises:
            TypeError: If inputs are not of the expected types
            ValueError: If token sequence is invalid
        """
        if not isinstance(tokens, list):
            raise FormattingError("tokens must be a list")
        if not isinstance(original_s1, str):
            raise FormattingError("original_s1 must be a string")
        if not isinstance(original_s2, str):
            raise FormattingError("original_s2 must be a string")
        
        # Validate token types
        for i, token in enumerate(tokens):
            if not isinstance(token, CombinedToken):
                raise FormattingError(f"Token at index {i} must be a CombinedToken object")
        
        # Set defaults for optional parameters
        if validation_errors is None:
            validation_errors = []
        if processing_warnings is None:
            processing_warnings = []
        
        if not isinstance(validation_errors, list):
            raise FormattingError("validation_errors must be a list")
        if not isinstance(processing_warnings, list):
            raise FormattingError("processing_warnings must be a list")
        
        # Assemble the output string from tokens
        combined_string = self._assemble_output_string(tokens)
        
        # Calculate optimization metrics
        metrics = self._calculate_metrics(original_s1, original_s2, combined_string)
        
        # Determine if result is valid (no validation errors)
        is_valid = len(validation_errors) == 0
        
        return AlgorithmResult(
            combined_string=combined_string,
            metrics=metrics,
            is_valid=is_valid,
            validation_errors=validation_errors,
            processing_warnings=processing_warnings
        )
    
    def _assemble_output_string(self, tokens: List[CombinedToken]) -> str:
        """
        Assemble the final output string from CombinedToken sequence.
        
        Concatenates all token content and applies leading/trailing space trimming
        to the final result as specified in the requirements, except for whitespace-only
        strings which are preserved as-is.
        
        Args:
            tokens: Sequence of CombinedToken objects
            
        Returns:
            The assembled and trimmed output string
        """
        if not tokens:
            return ""
        
        # Concatenate all token content
        result_parts = []
        for token in tokens:
            result_parts.append(token.content)
        
        # Join all parts
        combined = ''.join(result_parts)
        
        # Special case: if the combined string contains only whitespace, preserve it
        if combined and combined.isspace():
            return combined
            
        # Trim leading and trailing spaces as required for normal strings
        trimmed = combined.strip()
        
        return trimmed
    
    def _calculate_metrics(self, original_s1: str, original_s2: str, combined_string: str) -> OptimizationMetrics:
        """
        Calculate optimization metrics for the algorithm result.
        
        Computes savings, compression ratio, and other metrics that quantify
        the optimization achieved by the algorithm.
        
        Args:
            original_s1: Original first input string
            original_s2: Original second input string
            combined_string: The optimized combined output string
            
        Returns:
            OptimizationMetrics object with calculated values
        """
        original_s1_length = len(original_s1)
        original_s2_length = len(original_s2)
        combined_length = len(combined_string)
        
        # Calculate total savings
        total_original_length = original_s1_length + original_s2_length
        total_savings = total_original_length - combined_length
        
        # Calculate compression ratio
        if total_original_length > 0:
            compression_ratio = combined_length / total_original_length
        else:
            # Handle edge case where both inputs are empty
            compression_ratio = 0.0 if combined_length == 0 else float('inf')
        
        return OptimizationMetrics(
            original_s1_length=original_s1_length,
            original_s2_length=original_s2_length,
            combined_length=combined_length,
            total_savings=total_savings,
            compression_ratio=compression_ratio
        )
    
    def format_metrics_summary(self, metrics: OptimizationMetrics) -> str:
        """
        Format a human-readable summary of optimization metrics.
        
        Creates a formatted string that displays the key metrics in a readable format
        for reporting and debugging purposes.
        
        Args:
            metrics: OptimizationMetrics object to format
            
        Returns:
            Formatted string summary of metrics
            
        Raises:
            FormattingError: If metrics is not an OptimizationMetrics object
        """
        if not isinstance(metrics, OptimizationMetrics):
            raise FormattingError("metrics must be an OptimizationMetrics instance")
        
        summary_lines = [
            f"Original lengths: s1={metrics.original_s1_length}, s2={metrics.original_s2_length}",
            f"Combined length: {metrics.combined_length}",
            f"Total savings: {metrics.total_savings} characters",
            f"Compression ratio: {metrics.compression_ratio:.3f}"
        ]
        
        return "\n".join(summary_lines)