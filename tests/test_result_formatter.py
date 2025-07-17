"""
Unit tests for the ResultFormatter class.

Tests the assembly of final output strings from CombinedToken sequences,
optimization metrics calculation, and proper space trimming functionality.
"""

import pytest
from shortest_combined_string.result_formatter import ResultFormatter
from shortest_combined_string.models import (
    CombinedToken, TokenType, OptimizationMetrics, AlgorithmResult
)


class TestResultFormatter:
    """Test cases for the ResultFormatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ResultFormatter()
    
    def test_init(self):
        """Test ResultFormatter initialization."""
        formatter = ResultFormatter()
        assert formatter is not None
    
    def test_format_result_basic(self):
        """Test basic result formatting with simple tokens."""
        tokens = [
            CombinedToken(
                content="hello",
                source_s1_words=[0],
                source_s2_words=[],
                type=TokenType.S1_ONLY
            ),
            CombinedToken(
                content=" world",
                source_s1_words=[],
                source_s2_words=[0],
                type=TokenType.S2_ONLY
            )
        ]
        
        result = self.formatter.format_result(tokens, "hello", "world")
        
        assert isinstance(result, AlgorithmResult)
        assert result.combined_string == "hello world"
        assert result.is_valid is True
        assert result.validation_errors == []
        assert result.processing_warnings == []
        
        # Check metrics
        assert result.metrics.original_s1_length == 5
        assert result.metrics.original_s2_length == 5
        assert result.metrics.combined_length == 11
        assert result.metrics.total_savings == -1  # Actually longer due to space
        assert result.metrics.compression_ratio == 1.1
    
    def test_format_result_with_merged_tokens(self):
        """Test result formatting with merged tokens."""
        tokens = [
            CombinedToken(
                content="helloworld",
                source_s1_words=[0],
                source_s2_words=[0],
                type=TokenType.MERGED
            )
        ]
        
        result = self.formatter.format_result(tokens, "hello", "world")
        
        assert result.combined_string == "helloworld"
        assert result.metrics.original_s1_length == 5
        assert result.metrics.original_s2_length == 5
        assert result.metrics.combined_length == 10
        assert result.metrics.total_savings == 0
        assert result.metrics.compression_ratio == 1.0
    
    def test_format_result_with_spacing_tokens(self):
        """Test result formatting with spacing tokens."""
        tokens = [
            CombinedToken(
                content="hello",
                source_s1_words=[0],
                source_s2_words=[],
                type=TokenType.S1_ONLY
            ),
            CombinedToken(
                content=" ",
                source_s1_words=[],
                source_s2_words=[],
                type=TokenType.SPACING
            ),
            CombinedToken(
                content="world",
                source_s1_words=[],
                source_s2_words=[0],
                type=TokenType.S2_ONLY
            )
        ]
        
        result = self.formatter.format_result(tokens, "hello", "world")
        
        assert result.combined_string == "hello world"
        assert result.metrics.combined_length == 11
    
    def test_format_result_with_validation_errors(self):
        """Test result formatting with validation errors."""
        tokens = [
            CombinedToken(
                content="invalid",
                source_s1_words=[0],
                source_s2_words=[],
                type=TokenType.S1_ONLY
            )
        ]
        
        validation_errors = ["Subsequence validation failed"]
        result = self.formatter.format_result(
            tokens, "hello", "world", 
            validation_errors=validation_errors
        )
        
        assert result.is_valid is False
        assert result.validation_errors == ["Subsequence validation failed"]
    
    def test_format_result_with_processing_warnings(self):
        """Test result formatting with processing warnings."""
        tokens = [
            CombinedToken(
                content="hello world",
                source_s1_words=[0],
                source_s2_words=[0],
                type=TokenType.MERGED
            )
        ]
        
        processing_warnings = ["Consecutive spaces normalized"]
        result = self.formatter.format_result(
            tokens, "hello", "world",
            processing_warnings=processing_warnings
        )
        
        assert result.processing_warnings == ["Consecutive spaces normalized"]
    
    def test_format_result_empty_tokens(self):
        """Test result formatting with empty token list."""
        result = self.formatter.format_result([], "hello", "world")
        
        assert result.combined_string == ""
        assert result.metrics.combined_length == 0
        assert result.metrics.total_savings == 10
        assert result.metrics.compression_ratio == 0.0
    
    def test_format_result_empty_inputs(self):
        """Test result formatting with empty input strings."""
        tokens = [
            CombinedToken(
                content="test",
                source_s1_words=[],
                source_s2_words=[],
                type=TokenType.SPACING
            )
        ]
        
        result = self.formatter.format_result(tokens, "", "")
        
        assert result.combined_string == "test"
        assert result.metrics.original_s1_length == 0
        assert result.metrics.original_s2_length == 0
        assert result.metrics.combined_length == 4
        assert result.metrics.total_savings == -4
        assert result.metrics.compression_ratio == float('inf')
    
    def test_format_result_both_empty_inputs_empty_output(self):
        """Test result formatting with both inputs and output empty."""
        result = self.formatter.format_result([], "", "")
        
        assert result.combined_string == ""
        assert result.metrics.original_s1_length == 0
        assert result.metrics.original_s2_length == 0
        assert result.metrics.combined_length == 0
        assert result.metrics.total_savings == 0
        assert result.metrics.compression_ratio == 0.0
    
    def test_format_result_type_validation(self):
        """Test type validation for format_result parameters."""
        tokens = []
        
        # Test invalid tokens type
        with pytest.raises(TypeError, match="tokens must be a list"):
            self.formatter.format_result("not a list", "hello", "world")
        
        # Test invalid original_s1 type
        with pytest.raises(TypeError, match="original_s1 must be a string"):
            self.formatter.format_result(tokens, 123, "world")
        
        # Test invalid original_s2 type
        with pytest.raises(TypeError, match="original_s2 must be a string"):
            self.formatter.format_result(tokens, "hello", 123)
        
        # Test invalid validation_errors type
        with pytest.raises(TypeError, match="validation_errors must be a list"):
            self.formatter.format_result(tokens, "hello", "world", validation_errors="not a list")
        
        # Test invalid processing_warnings type
        with pytest.raises(TypeError, match="processing_warnings must be a list"):
            self.formatter.format_result(tokens, "hello", "world", processing_warnings="not a list")
    
    def test_format_result_invalid_token_type(self):
        """Test validation of token types in the list."""
        invalid_tokens = [
            "not a token",
            CombinedToken("valid", [], [], TokenType.S1_ONLY)
        ]
        
        with pytest.raises(TypeError, match="Token at index 0 must be a CombinedToken object"):
            self.formatter.format_result(invalid_tokens, "hello", "world")
    
    def test_assemble_output_string_basic(self):
        """Test basic output string assembly."""
        tokens = [
            CombinedToken("hello", [0], [], TokenType.S1_ONLY),
            CombinedToken(" ", [], [], TokenType.SPACING),
            CombinedToken("world", [], [0], TokenType.S2_ONLY)
        ]
        
        result = self.formatter._assemble_output_string(tokens)
        assert result == "hello world"
    
    def test_assemble_output_string_with_leading_trailing_spaces(self):
        """Test output string assembly with leading/trailing space trimming."""
        tokens = [
            CombinedToken("  hello world  ", [0], [0], TokenType.MERGED)
        ]
        
        result = self.formatter._assemble_output_string(tokens)
        assert result == "hello world"  # Spaces should be trimmed
    
    def test_assemble_output_string_empty_tokens(self):
        """Test output string assembly with empty token list."""
        result = self.formatter._assemble_output_string([])
        assert result == ""
    
    def test_assemble_output_string_only_spaces(self):
        """Test output string assembly with only space tokens."""
        tokens = [
            CombinedToken("   ", [], [], TokenType.SPACING)
        ]
        
        result = self.formatter._assemble_output_string(tokens)
        # With our updated behavior, whitespace-only strings are preserved
        assert result == "   "  # Spaces are preserved for whitespace-only strings
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        metrics = self.formatter._calculate_metrics("hello", "world", "helloworld")
        
        assert isinstance(metrics, OptimizationMetrics)
        assert metrics.original_s1_length == 5
        assert metrics.original_s2_length == 5
        assert metrics.combined_length == 10
        assert metrics.total_savings == 0
        assert metrics.compression_ratio == 1.0
    
    def test_calculate_metrics_with_savings(self):
        """Test metrics calculation with actual savings."""
        # Overlapping case: "hello" + "world" -> "helloworld" (10 chars vs 11 with space)
        metrics = self.formatter._calculate_metrics("hello ", " world", "helloworld")
        
        assert metrics.original_s1_length == 6
        assert metrics.original_s2_length == 6
        assert metrics.combined_length == 10
        assert metrics.total_savings == 2
        assert abs(metrics.compression_ratio - (10/12)) < 1e-6
    
    def test_calculate_metrics_no_savings(self):
        """Test metrics calculation with no savings (concatenation)."""
        metrics = self.formatter._calculate_metrics("hello", "world", "hello world")
        
        assert metrics.original_s1_length == 5
        assert metrics.original_s2_length == 5
        assert metrics.combined_length == 11
        assert metrics.total_savings == -1  # Actually longer due to space
        assert abs(metrics.compression_ratio - 1.1) < 1e-6
    
    def test_calculate_metrics_empty_inputs(self):
        """Test metrics calculation with empty inputs."""
        metrics = self.formatter._calculate_metrics("", "", "")
        
        assert metrics.original_s1_length == 0
        assert metrics.original_s2_length == 0
        assert metrics.combined_length == 0
        assert metrics.total_savings == 0
        assert metrics.compression_ratio == 0.0
    
    def test_calculate_metrics_empty_inputs_non_empty_output(self):
        """Test metrics calculation with empty inputs but non-empty output."""
        metrics = self.formatter._calculate_metrics("", "", "result")
        
        assert metrics.original_s1_length == 0
        assert metrics.original_s2_length == 0
        assert metrics.combined_length == 6
        assert metrics.total_savings == -6
        assert metrics.compression_ratio == float('inf')
    
    def test_calculate_metrics_one_empty_input(self):
        """Test metrics calculation with one empty input."""
        metrics = self.formatter._calculate_metrics("", "world", "world")
        
        assert metrics.original_s1_length == 0
        assert metrics.original_s2_length == 5
        assert metrics.combined_length == 5
        assert metrics.total_savings == 0
        assert metrics.compression_ratio == 1.0
    
    def test_format_metrics_summary_basic(self):
        """Test basic metrics summary formatting."""
        metrics = OptimizationMetrics(
            original_s1_length=5,
            original_s2_length=5,
            combined_length=10,
            total_savings=0,
            compression_ratio=1.0
        )
        
        summary = self.formatter.format_metrics_summary(metrics)
        
        expected_lines = [
            "Original lengths: s1=5, s2=5",
            "Combined length: 10",
            "Total savings: 0 characters",
            "Compression ratio: 1.000"
        ]
        assert summary == "\n".join(expected_lines)
    
    def test_format_metrics_summary_with_savings(self):
        """Test metrics summary formatting with savings."""
        metrics = OptimizationMetrics(
            original_s1_length=10,
            original_s2_length=10,
            combined_length=15,
            total_savings=5,
            compression_ratio=0.75
        )
        
        summary = self.formatter.format_metrics_summary(metrics)
        
        assert "Total savings: 5 characters" in summary
        assert "Compression ratio: 0.750" in summary
    
    def test_format_metrics_summary_type_validation(self):
        """Test type validation for format_metrics_summary."""
        with pytest.raises(TypeError, match="metrics must be an OptimizationMetrics object"):
            self.formatter.format_metrics_summary("not metrics")
    
    def test_integration_realistic_case(self):
        """Test integration with a realistic case similar to the main test case."""
        # Simulate tokens for "this is a red vase" + "his son freddy love vase"
        tokens = [
            CombinedToken("this", [0], [], TokenType.S1_ONLY),
            CombinedToken(" ", [], [], TokenType.SPACING),
            CombinedToken("his", [0], [0], TokenType.MERGED),  # "is" from s1, "his" from s2
            CombinedToken(" son freddy love", [], [1, 2, 3], TokenType.S2_ONLY),
            CombinedToken(" a red vase", [1, 2, 3], [4], TokenType.MERGED)
        ]
        
        original_s1 = "this is a red vase"
        original_s2 = "his son freddy love vase"
        
        result = self.formatter.format_result(tokens, original_s1, original_s2)
        
        # Verify the result structure
        assert isinstance(result, AlgorithmResult)
        assert len(result.combined_string) > 0
        assert result.metrics.original_s1_length == len(original_s1)
        assert result.metrics.original_s2_length == len(original_s2)
        assert result.metrics.combined_length == len(result.combined_string)
        
        # Verify metrics consistency
        expected_savings = len(original_s1) + len(original_s2) - len(result.combined_string)
        assert result.metrics.total_savings == expected_savings
        
        total_original = len(original_s1) + len(original_s2)
        expected_ratio = len(result.combined_string) / total_original
        assert abs(result.metrics.compression_ratio - expected_ratio) < 1e-6
    
    def test_complex_spacing_handling(self):
        """Test complex spacing scenarios in output assembly."""
        tokens = [
            CombinedToken("  hello ", [0], [], TokenType.S1_ONLY),
            CombinedToken(" world  ", [], [0], TokenType.S2_ONLY)
        ]
        
        result = self.formatter._assemble_output_string(tokens)
        assert result == "hello  world"  # Leading/trailing spaces trimmed, internal preserved
    
    def test_multiple_space_tokens(self):
        """Test handling of multiple spacing tokens."""
        tokens = [
            CombinedToken("hello", [0], [], TokenType.S1_ONLY),
            CombinedToken("   ", [], [], TokenType.SPACING),
            CombinedToken("world", [], [0], TokenType.S2_ONLY),
            CombinedToken("  ", [], [], TokenType.SPACING)
        ]
        
        result = self.formatter._assemble_output_string(tokens)
        assert result == "hello   world"  # Trailing spaces trimmed, internal preserved