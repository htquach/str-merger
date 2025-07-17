"""
Integration tests for the CLI functionality of the Shortest Combined String algorithm.
"""

import sys
import pytest
from unittest.mock import patch
from io import StringIO

from shortest_combined_string.cli import main, parse_args, format_result
from shortest_combined_string.shortest_combined_string import ShortestCombinedString
from shortest_combined_string.models import AlgorithmResult, OptimizationMetrics


class TestCLI:
    """Test suite for the CLI functionality."""
    
    def test_parse_args(self):
        """Test argument parsing."""
        # Test basic argument parsing
        args = parse_args(["string1", "string2"])
        assert args.string1 == "string1"
        assert args.string2 == "string2"
        assert not args.no_color
        assert not args.quote
        
        # Test with optional flags
        args = parse_args(["string1", "string2", "--no-color", "--quote"])
        assert args.string1 == "string1"
        assert args.string2 == "string2"
        assert args.no_color
        assert args.quote
    
    def test_format_result(self):
        """Test result formatting."""
        # Create a mock result
        metrics = OptimizationMetrics(
            original_s1_length=10,
            original_s2_length=15,
            combined_length=20,
            total_savings=5,
            compression_ratio=0.8
        )
        
        result = AlgorithmResult(
            combined_string="test combined string",
            metrics=metrics,
            is_valid=True,
            validation_errors=[],
            processing_warnings=["Test warning"]
        )
        
        # Test formatting without color
        formatted = format_result(result, use_color=False, quote=False)
        assert "test combined string" in formatted
        assert "VALID" in formatted
        assert "Original lengths: s1=10, s2=15" in formatted
        assert "Combined length: 20" in formatted
        assert "Total savings: 5 characters" in formatted
        assert "Compression ratio: 0.800" in formatted
        assert "Test warning" in formatted
        
        # Test with quotes
        formatted = format_result(result, use_color=False, quote=True)
        assert '"test combined string"' in formatted
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_main_success(self, mock_stdout):
        """Test main function with successful execution."""
        # Test with the primary test case
        exit_code = main(["this is a red vase", "his son freddy love vase", "--no-color"])
        
        # Check output contains expected elements
        output = mock_stdout.getvalue()
        assert "Shortest Combined String Result" in output
        assert "VALID" in output
        assert "Optimization Metrics:" in output
        
        # Check exit code
        assert exit_code == 0
    
    @patch('sys.stdout', new_callable=StringIO)
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_with_invalid_input(self, mock_stderr, mock_stdout):
        """Test main function with invalid input."""
        # Test with invalid input (None is not allowed)
        with pytest.raises(SystemExit):
            # This should raise SystemExit because argparse will exit
            main([])
        
        # Check error message
        error_output = mock_stderr.getvalue()
        assert "error" in error_output.lower()
    
    def test_integration_primary_case(self):
        """Integration test with the primary test case."""
        # Create algorithm instance directly
        algorithm = ShortestCombinedString()
        
        # Process the primary test case
        s1 = "this is a red vase"
        s2 = "his son freddy love vase"
        result = algorithm.combine(s1, s2)
        
        # Verify the result
        assert result.is_valid
        assert len(result.combined_string) <= 26  # Per requirement 2.1
        assert result.metrics.compression_ratio < 1.0  # Should achieve some compression
    
    def test_integration_edge_cases(self):
        """Integration test with edge cases."""
        test_cases = [
            # Empty strings
            ("", "", ""),
            ("hello", "", "hello"),
            ("", "world", "world"),
            
            # Identical strings
            ("test", "test", "test"),
            
            # One string contains the other
            ("hello world", "hello", "hello world"),
            ("hi", "hi there", "hi there"),
            
            # No common characters
            ("abc", "def", "abc def"),
            # Single character inputs
            ("a", "b", "ab")
        ]
        
        for s1, s2, expected_output in test_cases:
            # Run through CLI main function with mocked args
            with patch('sys.argv', ['cli.py', s1, s2, '--no-color']):
                with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                    exit_code = main([s1, s2, '--no-color'])
                    
                    # Check exit code and output
                    assert exit_code == 0
                    output = mock_stdout.getvalue()
                    assert "VALID" in output
                    
                    # For empty string cases, we don't check the exact output
                    if s1 and s2:
                        # The output should contain the expected combined string
                        assert expected_output in output