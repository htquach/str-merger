"""
Comprehensive edge case tests for the ShortestCombinedString algorithm.

This module contains tests specifically focused on edge cases:
- Empty strings
- Identical strings
- One string containing the other
- Strings with no common characters
- Single character inputs
- Whitespace-only strings
- Boundary cases
"""

import pytest
from shortest_combined_string.shortest_combined_string import ShortestCombinedString
from shortest_combined_string.subsequence_verifier import SubsequenceVerifier


class TestEdgeCases:
    """Comprehensive edge case tests for the ShortestCombinedString algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.algorithm = ShortestCombinedString()
        self.verifier = SubsequenceVerifier()
    
    def test_empty_strings(self):
        """Test all empty string scenarios."""
        # Both empty
        result = self.algorithm.combine("", "")
        assert result.is_valid
        assert result.combined_string == ""
        assert result.metrics.total_savings == 0
        assert result.metrics.compression_ratio == 0.0  # Special case for empty strings
        
        # First empty, second non-empty
        result = self.algorithm.combine("", "hello")
        assert result.is_valid
        assert result.combined_string == "hello"
        assert result.metrics.total_savings == 0
        assert result.metrics.compression_ratio == 1.0
        
        # First non-empty, second empty
        result = self.algorithm.combine("world", "")
        assert result.is_valid
        assert result.combined_string == "world"
        assert result.metrics.total_savings == 0
        assert result.metrics.compression_ratio == 1.0
    
    def test_identical_strings(self):
        """Test identical string scenarios."""
        # Simple identical strings
        s = "hello"
        result = self.algorithm.combine(s, s)
        assert result.is_valid
        assert result.combined_string == s
        assert result.metrics.total_savings == len(s)
        assert result.metrics.compression_ratio == 0.5
        
        # Longer identical strings with spaces
        s = "this is a test string"
        result = self.algorithm.combine(s, s)
        assert result.is_valid
        assert result.combined_string == s
        assert result.metrics.total_savings == len(s)
        assert result.metrics.compression_ratio == 0.5
        
        # Identical strings with special characters
        s = "!@#$%^&*()"
        result = self.algorithm.combine(s, s)
        assert result.is_valid
        assert result.combined_string == s
        assert result.metrics.total_savings == len(s)
        assert result.metrics.compression_ratio == 0.5
    
    def test_one_string_contains_other(self):
        """Test scenarios where one string contains the other."""
        # Simple containment
        s1 = "hello world"
        s2 = "hello"
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        assert result.combined_string == s1
        
        # Reverse order
        result = self.algorithm.combine(s2, s1)
        assert result.is_valid
        assert result.combined_string == s1
        
        # Containment with spaces
        s1 = "this is a test"
        s2 = "is a"
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        assert result.combined_string == s1
        
        # Containment at the beginning
        s1 = "start of the string"
        s2 = "start"
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        assert result.combined_string == s1
        
        # Containment at the end
        s1 = "end of the string"
        s2 = "string"
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        assert result.combined_string == s1
    
    def test_no_common_characters(self):
        """Test strings with no common characters."""
        # Simple case
        s1 = "abcde"
        s2 = "fghij"
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        
        # Verify both strings are subsequences
        verification = self.verifier.verify(s1, s2, result.combined_string)
        assert verification.is_valid
        
        # Should be close to the sum of lengths (plus maybe a space)
        assert len(result.combined_string) >= len(s1) + len(s2)
        assert len(result.combined_string) <= len(s1) + len(s2) + 1
        
        # Different character sets
        s1 = "12345"
        s2 = "abcde"
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        
        # Verify both strings are subsequences
        verification = self.verifier.verify(s1, s2, result.combined_string)
        assert verification.is_valid
        
        # With spaces
        s1 = "abc def"
        s2 = "123 456"
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        
        # Verify both strings are subsequences
        verification = self.verifier.verify(s1, s2, result.combined_string)
        assert verification.is_valid
    
    def test_single_character_inputs(self):
        """Test single character input scenarios."""
        # Two different characters
        result = self.algorithm.combine("a", "b")
        assert result.is_valid
        assert len(result.combined_string) <= 2  # Should be "ab" or "ba"
        
        # Same character
        result = self.algorithm.combine("a", "a")
        assert result.is_valid
        assert result.combined_string == "a"
        
        # Special characters
        result = self.algorithm.combine("!", "?")
        assert result.is_valid
        assert len(result.combined_string) <= 2
        
        # Space and character
        result = self.algorithm.combine(" ", "x")
        assert result.is_valid
        assert len(result.combined_string) <= 2
    
    def test_whitespace_only_strings(self):
        """Test whitespace-only string scenarios."""
        # Single spaces
        result = self.algorithm.combine(" ", " ")
        assert result.is_valid
        # The result should be a single space or empty string (both are valid)
        assert result.combined_string in [" ", ""]
        
        # Different number of spaces - note that consecutive spaces are normalized
        result = self.algorithm.combine(" ", "  ")
        assert result.is_valid
        # The result should be a single space since consecutive spaces are normalized
        assert result.combined_string == " "
        # Verify that a warning was generated about space normalization
        assert any("Consecutive spaces detected" in warning for warning in result.processing_warnings)
        
        # Multiple spaces - note that consecutive spaces are normalized
        result = self.algorithm.combine("   ", "  ")
        assert result.is_valid
        # The result should be a single space since consecutive spaces are normalized
        assert result.combined_string == " "
        # Verify that warnings were generated about space normalization
        assert len([w for w in result.processing_warnings if "Consecutive spaces detected" in w]) == 2
    
    def test_boundary_cases(self):
        """Test boundary cases."""
        # Very long string with very short string
        long_str = "a" * 1000
        short_str = "b"
        
        result = self.algorithm.combine(long_str, short_str)
        assert result.is_valid
        
        # Verify both strings are subsequences
        verification = self.verifier.verify(long_str, short_str, result.combined_string)
        assert verification.is_valid
        
        # String with only repeated characters
        s1 = "aaaaa"
        s2 = "bbbbb"
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        
        # Verify both strings are subsequences
        verification = self.verifier.verify(s1, s2, result.combined_string)
        assert verification.is_valid
        
        # String with mixed spaces and characters
        s1 = " a b c "
        s2 = "d e f "
        
        result = self.algorithm.combine(s1, s2)
        assert result.is_valid
        
        # Verify both strings are subsequences
        verification = self.verifier.verify(s1, s2, result.combined_string)
        assert verification.is_valid
    
    def test_empty_string_validation(self):
        """Test that empty strings are always valid subsequences of any output."""
        # Empty string should be a valid subsequence of any string
        verification = self.verifier.verify("", "", "anything")
        assert verification.is_valid
        assert verification.s1_match.is_valid
        assert verification.s2_match.is_valid
        
        # Empty string should be a valid subsequence of empty string
        verification = self.verifier.verify("", "", "")
        assert verification.is_valid
        assert verification.s1_match.is_valid
        assert verification.s2_match.is_valid
        
        # Test with algorithm - empty string with non-empty string
        result = self.algorithm.combine("", "hello")
        assert result.is_valid
        assert result.combined_string == "hello"
        
        # Test with algorithm - non-empty string with empty string
        result = self.algorithm.combine("world", "")
        assert result.is_valid
        assert result.combined_string == "world"