"""
Unit tests for SubsequenceVerifier class.

Tests cover various valid and invalid subsequence scenarios including edge cases,
error reporting, and detailed match information generation.
"""

import pytest
from shortest_combined_string.subsequence_verifier import (
    SubsequenceVerifier, 
    VerificationResult, 
    SubsequenceMatch
)


class TestSubsequenceVerifier:
    """Test cases for SubsequenceVerifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = SubsequenceVerifier()
    
    def test_valid_subsequences_basic(self):
        """Test basic valid subsequence verification."""
        s1 = "abc"
        s2 = "def"
        output = "adbecf"
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        assert result.s1_match.output_positions == [0, 2, 4]
        assert result.s2_match.output_positions == [1, 3, 5]
        assert len(result.validation_errors) == 0
    
    def test_valid_subsequences_with_spaces(self):
        """Test valid subsequence verification with spaces."""
        s1 = "this is"
        s2 = "his son"
        output = "this is son"  # Fixed: now contains both subsequences
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        assert len(result.validation_errors) == 0
    
    def test_valid_subsequences_overlapping_chars(self):
        """Test valid subsequences with overlapping characters."""
        s1 = "red vase"
        s2 = "freddy"
        output = "freddy vase"
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        # s1 "red vase" should be found as subsequence in "freddy vase"
        # r-e-d from "freddy", space and "vase" directly
        expected_s1_positions = [1, 2, 3, 6, 7, 8, 9, 10]  # "red vase"
        assert result.s1_match.output_positions == expected_s1_positions
    
    def test_invalid_subsequence_missing_character(self):
        """Test invalid subsequence when character is missing."""
        s1 = "abc"
        s2 = "def"
        output = "abef"  # Missing 'c' and 'd'
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_invalid
        assert result.s1_match.is_invalid
        assert result.s2_match.is_invalid
        assert 'c' in result.s1_match.missing_chars
        assert 'd' in result.s2_match.missing_chars
        assert len(result.validation_errors) == 2
    
    def test_invalid_subsequence_wrong_order(self):
        """Test invalid subsequence when characters are in wrong order."""
        s1 = "abc"
        s2 = "def"
        output = "cbadef"  # s1 characters in reverse order
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_invalid
        assert result.s1_match.is_invalid  # 'a' after 'c' violates subsequence order
        assert result.s2_match.is_valid  # s2 is still valid
        assert len(result.validation_errors) == 1
    
    def test_empty_input_strings(self):
        """Test verification with empty input strings."""
        result = self.verifier.verify("", "", "anything")
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        assert result.s1_match.output_positions == []
        assert result.s2_match.output_positions == []
        assert len(result.validation_errors) == 0
    
    def test_empty_output_string(self):
        """Test verification with empty output string."""
        result = self.verifier.verify("a", "b", "")
        
        assert result.is_invalid
        assert result.s1_match.is_invalid
        assert result.s2_match.is_invalid
        assert 'a' in result.s1_match.missing_chars
        assert 'b' in result.s2_match.missing_chars
        assert len(result.validation_errors) == 2
    
    def test_one_empty_input(self):
        """Test verification with one empty input string."""
        result = self.verifier.verify("", "abc", "abc")
        
        assert result.is_valid
        assert result.s1_match.is_valid  # Empty string is always valid subsequence
        assert result.s2_match.is_valid
        assert result.s1_match.output_positions == []
        assert result.s2_match.output_positions == [0, 1, 2]
    
    def test_identical_strings(self):
        """Test verification when both inputs are identical."""
        s1 = s2 = "hello"
        output = "hello"
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        assert result.s1_match.output_positions == [0, 1, 2, 3, 4]
        assert result.s2_match.output_positions == [0, 1, 2, 3, 4]
    
    def test_one_string_contains_other(self):
        """Test verification when one input contains the other."""
        s1 = "cat"
        s2 = "catch"
        output = "catch"
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        assert result.s1_match.output_positions == [0, 1, 2]  # "cat"
        assert result.s2_match.output_positions == [0, 1, 2, 3, 4]  # "catch"
    
    def test_repeated_characters(self):
        """Test verification with repeated characters."""
        s1 = "aaa"
        s2 = "bbb"
        output = "ababab"

        # FIXME: That test is not valid.  "aaa" is a word, so cannot have space or interleaving characters between them.  
        # only aaabbb or bbbaaa is valid for the above input string s1 and s2.
        
        result = self.verifier.verify(s1, s2, output)
        
        # Actually "ababab" does contain "aaa" and "bbb" as subsequences
        # "aaa" can be found at positions [0, 2, 4] and "bbb" at [1, 3, 5]
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
    
    def test_repeated_characters_sufficient(self):
        """Test verification with sufficient repeated characters."""
        s1 = "aa"
        s2 = "bb"
        output = "abab"
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        assert result.s1_match.output_positions == [0, 2]  # Two 'a's
        assert result.s2_match.output_positions == [1, 3]  # Two 'b's
    
    def test_case_sensitive_verification(self):
        """Test that verification is case sensitive."""
        s1 = "ABC"
        s2 = "def"
        output = "abcdef"  # lowercase vs uppercase
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_invalid
        assert result.s1_match.is_invalid  # 'A' != 'a'
        assert result.s2_match.is_valid
    
    def test_special_characters(self):
        """Test verification with special characters."""
        s1 = "a!@#"
        s2 = "$%^b"
        output = "a!@#$%^b"
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
    
    def test_unicode_characters(self):
        """Test verification with unicode characters."""
        s1 = "café"
        s2 = "naïve"
        output = "café naïve"

        # FIXME: the valid output should be either "cafénaïve" or "naïvecafé"
        # While "café naïve" contains both strings, but it is not the shortest output.
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
    
    def test_primary_test_case_example(self):
        """Test with the primary algorithm test case."""
        s1 = "this is a red vase"
        s2 = "his son freddy love vase"
        # Example valid output that contains both as subsequences
        output = "this is son freddy love a red vase"

        # FIXME: a valid output for the s1 and s2 above should be this "this isonafreddy love vase"
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
    
    def test_error_details_generation(self):
        """Test detailed error message generation."""
        s1 = "abcdef"
        s2 = "xyz"
        output = "abcxyz"  # Missing 'd', 'e', 'f' from s1
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_invalid
        assert result.s1_match.is_invalid
        assert result.s2_match.is_valid
        
        # Check error details
        assert result.s1_match.error_details is not None
        assert "could not find 'd'" in result.s1_match.error_details
        assert "abc" in result.s1_match.error_details  # Successfully matched part
        assert "def" in result.s1_match.error_details  # Remaining unmatched part
    
    def test_detailed_match_info_report(self):
        """Test generation of detailed match information report."""
        s1 = "abc"
        s2 = "def"
        output = "adbecf"
        
        result = self.verifier.verify(s1, s2, output)
        report = self.verifier.get_detailed_match_info(result)
        
        assert "=== Subsequence Verification Report ===" in report
        assert "Overall Status: VALID" in report
        assert "First Input (s1): 'abc'" in report
        assert "Second Input (s2): 'def'" in report
        assert "Status: VALID" in report
        assert "Matched at positions:" in report
    
    def test_detailed_match_info_report_invalid(self):
        """Test detailed match report for invalid subsequences."""
        s1 = "abc"
        s2 = "def"
        output = "ab"  # Missing characters
        
        result = self.verifier.verify(s1, s2, output)
        report = self.verifier.get_detailed_match_info(result)
        
        assert "Overall Status: INVALID" in report
        assert "Status: INVALID" in report
        assert "Missing characters:" in report
        assert "Validation Errors:" in report
    
    def test_input_validation_type_errors(self):
        """Test that proper TypeErrors are raised for invalid input types."""
        with pytest.raises(TypeError, match="s1 must be a string"):
            self.verifier.verify(123, "def", "output")
        
        with pytest.raises(TypeError, match="s2 must be a string"):
            self.verifier.verify("abc", 456, "output")
        
        with pytest.raises(TypeError, match="output must be a string"):
            self.verifier.verify("abc", "def", 789)
    
    def test_long_strings_performance(self):
        """Test verification with longer strings to ensure reasonable performance."""
        s1 = "a" * 100 + "b" * 100
        s2 = "c" * 100 + "d" * 100
        output = "a" * 100 + "c" * 100 + "b" * 100 + "d" * 100
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        assert len(result.s1_match.output_positions) == 200
        assert len(result.s2_match.output_positions) == 200
    
    def test_edge_case_single_character_strings(self):
        """Test verification with single character strings."""
        result = self.verifier.verify("a", "b", "ab")
        assert result.is_valid
        
        result = self.verifier.verify("a", "b", "ba")
        assert result.is_valid
        
        result = self.verifier.verify("a", "b", "a")
        assert result.is_invalid
        assert result.s1_match.is_valid
        assert result.s2_match.is_invalid
    
    def test_whitespace_only_strings(self):
        """Test verification with whitespace-only strings."""
        result = self.verifier.verify(" ", "  ", "   ")
        assert result.is_valid
        
        result = self.verifier.verify("  ", " ", " ")
        assert result.is_invalid  # Need two spaces for first input
        assert result.s1_match.is_invalid
        assert result.s2_match.is_valid
        
    def test_space_matching_with_characters(self):
        """Test verification where characters match with spaces from another string."""
        s1 = "aa bb"
        s2 = "cc"
        output = "aaccbb"  # 'cc' from s2 matches with the space in s1
        
        result = self.verifier.verify(s1, s2, output)
        
        assert result.is_valid
        assert result.s1_match.is_valid
        assert result.s2_match.is_valid
        # Verify positions for s1 "aa bb" in "aaccbb"
        assert result.s1_match.output_positions == [0, 1, 4, 5]
        # Verify positions for s2 "cc" in "aaccbb"
        assert result.s2_match.output_positions == [2, 3]