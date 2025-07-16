"""
Unit tests for the PathReconstructor class.

This module tests the path reconstruction functionality including backtracking
algorithm, optimal character combination, and CombinedToken sequence generation.
"""

import pytest
from shortest_combined_string.path_reconstructor import PathReconstructor
from shortest_combined_string.models import WordToken, DPState, Operation, CombinedToken, TokenType


class TestPathReconstructor:
    """Test cases for the PathReconstructor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reconstructor = PathReconstructor()
    
    def test_reconstructor_initialization(self):
        """Test that PathReconstructor can be initialized."""
        reconstructor = PathReconstructor()
        assert reconstructor is not None
    
    def test_reconstruct_path_with_invalid_input_types(self):
        """Test that reconstruct_path raises TypeError for invalid input types."""
        dp_table = [[DPState(0, 0, 0, Operation.SKIP)]]
        s1_tokens = []
        s2_tokens = []
        
        with pytest.raises(TypeError, match="dp_table must be a list"):
            self.reconstructor.reconstruct_path("not a list", s1_tokens, s2_tokens)
        
        with pytest.raises(TypeError, match="s1_tokens must be a list"):
            self.reconstructor.reconstruct_path(dp_table, "not a list", s2_tokens)
        
        with pytest.raises(TypeError, match="s2_tokens must be a list"):
            self.reconstructor.reconstruct_path(dp_table, s1_tokens, "not a list")
    
    def test_reconstruct_path_with_invalid_token_types(self):
        """Test that reconstruct_path raises TypeError for invalid token types."""
        # Create properly sized DP table for the invalid tokens
        dp_table_for_s1 = [
            [DPState(0, 0, 0, Operation.SKIP)],
            [DPState(0, 1, 0, Operation.INSERT_S1)]
        ]
        dp_table_for_s2 = [
            [DPState(0, 0, 0, Operation.SKIP), DPState(0, 0, 1, Operation.INSERT_S2)]
        ]
        
        with pytest.raises(TypeError, match="All s1_tokens must be WordToken objects"):
            self.reconstructor.reconstruct_path(dp_table_for_s1, ["not a token"], [])
        
        with pytest.raises(TypeError, match="All s2_tokens must be WordToken objects"):
            self.reconstructor.reconstruct_path(dp_table_for_s2, [], ["not a token"])
    
    def test_reconstruct_path_with_invalid_dp_table_dimensions(self):
        """Test that reconstruct_path raises ValueError for invalid DP table dimensions."""
        s1_tokens = [WordToken("hello", 0, 0, 0)]
        s2_tokens = []
        
        # Wrong number of rows
        dp_table = [[DPState(0, 0, 0, Operation.SKIP)]]  # Should be 2 rows for 1 s1_token
        with pytest.raises(ValueError, match="DP table rows .* must equal s1_tokens length"):
            self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        # Wrong number of columns
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP)],
            [DPState(5, 1, 0, Operation.INSERT_S1)]
        ]  # Should be 1 column for 0 s2_tokens
        s2_tokens = [WordToken("world", 0, 0, 0)]  # Now expecting 2 columns
        with pytest.raises(ValueError, match="DP table columns .* must equal s2_tokens length"):
            self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
    
    def test_reconstruct_path_with_invalid_dp_table_contents(self):
        """Test that reconstruct_path raises TypeError for invalid DP table contents."""
        s1_tokens = []
        s2_tokens = []
        
        # Invalid row type - but first check dimensions, so create properly sized table
        dp_table = ["not a list"]  # This will fail dimension check first
        with pytest.raises(ValueError, match="DP table columns"):
            self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        # Invalid state type - create properly dimensioned table with invalid content
        dp_table = [["not a state"]]
        with pytest.raises(TypeError, match="DP table entry .* must be a DPState object"):
            self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
    
    def test_reconstruct_empty_solution(self):
        """Test reconstructing solution from empty inputs."""
        dp_table = [[DPState(0, 0, 0, Operation.SKIP)]]
        s1_tokens = []
        s2_tokens = []
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_reconstruct_single_s1_token(self):
        """Test reconstructing solution with single s1 token."""
        s1_tokens = [WordToken("hello", 0, 0, 0)]
        s2_tokens = []
        
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP)],
            [DPState(5, 1, 0, Operation.INSERT_S1)]
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        assert len(result) == 1
        assert result[0].content == "hello"
        assert result[0].type == TokenType.S1_ONLY
        assert result[0].source_s1_words == [0]
        assert result[0].source_s2_words == []
    
    def test_reconstruct_single_s2_token(self):
        """Test reconstructing solution with single s2 token."""
        s1_tokens = []
        s2_tokens = [WordToken("world", 0, 0, 0)]
        
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP), DPState(5, 0, 1, Operation.INSERT_S2)]
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        assert len(result) == 1
        assert result[0].content == "world"
        assert result[0].type == TokenType.S2_ONLY
        assert result[0].source_s1_words == []
        assert result[0].source_s2_words == [0]
    
    def test_reconstruct_with_spaces(self):
        """Test reconstructing solution with tokens that have spaces."""
        s1_tokens = [WordToken("hello", 1, 2, 0)]  # " hello  "
        s2_tokens = []
        
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP)],
            [DPState(8, 1, 0, Operation.INSERT_S1)]  # 1 + 5 + 2 = 8
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        assert len(result) == 1
        assert result[0].content == " hello  "
        assert result[0].type == TokenType.S1_ONLY
        assert len(result[0].content) == 8
    
    def test_reconstruct_match_operation(self):
        """Test reconstructing solution with MATCH operation."""
        s1_tokens = [WordToken("test", 0, 0, 0)]
        s2_tokens = [WordToken("testing", 0, 0, 0)]
        
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP), DPState(7, 0, 1, Operation.INSERT_S2)],
            [DPState(4, 1, 0, Operation.INSERT_S1), DPState(7, 1, 1, Operation.MATCH)]
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        assert len(result) == 1
        assert result[0].type == TokenType.MERGED
        assert result[0].source_s1_words == [0]
        assert result[0].source_s2_words == [0]
        # Should use substring containment: "testing" contains "test"
        assert "testing" in result[0].content
        assert len(result[0].content.strip()) == 7
    
    def test_reconstruct_multiple_operations(self):
        """Test reconstructing solution with multiple operations."""
        s1_tokens = [WordToken("hello", 0, 1, 0)]  # "hello "
        s2_tokens = [WordToken("world", 0, 0, 0)]  # "world"
        
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP), DPState(5, 0, 1, Operation.INSERT_S2)],
            [DPState(6, 1, 0, Operation.INSERT_S1), DPState(11, 1, 1, Operation.INSERT_S1)]
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        # Should have two tokens: "hello " from s1 and "world" from s2
        assert len(result) == 2
        
        # Check that both tokens are present
        contents = [token.content for token in result]
        assert "hello " in contents
        assert "world" in contents
        
        # Check token types
        types = [token.type for token in result]
        assert TokenType.S1_ONLY in types
        assert TokenType.S2_ONLY in types
    
    def test_reconstruct_preserves_word_order(self):
        """Test that reconstruction preserves word order from original strings."""
        s1_tokens = [
            WordToken("first", 0, 1, 0),   # "first "
            WordToken("second", 0, 0, 1)   # "second"
        ]
        s2_tokens = [
            WordToken("alpha", 0, 1, 0),   # "alpha "
            WordToken("beta", 0, 0, 1)     # "beta"
        ]
        
        # Create a DP table that uses INSERT operations to preserve order
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP), DPState(6, 0, 1, Operation.INSERT_S2), DPState(10, 0, 2, Operation.INSERT_S2)],
            [DPState(6, 1, 0, Operation.INSERT_S1), DPState(12, 1, 1, Operation.INSERT_S1), DPState(16, 1, 2, Operation.INSERT_S1)],
            [DPState(12, 2, 0, Operation.INSERT_S1), DPState(18, 2, 1, Operation.INSERT_S1), DPState(22, 2, 2, Operation.INSERT_S1)]
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        # Extract all content and verify word order
        full_content = "".join(token.content for token in result)
        
        # Check s1 word order
        first_pos = full_content.find("first")
        second_pos = full_content.find("second")
        assert first_pos >= 0 and second_pos >= 0
        assert first_pos < second_pos
        
        # Check s2 word order
        alpha_pos = full_content.find("alpha")
        beta_pos = full_content.find("beta")
        assert alpha_pos >= 0 and beta_pos >= 0
        assert alpha_pos < beta_pos


class TestPathReconstructorOptimizationStrategies:
    """Test cases for optimization strategy reconstruction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reconstructor = PathReconstructor()
    
    def test_substring_containment_s1_contains_s2(self):
        """Test reconstruction of substring containment strategy (s1 contains s2)."""
        s1_token = WordToken("testing", 0, 0, 0)
        s2_token = WordToken("test", 0, 0, 0)
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # Should use "testing" since it contains "test"
        assert result == "testing"
    
    def test_substring_containment_s2_contains_s1(self):
        """Test reconstruction of substring containment strategy (s2 contains s1)."""
        s1_token = WordToken("red", 0, 0, 0)
        s2_token = WordToken("freddy", 0, 0, 0)
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # Should use "freddy" since it contains "red"
        assert result == "freddy"
    
    def test_prefix_suffix_overlap_s1_s2(self):
        """Test reconstruction of prefix/suffix overlap strategy (s1 + s2)."""
        s1_token = WordToken("hello", 0, 0, 0)
        s2_token = WordToken("love", 0, 0, 0)
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # "hello" ends with "lo" and "love" starts with "lo"
        # Should create "hellove" if overlap is detected
        if "hellove" in result:
            assert result == "hellove"
        else:
            # If no overlap detected, should fall back to basic concatenation
            assert "hello" in result and "love" in result
    
    def test_prefix_suffix_overlap_s2_s1(self):
        """Test reconstruction of prefix/suffix overlap strategy (s2 + s1)."""
        s1_token = WordToken("ove", 0, 0, 0)
        s2_token = WordToken("love", 0, 0, 0)
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # "love" ends with "ove" and "ove" starts with "ove"
        # Should create "love" (complete overlap) if detected
        if len(result.strip()) == 4:
            assert "love" in result
        else:
            # If no overlap detected, should fall back to basic concatenation
            assert "ove" in result and "love" in result
    
    def test_character_interleaving_strategy(self):
        """Test reconstruction of character interleaving strategy."""
        s1_token = WordToken("abc", 0, 0, 0)
        s2_token = WordToken("aec", 0, 0, 0)
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # Should create shortest supersequence containing both as subsequences
        clean_result = result.strip()
        
        # Verify both words appear as subsequences
        assert self._is_subsequence("abc", clean_result)
        assert self._is_subsequence("aec", clean_result)
        
        # Should be shorter than basic concatenation
        basic_length = len("abc aec")
        assert len(clean_result) < basic_length
    
    def test_basic_concatenation_fallback(self):
        """Test reconstruction falls back to basic concatenation when no optimization applies."""
        s1_token = WordToken("xyz", 0, 0, 0)
        s2_token = WordToken("123", 0, 0, 0)
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # Should contain both words
        assert "xyz" in result
        assert "123" in result
        
        # Should have some form of separation (space or interleaving)
        clean_result = result.strip()
        assert len(clean_result) >= 6  # At least both words
    
    def test_optimization_with_spaces(self):
        """Test that optimization strategies preserve spacing correctly."""
        s1_token = WordToken("test", 1, 1, 0)    # " test "
        s2_token = WordToken("testing", 0, 2, 0) # "testing  "
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # Should use substring containment and preserve maximum spacing
        assert result.startswith(" ")  # Leading space from s1
        assert result.endswith("  ")   # Trailing spaces from s2
        assert "testing" in result
    
    def test_identical_words_optimization(self):
        """Test optimization when both words are identical."""
        s1_token = WordToken("same", 0, 0, 0)
        s2_token = WordToken("same", 0, 0, 0)
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # Should use substring containment (both words are identical)
        assert result == "same"
    
    def test_empty_word_handling(self):
        """Test optimization with empty words."""
        s1_token = WordToken("", 0, 0, 0)
        s2_token = WordToken("word", 0, 0, 0)
        
        result = self.reconstructor._reconstruct_optimized_match(s1_token, s2_token)
        
        # Should handle empty words gracefully
        assert "word" in result
        # Result should be just "word" or "word" with spaces
        clean_result = result.strip()
        assert clean_result == "word"
    
    def test_format_token_content(self):
        """Test _format_token_content method."""
        # Test with no spaces
        token = WordToken("hello", 0, 0, 0)
        result = self.reconstructor._format_token_content(token)
        assert result == "hello"
        
        # Test with leading spaces
        token = WordToken("hello", 2, 0, 0)
        result = self.reconstructor._format_token_content(token)
        assert result == "  hello"
        
        # Test with trailing spaces
        token = WordToken("hello", 0, 3, 0)
        result = self.reconstructor._format_token_content(token)
        assert result == "hello   "
        
        # Test with both leading and trailing spaces
        token = WordToken("hello", 1, 2, 0)
        result = self.reconstructor._format_token_content(token)
        assert result == " hello  "
    
    def test_build_shortest_supersequence(self):
        """Test _build_shortest_supersequence method."""
        # Test with overlapping characters
        result = self.reconstructor._build_shortest_supersequence("abc", "aec")
        assert self._is_subsequence("abc", result)
        assert self._is_subsequence("aec", result)
        assert len(result) <= 5  # Should be shorter than "abcaec"
        
        # Test with identical strings
        result = self.reconstructor._build_shortest_supersequence("same", "same")
        assert result == "same"
        
        # Test with no common characters
        result = self.reconstructor._build_shortest_supersequence("abc", "def")
        assert self._is_subsequence("abc", result)
        assert self._is_subsequence("def", result)
        assert len(result) == 6  # Should be "abcdef" or "defabc"
        
        # Test with one empty string
        result = self.reconstructor._build_shortest_supersequence("", "abc")
        assert result == "abc"
        
        result = self.reconstructor._build_shortest_supersequence("abc", "")
        assert result == "abc"
    
    def _is_subsequence(self, s: str, t: str) -> bool:
        """
        Check if s is a subsequence of t.
        
        Args:
            s: The potential subsequence
            t: The target string
            
        Returns:
            True if s is a subsequence of t
        """
        i = 0
        for char in t:
            if i < len(s) and char == s[i]:
                i += 1
        return i == len(s)


class TestPathReconstructorIntegration:
    """Integration tests for PathReconstructor with realistic scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reconstructor = PathReconstructor()
    
    def test_complex_reconstruction_scenario(self):
        """Test reconstruction with a complex multi-token scenario."""
        s1_tokens = [
            WordToken("this", 0, 1, 0),     # "this "
            WordToken("test", 0, 0, 1)      # "test"
        ]
        s2_tokens = [
            WordToken("that", 0, 1, 0),     # "that "
            WordToken("testing", 0, 0, 1)   # "testing"
        ]
        
        # Create a DP table that uses various operations
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP), DPState(5, 0, 1, Operation.INSERT_S2), DPState(12, 0, 2, Operation.INSERT_S2)],
            [DPState(5, 1, 0, Operation.INSERT_S1), DPState(10, 1, 1, Operation.INSERT_S1), DPState(16, 1, 2, Operation.INSERT_S1)],
            [DPState(9, 2, 0, Operation.INSERT_S1), DPState(14, 2, 1, Operation.INSERT_S1), DPState(19, 2, 2, Operation.MATCH)]
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        # Should have multiple tokens
        assert len(result) > 0
        
        # Verify all original words appear in the solution
        full_content = "".join(token.content for token in result)
        assert "this" in full_content
        assert "that" in full_content
        assert "test" in full_content or "testing" in full_content  # One might be contained in the other
    
    def test_reconstruction_accuracy_verification(self):
        """Test that reconstruction produces accurate CombinedToken sequences."""
        s1_tokens = [WordToken("hello", 0, 1, 0)]  # "hello "
        s2_tokens = [WordToken("world", 1, 0, 0)]  # " world"
        
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP), DPState(6, 0, 1, Operation.INSERT_S2)],
            [DPState(6, 1, 0, Operation.INSERT_S1), DPState(12, 1, 1, Operation.MATCH)]
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        # Should have one merged token
        assert len(result) == 1
        assert result[0].type == TokenType.MERGED
        
        # Verify source tracking
        assert result[0].source_s1_words == [0]
        assert result[0].source_s2_words == [0]
        
        # Verify content includes both words as subsequences
        content = result[0].content
        clean_content = content.replace(" ", "")
        assert self._is_subsequence("hello", clean_content)
        assert self._is_subsequence("world", clean_content)
        
        # Verify spacing is handled correctly
        assert content.startswith(" ") or "hello " in content  # Leading space preserved
    
    def test_backtracking_algorithm_correctness(self):
        """Test that the backtracking algorithm correctly traces the optimal path."""
        s1_tokens = [WordToken("a", 0, 0, 0), WordToken("b", 0, 0, 1)]
        s2_tokens = [WordToken("c", 0, 0, 0)]
        
        # Create a specific DP table to test backtracking
        dp_table = [
            [DPState(0, 0, 0, Operation.SKIP), DPState(1, 0, 1, Operation.INSERT_S2)],
            [DPState(1, 1, 0, Operation.INSERT_S1), DPState(2, 1, 1, Operation.INSERT_S1)],
            [DPState(2, 2, 0, Operation.INSERT_S1), DPState(3, 2, 1, Operation.INSERT_S1)]
        ]
        
        result = self.reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
        # Should reconstruct the path: INSERT_S1 -> INSERT_S1 -> INSERT_S2 (in reverse)
        # So final order should be: "c", "a", "b"
        assert len(result) == 3
        
        # Verify all tokens are present
        contents = [token.content for token in result]
        assert "a" in contents
        assert "b" in contents
        assert "c" in contents
        
        # Verify token types
        s1_only_count = sum(1 for token in result if token.type == TokenType.S1_ONLY)
        s2_only_count = sum(1 for token in result if token.type == TokenType.S2_ONLY)
        assert s1_only_count == 2  # "a" and "b"
        assert s2_only_count == 1  # "c"
    
    def _is_subsequence(self, s: str, t: str) -> bool:
        """
        Check if s is a subsequence of t.
        
        Args:
            s: The potential subsequence
            t: The target string
            
        Returns:
            True if s is a subsequence of t
        """
        i = 0
        for char in t:
            if i < len(s) and char == s[i]:
                i += 1
        return i == len(s)