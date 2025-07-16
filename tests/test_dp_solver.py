"""
Unit tests for the DPSolver class.

This module tests the dynamic programming algorithm implementation including
DP table initialization, state transitions, and basic solution reconstruction.
"""

import pytest
from shortest_combined_string.dp_solver import DPSolver, DPResult
from shortest_combined_string.models import WordToken, DPState, Operation, CombinedToken, TokenType


class TestDPSolver:
    """Test cases for the DPSolver class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = DPSolver()
    
    def test_solver_initialization(self):
        """Test that DPSolver can be initialized."""
        solver = DPSolver()
        assert solver is not None
    
    def test_solve_with_invalid_input_types(self):
        """Test that solve raises TypeError for invalid input types."""
        with pytest.raises(TypeError, match="s1_tokens must be a list"):
            self.solver.solve("not a list", [])
        
        with pytest.raises(TypeError, match="s2_tokens must be a list"):
            self.solver.solve([], "not a list")
        
        with pytest.raises(TypeError, match="All s1_tokens must be WordToken objects"):
            self.solver.solve(["not a token"], [])
        
        with pytest.raises(TypeError, match="All s2_tokens must be WordToken objects"):
            self.solver.solve([], ["not a token"])
    
    def test_solve_empty_inputs(self):
        """Test solving with empty token lists."""
        result = self.solver.solve([], [])
        
        assert isinstance(result, DPResult)
        assert result.optimal_length == 0
        assert len(result.dp_table) == 1
        assert len(result.dp_table[0]) == 1
        assert result.dp_table[0][0].length == 0
        assert result.dp_table[0][0].operation == Operation.SKIP
        assert len(result.solution) == 0
    
    def test_solve_single_token_s1_only(self):
        """Test solving with only s1 having tokens."""
        s1_tokens = [WordToken("hello", 0, 0, 0)]
        s2_tokens = []
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        assert result.optimal_length == 5  # "hello"
        assert len(result.dp_table) == 2
        assert len(result.dp_table[0]) == 1
        assert len(result.dp_table[1]) == 1
        
        # Check base case initialization
        assert result.dp_table[0][0].length == 0
        assert result.dp_table[1][0].length == 5
        assert result.dp_table[1][0].operation == Operation.INSERT_S1
        
        # Check solution
        assert len(result.solution) == 1
        assert result.solution[0].content == "hello"
        assert result.solution[0].type == TokenType.S1_ONLY
        assert result.solution[0].source_s1_words == [0]
        assert result.solution[0].source_s2_words == []
    
    def test_solve_single_token_s2_only(self):
        """Test solving with only s2 having tokens."""
        s1_tokens = []
        s2_tokens = [WordToken("world", 0, 0, 0)]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        assert result.optimal_length == 5  # "world"
        assert len(result.dp_table) == 1
        assert len(result.dp_table[0]) == 2
        
        # Check base case initialization
        assert result.dp_table[0][0].length == 0
        assert result.dp_table[0][1].length == 5
        assert result.dp_table[0][1].operation == Operation.INSERT_S2
        
        # Check solution
        assert len(result.solution) == 1
        assert result.solution[0].content == "world"
        assert result.solution[0].type == TokenType.S2_ONLY
        assert result.solution[0].source_s1_words == []
        assert result.solution[0].source_s2_words == [0]
    
    def test_solve_single_tokens_both_strings(self):
        """Test solving with one token in each string."""
        s1_tokens = [WordToken("hello", 0, 0, 0)]
        s2_tokens = [WordToken("world", 0, 0, 0)]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Should choose the optimal combination
        assert result.optimal_length <= 11  # "hello world" or better
        assert len(result.dp_table) == 2
        assert len(result.dp_table[0]) == 2
        assert len(result.dp_table[1]) == 2
        
        # Check that all DP states are properly initialized
        assert result.dp_table[0][0].length == 0
        assert result.dp_table[1][0].length == 5  # "hello"
        assert result.dp_table[0][1].length == 5  # "world"
        assert result.dp_table[1][1].length > 0  # Some combination
        
        # Solution should contain both words as subsequences
        assert len(result.solution) > 0
        solution_content = "".join(token.content for token in result.solution)
        clean_content = solution_content.replace(" ", "")
        assert self._is_subsequence("hello", clean_content)
        assert self._is_subsequence("world", clean_content)
    
    def test_dp_table_initialization_with_spaces(self):
        """Test DP table initialization with tokens that have spaces."""
        s1_tokens = [WordToken("hello", 1, 2, 0)]  # " hello  "
        s2_tokens = [WordToken("world", 0, 1, 0)]  # "world "
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Check that spaces are accounted for in base cases
        assert result.dp_table[1][0].length == 8  # 1 + 5 + 2 = " hello  "
        assert result.dp_table[0][1].length == 6  # 0 + 5 + 1 = "world "
    
    def test_dp_table_structure_multiple_tokens(self):
        """Test DP table structure with multiple tokens."""
        s1_tokens = [
            WordToken("this", 0, 1, 0),  # "this "
            WordToken("is", 0, 1, 1)     # "is "
        ]
        s2_tokens = [
            WordToken("that", 0, 1, 0),  # "that "
            WordToken("was", 0, 0, 1)    # "was"
        ]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Check table dimensions
        assert len(result.dp_table) == 3  # 0, 1, 2 for s1
        assert len(result.dp_table[0]) == 3  # 0, 1, 2 for s2
        assert len(result.dp_table[1]) == 3
        assert len(result.dp_table[2]) == 3
        
        # Check that all states are DPState objects
        for i in range(3):
            for j in range(3):
                assert isinstance(result.dp_table[i][j], DPState)
                assert result.dp_table[i][j].length >= 0
                assert isinstance(result.dp_table[i][j].operation, Operation)
    
    def test_state_transitions_insert_operations(self):
        """Test that state transitions correctly handle INSERT operations."""
        s1_tokens = [WordToken("hello", 0, 0, 0)]
        s2_tokens = [WordToken("world", 0, 0, 0)]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Check that INSERT_S1 and INSERT_S2 operations are used
        operations_used = set()
        for i in range(len(result.dp_table)):
            for j in range(len(result.dp_table[i])):
                operations_used.add(result.dp_table[i][j].operation)
        
        assert Operation.INSERT_S1 in operations_used
        assert Operation.INSERT_S2 in operations_used
    
    def test_solution_reconstruction_preserves_word_order(self):
        """Test that solution reconstruction preserves word order from inputs."""
        s1_tokens = [
            WordToken("first", 0, 1, 0),
            WordToken("second", 0, 0, 1)
        ]
        s2_tokens = [
            WordToken("alpha", 0, 1, 0),
            WordToken("beta", 0, 0, 1)
        ]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Extract all words from solution
        solution_text = "".join(token.content for token in result.solution)
        
        # Check that words from s1 appear in order
        first_pos = solution_text.find("first")
        second_pos = solution_text.find("second")
        assert first_pos >= 0 and second_pos >= 0
        assert first_pos < second_pos
        
        # Check that words from s2 appear in order
        alpha_pos = solution_text.find("alpha")
        beta_pos = solution_text.find("beta")
        assert alpha_pos >= 0 and beta_pos >= 0
        assert alpha_pos < beta_pos
    
    def test_match_operation_combines_tokens(self):
        """Test that MATCH operation properly combines tokens."""
        s1_tokens = [WordToken("test", 0, 0, 0)]
        s2_tokens = [WordToken("word", 0, 0, 0)]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Find any MERGED tokens in the solution
        merged_tokens = [token for token in result.solution if token.type == TokenType.MERGED]
        
        if merged_tokens:  # If MATCH operation was chosen
            merged_token = merged_tokens[0]
            assert len(merged_token.source_s1_words) > 0
            assert len(merged_token.source_s2_words) > 0
            assert "test" in merged_token.content
            assert "word" in merged_token.content
    
    def test_optimal_length_calculation(self):
        """Test that optimal length is correctly calculated."""
        s1_tokens = [WordToken("a", 0, 0, 0)]
        s2_tokens = [WordToken("b", 0, 0, 0)]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Verify that optimal_length matches the final DP table entry
        assert result.optimal_length == result.dp_table[-1][-1].length
        
        # Verify that solution length is consistent
        solution_length = sum(len(token.content) for token in result.solution)
        assert solution_length == result.optimal_length
    
    def test_dp_state_validation(self):
        """Test that all DP states have valid properties."""
        s1_tokens = [WordToken("hello", 0, 1, 0)]
        s2_tokens = [WordToken("world", 1, 0, 0)]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Check all DP states
        for i in range(len(result.dp_table)):
            for j in range(len(result.dp_table[i])):
                state = result.dp_table[i][j]
                
                # Validate state properties
                assert state.length >= 0
                assert state.s1_word_index == i
                assert state.s2_word_index == j
                assert isinstance(state.operation, Operation)
                
                # Length should be non-decreasing as we move through table
                if i > 0:
                    assert state.length >= result.dp_table[i-1][j].length
                if j > 0:
                    assert state.length >= result.dp_table[i][j-1].length
    
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


class TestDPResult:
    """Test cases for the DPResult class."""
    
    def test_dp_result_creation(self):
        """Test that DPResult can be created with valid parameters."""
        dp_table = [[DPState(0, 0, 0, Operation.SKIP)]]
        solution = []
        
        result = DPResult(0, dp_table, solution)
        
        assert result.optimal_length == 0
        assert result.dp_table == dp_table
        assert result.solution == solution
    
    def test_dp_result_properties(self):
        """Test that DPResult properties are accessible."""
        dp_table = [[DPState(5, 1, 1, Operation.MATCH)]]
        solution = [CombinedToken("test", [0], [0], TokenType.MERGED)]
        
        result = DPResult(5, dp_table, solution)
        
        assert result.optimal_length == 5
        assert len(result.dp_table) == 1
        assert len(result.solution) == 1
        assert result.solution[0].content == "test"


class TestCharacterReuseOptimization:
    """Test cases for character reuse optimization strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = DPSolver()
    
    def test_substring_containment_s1_contains_s2(self):
        """Test substring containment when s1 word contains s2 word."""
        s1_tokens = [WordToken("testing", 0, 0, 0)]  # "testing"
        s2_tokens = [WordToken("test", 0, 0, 0)]     # "test"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Should use substring containment optimization
        # "testing" contains "test", so result should be just "testing"
        assert result.optimal_length == 7  # len("testing")
        
        # Check that the solution uses the containment strategy
        merged_tokens = [token for token in result.solution if token.type == TokenType.MERGED]
        if merged_tokens:
            assert "testing" in merged_tokens[0].content
            assert len(merged_tokens[0].content.strip()) == 7
    
    def test_substring_containment_s2_contains_s1(self):
        """Test substring containment when s2 word contains s1 word."""
        s1_tokens = [WordToken("red", 0, 0, 0)]      # "red"
        s2_tokens = [WordToken("freddy", 0, 0, 0)]   # "freddy"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Should use substring containment optimization
        # "freddy" contains "red", so result should be just "freddy"
        assert result.optimal_length == 6  # len("freddy")
        
        # Check that the solution uses the containment strategy
        merged_tokens = [token for token in result.solution if token.type == TokenType.MERGED]
        if merged_tokens:
            assert "freddy" in merged_tokens[0].content
            assert len(merged_tokens[0].content.strip()) == 6
    
    def test_prefix_suffix_overlap_optimization(self):
        """Test prefix/suffix overlap optimization."""
        s1_tokens = [WordToken("hello", 0, 0, 0)]    # "hello"
        s2_tokens = [WordToken("love", 0, 0, 0)]     # "love"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # "hello" ends with "lo" and "love" starts with "lo"
        # So optimal combination should be "hellove" (length 7) instead of "hello love" (length 10)
        # But we need to check if this overlap actually exists
        
        # Check if any optimization was applied
        basic_length = len("hello") + len("love") + 1  # +1 for space = 10
        assert result.optimal_length <= basic_length
        
        # Verify solution contains both words as subsequences
        solution_content = "".join(token.content for token in result.solution)
        assert self._is_subsequence("hello", solution_content.replace(" ", ""))
        assert self._is_subsequence("love", solution_content.replace(" ", ""))
    
    def test_character_interleaving_optimization(self):
        """Test character interleaving optimization."""
        s1_tokens = [WordToken("abc", 0, 0, 0)]      # "abc"
        s2_tokens = [WordToken("aec", 0, 0, 0)]      # "aec"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Shortest supersequence of "abc" and "aec" should be "abec" (length 4)
        # instead of "abc aec" (length 7)
        basic_length = len("abc") + len("aec") + 1  # +1 for space = 7
        assert result.optimal_length < basic_length
        
        # Verify solution contains both words as subsequences
        solution_content = "".join(token.content for token in result.solution)
        clean_content = solution_content.replace(" ", "")
        assert self._is_subsequence("abc", clean_content)
        assert self._is_subsequence("aec", clean_content)
    
    def test_multiple_optimization_strategies_comparison(self):
        """Test that the algorithm chooses the best among multiple strategies."""
        s1_tokens = [WordToken("test", 0, 0, 0)]     # "test"
        s2_tokens = [WordToken("testing", 0, 0, 0)]  # "testing"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # "testing" contains "test", so substring containment should be chosen
        # Result should be "testing" (length 7) instead of "test testing" (length 12)
        assert result.optimal_length == 7
        
        # Verify both words appear as subsequences
        solution_content = "".join(token.content for token in result.solution)
        clean_content = solution_content.replace(" ", "")
        assert self._is_subsequence("test", clean_content)
        assert self._is_subsequence("testing", clean_content)
    
    def test_character_reuse_with_spaces(self):
        """Test character reuse optimization with leading/trailing spaces."""
        s1_tokens = [WordToken("hello", 1, 1, 0)]    # " hello "
        s2_tokens = [WordToken("world", 0, 1, 0)]    # "world "
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Should preserve maximum spacing while optimizing character reuse
        solution_content = "".join(token.content for token in result.solution)
        
        # Check that spacing is preserved appropriately
        assert solution_content.startswith(" ") or any(token.content.startswith(" ") for token in result.solution)
        assert solution_content.endswith(" ") or any(token.content.endswith(" ") for token in result.solution)
        
        # Verify both words appear as subsequences
        clean_content = solution_content.replace(" ", "")
        assert self._is_subsequence("hello", clean_content)
        assert self._is_subsequence("world", clean_content)
    
    def test_no_optimization_possible_fallback(self):
        """Test fallback to basic concatenation when no optimization is possible."""
        s1_tokens = [WordToken("xyz", 0, 0, 0)]      # "xyz"
        s2_tokens = [WordToken("abc", 0, 0, 0)]      # "abc"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # With character interleaving, "xyz" and "abc" can be optimized to "abcxyz" (length 6)
        # which is better than "xyz abc" (length 7)
        basic_length = len("xyz") + len("abc") + 1  # +1 for space = 7
        assert result.optimal_length <= basic_length
        
        # Verify both words appear as subsequences
        solution_content = "".join(token.content for token in result.solution)
        clean_content = solution_content.replace(" ", "")
        assert self._is_subsequence("xyz", clean_content)
        assert self._is_subsequence("abc", clean_content)
    
    def test_complex_character_interleaving(self):
        """Test complex character interleaving with longer words."""
        s1_tokens = [WordToken("abcdef", 0, 0, 0)]   # "abcdef"
        s2_tokens = [WordToken("aebdcf", 0, 0, 0)]   # "aebdcf"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Should find optimal interleaving
        basic_length = len("abcdef") + len("aebdcf") + 1  # +1 for space = 13
        assert result.optimal_length < basic_length
        
        # Verify both words appear as subsequences
        solution_content = "".join(token.content for token in result.solution)
        clean_content = solution_content.replace(" ", "")
        assert self._is_subsequence("abcdef", clean_content)
        assert self._is_subsequence("aebdcf", clean_content)
    
    def test_optimization_with_multiple_word_pairs(self):
        """Test optimization across multiple word pairs in longer sequences."""
        s1_tokens = [
            WordToken("this", 0, 1, 0),     # "this "
            WordToken("test", 0, 0, 1)      # "test"
        ]
        s2_tokens = [
            WordToken("that", 0, 1, 0),     # "that "
            WordToken("testing", 0, 0, 1)   # "testing"
        ]
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Should optimize each word pair independently
        # "test" and "testing" should use substring containment
        basic_total = sum(token.total_length for token in s1_tokens) + \
                     sum(token.total_length for token in s2_tokens)
        
        assert result.optimal_length < basic_total
        
        # Verify all words appear as subsequences in the solution
        solution_content = "".join(token.content for token in result.solution)
        clean_content = solution_content.replace(" ", "")
        assert self._is_subsequence("this", clean_content)
        assert self._is_subsequence("that", clean_content)
        assert self._is_subsequence("test", clean_content)
        assert self._is_subsequence("testing", clean_content)
    
    def test_edge_case_identical_words(self):
        """Test optimization when words are identical."""
        s1_tokens = [WordToken("same", 0, 0, 0)]     # "same"
        s2_tokens = [WordToken("same", 0, 0, 0)]     # "same"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Should use substring containment (both words are identical)
        assert result.optimal_length == 4  # len("same")
        
        # Verify the word appears in the solution
        solution_content = "".join(token.content for token in result.solution)
        clean_content = solution_content.replace(" ", "")
        assert "same" in clean_content
        assert len(clean_content) == 4
    
    def test_edge_case_empty_words(self):
        """Test optimization with empty words."""
        s1_tokens = [WordToken("", 0, 0, 0)]         # ""
        s2_tokens = [WordToken("word", 0, 0, 0)]     # "word"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # Should handle empty words gracefully
        assert result.optimal_length == 4  # len("word")
        
        # Verify the non-empty word appears in the solution
        solution_content = "".join(token.content for token in result.solution)
        assert "word" in solution_content
    
    def test_strategy_selection_correctness(self):
        """Test that the correct optimization strategy is selected."""
        # Test case where substring containment should be preferred over other strategies
        s1_tokens = [WordToken("programming", 0, 0, 0)]  # "programming"
        s2_tokens = [WordToken("gram", 0, 0, 0)]         # "gram"
        
        result = self.solver.solve(s1_tokens, s2_tokens)
        
        # "programming" contains "gram", so should use substring containment
        assert result.optimal_length == 11  # len("programming")
        
        # Verify both words appear as subsequences
        solution_content = "".join(token.content for token in result.solution)
        clean_content = solution_content.replace(" ", "")
        assert self._is_subsequence("programming", clean_content)
        assert self._is_subsequence("gram", clean_content)
        assert "programming" in clean_content
    
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