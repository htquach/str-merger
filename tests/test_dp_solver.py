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
        
        # Solution should contain both words
        assert len(result.solution) > 0
        solution_content = "".join(token.content for token in result.solution)
        assert "hello" in solution_content
        assert "world" in solution_content
    
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