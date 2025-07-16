"""
Dynamic Programming solver for the Shortest Combined String algorithm.

This module implements the core DP algorithm that finds the optimal way to combine
two sequences of word tokens while minimizing the total length and preserving
subsequence relationships.
"""

from typing import List, Optional
from .models import WordToken, DPState, Operation, CombinedToken, TokenType


class DPResult:
    """
    Result of the dynamic programming algorithm.
    
    Attributes:
        optimal_length: The minimum length achievable
        dp_table: The complete DP table for debugging/analysis
        solution: List of CombinedToken objects representing the solution
    """
    
    def __init__(self, optimal_length: int, dp_table: List[List[DPState]], solution: List[CombinedToken]):
        self.optimal_length = optimal_length
        self.dp_table = dp_table
        self.solution = solution


class DPSolver:
    """
    Dynamic Programming solver for optimal string combination.
    
    This class implements a word-boundary aware DP algorithm that finds the shortest
    way to combine two sequences of word tokens while preserving subsequence integrity.
    """
    
    def __init__(self):
        """Initialize the DP solver."""
        pass
    
    def solve(self, s1_tokens: List[WordToken], s2_tokens: List[WordToken]) -> DPResult:
        """
        Solve the shortest combined string problem using dynamic programming.
        
        Args:
            s1_tokens: Word tokens from the first string
            s2_tokens: Word tokens from the second string
            
        Returns:
            DPResult containing the optimal solution
            
        Raises:
            TypeError: If inputs are not lists of WordToken objects
        """
        if not isinstance(s1_tokens, list):
            raise TypeError("s1_tokens must be a list")
        if not isinstance(s2_tokens, list):
            raise TypeError("s2_tokens must be a list")
        
        for token in s1_tokens:
            if not isinstance(token, WordToken):
                raise TypeError("All s1_tokens must be WordToken objects")
        
        for token in s2_tokens:
            if not isinstance(token, WordToken):
                raise TypeError("All s2_tokens must be WordToken objects")
        
        # Initialize DP table
        dp_table = self._initialize_dp_table(s1_tokens, s2_tokens)
        
        # Fill DP table with optimal solutions
        self._fill_dp_table(dp_table, s1_tokens, s2_tokens)
        
        # Extract optimal length
        optimal_length = dp_table[-1][-1].length
        
        # Reconstruct solution path (basic implementation for now)
        solution = self._reconstruct_solution(dp_table, s1_tokens, s2_tokens)
        
        return DPResult(optimal_length, dp_table, solution)
    
    def _initialize_dp_table(self, s1_tokens: List[WordToken], s2_tokens: List[WordToken]) -> List[List[DPState]]:
        """
        Initialize the DP table with base cases.
        
        Args:
            s1_tokens: Word tokens from the first string
            s2_tokens: Word tokens from the second string
            
        Returns:
            Initialized DP table with base cases filled
        """
        rows = len(s1_tokens) + 1
        cols = len(s2_tokens) + 1
        
        # Create table with default states
        dp_table = [[None for _ in range(cols)] for _ in range(rows)]
        
        # Base case: dp[0][0] = empty strings
        dp_table[0][0] = DPState(
            length=0,
            s1_word_index=0,
            s2_word_index=0,
            operation=Operation.SKIP
        )
        
        # Base case: dp[i][0] = take all words from s1 only
        cumulative_length = 0
        for i in range(1, rows):
            token = s1_tokens[i-1]
            cumulative_length += token.total_length
            dp_table[i][0] = DPState(
                length=cumulative_length,
                s1_word_index=i,
                s2_word_index=0,
                operation=Operation.INSERT_S1
            )
        
        # Base case: dp[0][j] = take all words from s2 only
        cumulative_length = 0
        for j in range(1, cols):
            token = s2_tokens[j-1]
            cumulative_length += token.total_length
            dp_table[0][j] = DPState(
                length=cumulative_length,
                s1_word_index=0,
                s2_word_index=j,
                operation=Operation.INSERT_S2
            )
        
        return dp_table
    
    def _fill_dp_table(self, dp_table: List[List[DPState]], s1_tokens: List[WordToken], s2_tokens: List[WordToken]):
        """
        Fill the DP table using optimal substructure.
        
        Args:
            dp_table: The initialized DP table to fill
            s1_tokens: Word tokens from the first string
            s2_tokens: Word tokens from the second string
        """
        rows = len(s1_tokens) + 1
        cols = len(s2_tokens) + 1
        
        for i in range(1, rows):
            for j in range(1, cols):
                s1_token = s1_tokens[i-1]
                s2_token = s2_tokens[j-1]
                
                # Option 1: Insert word from s1 only
                insert_s1_cost = dp_table[i-1][j].length + s1_token.total_length
                
                # Option 2: Insert word from s2 only
                insert_s2_cost = dp_table[i][j-1].length + s2_token.total_length
                
                # Option 3: Try to match/merge words (basic implementation)
                match_cost = self._calculate_match_cost(dp_table[i-1][j-1], s1_token, s2_token)
                
                # Choose the option with minimum cost
                options = [
                    (insert_s1_cost, Operation.INSERT_S1),
                    (insert_s2_cost, Operation.INSERT_S2),
                    (match_cost, Operation.MATCH)
                ]
                
                min_cost, best_operation = min(options, key=lambda x: x[0])
                
                dp_table[i][j] = DPState(
                    length=min_cost,
                    s1_word_index=i,
                    s2_word_index=j,
                    operation=best_operation
                )
    
    def _calculate_match_cost(self, prev_state: DPState, s1_token: WordToken, s2_token: WordToken) -> int:
        """
        Calculate the cost of matching/merging two word tokens.
        
        This is a basic implementation that simply concatenates the words.
        Future optimizations will implement character reuse strategies.
        
        Args:
            prev_state: The previous DP state
            s1_token: Word token from first string
            s2_token: Word token from second string
            
        Returns:
            Cost of matching these two tokens
        """
        # Basic implementation: just concatenate both words with a space
        # This ensures both words appear in the output as required
        combined_length = s1_token.total_length + s2_token.total_length
        
        # Add minimal spacing between words if needed
        if s1_token.trailing_spaces == 0 and s2_token.leading_spaces == 0:
            combined_length += 1  # Add one space between words
        
        return prev_state.length + combined_length
    
    def _reconstruct_solution(self, dp_table: List[List[DPState]], s1_tokens: List[WordToken], s2_tokens: List[WordToken]) -> List[CombinedToken]:
        """
        Reconstruct the optimal solution from the DP table.
        
        Args:
            dp_table: The filled DP table
            s1_tokens: Word tokens from the first string
            s2_tokens: Word tokens from the second string
            
        Returns:
            List of CombinedToken objects representing the solution
        """
        solution = []
        i = len(s1_tokens)
        j = len(s2_tokens)
        
        # Backtrack through the DP table
        while i > 0 or j > 0:
            current_state = dp_table[i][j]
            
            if current_state.operation == Operation.INSERT_S1:
                # Take word from s1 only
                token = s1_tokens[i-1]
                combined_token = CombinedToken(
                    content=self._format_token_content(token),
                    source_s1_words=[i-1],
                    source_s2_words=[],
                    type=TokenType.S1_ONLY
                )
                solution.append(combined_token)
                i -= 1
                
            elif current_state.operation == Operation.INSERT_S2:
                # Take word from s2 only
                token = s2_tokens[j-1]
                combined_token = CombinedToken(
                    content=self._format_token_content(token),
                    source_s1_words=[],
                    source_s2_words=[j-1],
                    type=TokenType.S2_ONLY
                )
                solution.append(combined_token)
                j -= 1
                
            elif current_state.operation == Operation.MATCH:
                # Match/merge both words
                s1_token = s1_tokens[i-1]
                s2_token = s2_tokens[j-1]
                
                # Basic implementation: concatenate with space
                content = self._format_token_content(s1_token) + " " + self._format_token_content(s2_token)
                
                combined_token = CombinedToken(
                    content=content,
                    source_s1_words=[i-1],
                    source_s2_words=[j-1],
                    type=TokenType.MERGED
                )
                solution.append(combined_token)
                i -= 1
                j -= 1
                
            else:  # Operation.SKIP
                break
        
        # Reverse solution since we built it backwards
        solution.reverse()
        return solution
    
    def _format_token_content(self, token: WordToken) -> str:
        """
        Format a WordToken into its string representation.
        
        Args:
            token: The WordToken to format
            
        Returns:
            String representation including spaces
        """
        return " " * token.leading_spaces + token.word + " " * token.trailing_spaces