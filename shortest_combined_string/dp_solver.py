"""
Dynamic Programming solver for the Shortest Combined String algorithm.

This module implements the core DP algorithm that finds the optimal way to combine
two sequences of word tokens while minimizing the total length and preserving
subsequence relationships.
"""

from typing import List, Optional
from .models import WordToken, DPState, Operation, CombinedToken, TokenType
from .path_reconstructor import PathReconstructor
from .exceptions import DPSolverError


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
        self.path_reconstructor = PathReconstructor()
    
    def solve(self, s1_tokens: List[WordToken], s2_tokens: List[WordToken]) -> DPResult:
        """
        Solve the shortest combined string problem using dynamic programming.
        
        Args:
            s1_tokens: Word tokens from the first string
            s2_tokens: Word tokens from the second string
            
        Returns:
            DPResult containing the optimal solution
            
        Raises:
            DPSolverError: If inputs are not lists of WordToken objects or are invalid
        """
        if not isinstance(s1_tokens, list):
            raise DPSolverError(f"First token list must be a list, got {type(s1_tokens).__name__}")
        if not isinstance(s2_tokens, list):
            raise DPSolverError(f"Second token list must be a list, got {type(s2_tokens).__name__}")
        
        for i, token in enumerate(s1_tokens):
            if not isinstance(token, WordToken):
                raise DPSolverError(f"Token at index {i} in first token list must be a WordToken object, got {type(token).__name__}")
        
        for i, token in enumerate(s2_tokens):
            if not isinstance(token, WordToken):
                raise DPSolverError(f"Token at index {i} in second token list must be a WordToken object, got {type(token).__name__}")
        
        # Initialize DP table
        dp_table = self._initialize_dp_table(s1_tokens, s2_tokens)
        
        # Fill DP table with optimal solutions
        self._fill_dp_table(dp_table, s1_tokens, s2_tokens)
        
        # Extract optimal length
        optimal_length = dp_table[-1][-1].length
        
        # Reconstruct solution path using PathReconstructor
        solution = self.path_reconstructor.reconstruct_path(dp_table, s1_tokens, s2_tokens)
        
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
        Calculate the cost of matching/merging two word tokens with character reuse optimization.
        
        This implementation uses advanced character reuse strategies:
        - Prefix/suffix overlap detection
        - Substring containment optimization
        - Strategic character interleaving while maintaining word boundaries
        
        Args:
            prev_state: The previous DP state
            s1_token: Word token from first string
            s2_token: Word token from second string
            
        Returns:
            Cost of matching these two tokens with optimal character reuse
        """
        # Get the actual word content without spaces for optimization analysis
        word1 = s1_token.word
        word2 = s2_token.word
        
        # Try different character reuse strategies and pick the best
        strategies = [
            self._try_substring_containment(s1_token, s2_token),
            self._try_prefix_suffix_overlap(s1_token, s2_token),
            self._try_character_interleaving(s1_token, s2_token),
            self._try_basic_concatenation(s1_token, s2_token)
        ]
        
        # Choose the strategy with minimum length
        best_result = min(strategies, key=lambda x: x['length'])
        
        return prev_state.length + best_result['length']
    
    def _try_substring_containment(self, s1_token: WordToken, s2_token: WordToken) -> dict:
        """
        Try substring containment optimization.
        
        If one word contains the other as a substring, we can reuse those characters.
        
        Args:
            s1_token: Word token from first string
            s2_token: Word token from second string
            
        Returns:
            Dict with 'length' and 'strategy' keys
        """
        word1 = s1_token.word
        word2 = s2_token.word
        
        # Check if word1 contains word2
        if word2 in word1:
            # Use word1 as the base, word2 is contained within it
            total_spaces = max(s1_token.leading_spaces, s2_token.leading_spaces) + \
                          max(s1_token.trailing_spaces, s2_token.trailing_spaces)
            return {
                'length': len(word1) + total_spaces,
                'strategy': 'substring_containment_s1_contains_s2'
            }
        
        # Check if word2 contains word1
        if word1 in word2:
            # Use word2 as the base, word1 is contained within it
            total_spaces = max(s1_token.leading_spaces, s2_token.leading_spaces) + \
                          max(s1_token.trailing_spaces, s2_token.trailing_spaces)
            return {
                'length': len(word2) + total_spaces,
                'strategy': 'substring_containment_s2_contains_s1'
            }
        
        # No containment possible, return a high cost
        return {
            'length': float('inf'),
            'strategy': 'no_substring_containment'
        }
    
    def _try_prefix_suffix_overlap(self, s1_token: WordToken, s2_token: WordToken) -> dict:
        """
        Try prefix/suffix overlap optimization.
        
        Find the maximum overlap where the suffix of one word matches the prefix of another.
        
        Args:
            s1_token: Word token from first string
            s2_token: Word token from second string
            
        Returns:
            Dict with 'length' and 'strategy' keys
        """
        word1 = s1_token.word
        word2 = s2_token.word
        
        # Try word1 + word2 with overlap (word1's suffix overlaps with word2's prefix)
        max_overlap_12 = 0
        for i in range(1, min(len(word1), len(word2)) + 1):
            if word1[-i:] == word2[:i]:
                max_overlap_12 = i
        
        # Try word2 + word1 with overlap (word2's suffix overlaps with word1's prefix)
        max_overlap_21 = 0
        for i in range(1, min(len(word1), len(word2)) + 1):
            if word2[-i:] == word1[:i]:
                max_overlap_21 = i
        
        # Calculate costs for both arrangements
        if max_overlap_12 > 0:
            # word1 + word2 with overlap
            combined_word_length = len(word1) + len(word2) - max_overlap_12
            total_spaces = max(s1_token.leading_spaces, s2_token.leading_spaces) + \
                          max(s1_token.trailing_spaces, s2_token.trailing_spaces)
            cost_12 = combined_word_length + total_spaces
        else:
            cost_12 = float('inf')
        
        if max_overlap_21 > 0:
            # word2 + word1 with overlap
            combined_word_length = len(word1) + len(word2) - max_overlap_21
            total_spaces = max(s1_token.leading_spaces, s2_token.leading_spaces) + \
                          max(s1_token.trailing_spaces, s2_token.trailing_spaces)
            cost_21 = combined_word_length + total_spaces
        else:
            cost_21 = float('inf')
        
        # Return the better option
        if cost_12 <= cost_21 and cost_12 != float('inf'):
            return {
                'length': cost_12,
                'strategy': f'prefix_suffix_overlap_s1_s2_{max_overlap_12}'
            }
        elif cost_21 != float('inf'):
            return {
                'length': cost_21,
                'strategy': f'prefix_suffix_overlap_s2_s1_{max_overlap_21}'
            }
        else:
            return {
                'length': float('inf'),
                'strategy': 'no_prefix_suffix_overlap'
            }
    
    def _try_character_interleaving(self, s1_token: WordToken, s2_token: WordToken) -> dict:
        """
        Try strategic character interleaving while maintaining word boundaries.
        
        This strategy is conservative and only applies when it can maintain word integrity.
        For now, it returns a high cost to prefer other strategies that better preserve
        word boundaries as required by the design.
        
        Args:
            s1_token: Word token from first string
            s2_token: Word token from second string
            
        Returns:
            Dict with 'length' and 'strategy' keys
        """
        word1 = s1_token.word
        word2 = s2_token.word
        
        # Only apply character interleaving if words have significant character overlap
        # and the result would preserve word readability
        common_chars = set(word1) & set(word2)
        if len(common_chars) < 2:  # Need at least 2 common characters to be worthwhile
            return {
                'length': float('inf'),
                'strategy': 'no_character_interleaving'
            }
        
        # Use dynamic programming to find the shortest supersequence
        # that contains both words as subsequences
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        
        # Initialize base cases
        for i in range(len(word1) + 1):
            dp[i][0] = i
        for j in range(len(word2) + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i-1] == word2[j-1]:
                    # Characters match, we can reuse this character
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    # Characters don't match, take the minimum of adding either character
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1
        
        # The result is the length of the shortest common supersequence
        supersequence_length = dp[len(word1)][len(word2)]
        
        # Only use this strategy if it provides any savings
        basic_length = len(word1) + len(word2) + 1  # +1 for space between words
        if supersequence_length >= basic_length:  # No savings
            return {
                'length': float('inf'),
                'strategy': 'insufficient_interleaving_savings'
            }
        
        # Add spacing
        total_spaces = max(s1_token.leading_spaces, s2_token.leading_spaces) + \
                      max(s1_token.trailing_spaces, s2_token.trailing_spaces)
        
        return {
            'length': supersequence_length + total_spaces,
            'strategy': 'character_interleaving'
        }
    
    def _try_basic_concatenation(self, s1_token: WordToken, s2_token: WordToken) -> dict:
        """
        Try basic concatenation as a fallback strategy.
        
        Args:
            s1_token: Word token from first string
            s2_token: Word token from second string
            
        Returns:
            Dict with 'length' and 'strategy' keys
        """
        # Calculate total length with both words and necessary spacing
        word1_length = len(s1_token.word)
        word2_length = len(s2_token.word)
        
        # Use maximum leading and trailing spaces to preserve spacing requirements
        leading_spaces = max(s1_token.leading_spaces, s2_token.leading_spaces)
        trailing_spaces = max(s1_token.trailing_spaces, s2_token.trailing_spaces)
        
        # Add one space between words if neither has trailing/leading spaces
        inter_word_space = 0
        if s1_token.trailing_spaces == 0 and s2_token.leading_spaces == 0:
            inter_word_space = 1
        
        total_length = leading_spaces + word1_length + inter_word_space + word2_length + trailing_spaces
        
        return {
            'length': total_length,
            'strategy': 'basic_concatenation'
        }
    
