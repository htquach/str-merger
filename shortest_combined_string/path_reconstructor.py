"""
Path reconstruction for the Shortest Combined String algorithm.

This module implements the PathReconstructor class that rebuilds the optimal solution
from the DP table using backtracking algorithm to find the optimal character combination.
"""

from typing import List
from .models import WordToken, DPState, Operation, CombinedToken, TokenType


class PathReconstructor:
    """
    Reconstructs the optimal solution path from a DP table.
    
    This class implements a backtracking algorithm that traces through the DP table
    to rebuild the sequence of operations that led to the optimal solution, generating
    a sequence of CombinedToken objects that represent the final result.
    """
    
    def __init__(self):
        """Initialize the PathReconstructor."""
        pass
    
    def reconstruct_path(self, dp_table: List[List[DPState]], s1_tokens: List[WordToken], s2_tokens: List[WordToken]) -> List[CombinedToken]:
        """
        Reconstruct the optimal solution path from the DP table.
        
        Uses backtracking algorithm to trace through the DP table and generate
        the sequence of CombinedToken objects that represent the optimal solution.
        
        Args:
            dp_table: The filled DP table from the solver
            s1_tokens: Word tokens from the first string
            s2_tokens: Word tokens from the second string
            
        Returns:
            List of CombinedToken objects representing the optimal solution
            
        Raises:
            TypeError: If inputs are not of the expected types
            ValueError: If DP table dimensions don't match token lists
        """
        if not isinstance(dp_table, list):
            raise TypeError("dp_table must be a list")
        if not isinstance(s1_tokens, list):
            raise TypeError("s1_tokens must be a list")
        if not isinstance(s2_tokens, list):
            raise TypeError("s2_tokens must be a list")
        
        # Validate DP table structure
        if len(dp_table) != len(s1_tokens) + 1:
            raise ValueError(f"DP table rows ({len(dp_table)}) must equal s1_tokens length + 1 ({len(s1_tokens) + 1})")
        
        if dp_table and len(dp_table[0]) != len(s2_tokens) + 1:
            raise ValueError(f"DP table columns ({len(dp_table[0])}) must equal s2_tokens length + 1 ({len(s2_tokens) + 1})")
        
        # Validate token types
        for token in s1_tokens:
            if not isinstance(token, WordToken):
                raise TypeError("All s1_tokens must be WordToken objects")
        
        for token in s2_tokens:
            if not isinstance(token, WordToken):
                raise TypeError("All s2_tokens must be WordToken objects")
        
        # Validate DP table contents
        for i, row in enumerate(dp_table):
            if not isinstance(row, list):
                raise TypeError(f"DP table row {i} must be a list")
            for j, state in enumerate(row):
                if not isinstance(state, DPState):
                    raise TypeError(f"DP table entry [{i}][{j}] must be a DPState object")
        
        solution = []
        i = len(s1_tokens)
        j = len(s2_tokens)
        
        # Backtrack through the DP table to reconstruct the solution
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
                # Match/merge both words using optimal character reuse strategy
                s1_token = s1_tokens[i-1]
                s2_token = s2_tokens[j-1]
                
                # Determine which strategy was used and reconstruct accordingly
                content = self._reconstruct_optimized_match(s1_token, s2_token)
                
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
    
    def _reconstruct_optimized_match(self, s1_token: WordToken, s2_token: WordToken) -> str:
        """
        Reconstruct the optimized match content using the same strategy selection logic.
        
        This method replicates the optimization strategy selection from the DPSolver
        to ensure consistent reconstruction of the optimal character combination.
        
        Args:
            s1_token: Word token from first string
            s2_token: Word token from second string
            
        Returns:
            The optimized combined content string
        """
        # Use the same strategy selection logic as in DPSolver._calculate_match_cost
        strategies = [
            self._try_substring_containment(s1_token, s2_token),
            self._try_prefix_suffix_overlap(s1_token, s2_token),
            self._try_character_interleaving(s1_token, s2_token),
            self._try_basic_concatenation(s1_token, s2_token)
        ]
        
        # Choose the strategy with minimum length
        best_result = min(strategies, key=lambda x: x['length'])
        
        # Reconstruct the content based on the chosen strategy
        return self._build_content_from_strategy(s1_token, s2_token, best_result['strategy'])
    
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
    
    def _build_content_from_strategy(self, s1_token: WordToken, s2_token: WordToken, strategy: str) -> str:
        """
        Build the actual content string based on the optimization strategy.
        
        Args:
            s1_token: Word token from first string
            s2_token: Word token from second string
            strategy: The strategy identifier
            
        Returns:
            The combined content string
        """
        word1 = s1_token.word
        word2 = s2_token.word
        
        # Calculate spacing
        leading_spaces = max(s1_token.leading_spaces, s2_token.leading_spaces)
        trailing_spaces = max(s1_token.trailing_spaces, s2_token.trailing_spaces)
        
        if strategy == 'substring_containment_s1_contains_s2':
            # word1 contains word2, use word1
            return " " * leading_spaces + word1 + " " * trailing_spaces
            
        elif strategy == 'substring_containment_s2_contains_s1':
            # word2 contains word1, use word2
            return " " * leading_spaces + word2 + " " * trailing_spaces
            
        elif strategy.startswith('prefix_suffix_overlap_s1_s2_'):
            # word1 + word2 with overlap
            overlap_size = int(strategy.split('_')[-1])
            combined_word = word1 + word2[overlap_size:]
            return " " * leading_spaces + combined_word + " " * trailing_spaces
            
        elif strategy.startswith('prefix_suffix_overlap_s2_s1_'):
            # word2 + word1 with overlap
            overlap_size = int(strategy.split('_')[-1])
            combined_word = word2 + word1[overlap_size:]
            return " " * leading_spaces + combined_word + " " * trailing_spaces
            
        elif strategy == 'character_interleaving':
            # Reconstruct the shortest common supersequence
            supersequence = self._build_shortest_supersequence(word1, word2)
            return " " * leading_spaces + supersequence + " " * trailing_spaces
            
        else:  # basic_concatenation or fallback
            # Basic concatenation with space between words if needed
            inter_word_space = ""
            if s1_token.trailing_spaces == 0 and s2_token.leading_spaces == 0:
                inter_word_space = " "
            return " " * leading_spaces + word1 + inter_word_space + word2 + " " * trailing_spaces
    
    def _build_shortest_supersequence(self, word1: str, word2: str) -> str:
        """
        Build the shortest common supersequence of two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            The shortest supersequence containing both words as subsequences
        """
        # Use DP to build the supersequence
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1
        
        # Reconstruct the supersequence
        result = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if word1[i-1] == word2[j-1]:
                # Characters match, add once
                result.append(word1[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] < dp[i][j-1]:
                # Take from word1
                result.append(word1[i-1])
                i -= 1
            else:
                # Take from word2
                result.append(word2[j-1])
                j -= 1
        
        # Add remaining characters
        while i > 0:
            result.append(word1[i-1])
            i -= 1
        while j > 0:
            result.append(word2[j-1])
            j -= 1
        
        # Reverse since we built it backwards
        result.reverse()
        return ''.join(result)