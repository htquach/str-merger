"""
Word tokenization module for the Shortest Combined String algorithm.

This module provides the WordTokenizer class that splits strings into word tokens
while preserving space metadata for optimal character reuse and reconstruction.
"""

import re
from typing import List
from .models import WordToken


class WordTokenizer:
    """
    Tokenizes strings into words while preserving space information.
    
    This class splits input strings into WordToken objects that maintain
    information about leading and trailing spaces, enabling word-boundary
    aware dynamic programming and accurate reconstruction.
    """
    
    def tokenize(self, input_str: str) -> List[WordToken]:
        """
        Split a string into WordToken objects with space metadata.
        
        Args:
            input_str: The string to tokenize
            
        Returns:
            List of WordToken objects with space information preserved
            
        Raises:
            TypeError: If input_str is not a string
        """
        if not isinstance(input_str, str):
            raise TypeError("input_str must be a string")
        
        if not input_str:
            return []
        
        tokens = []
        word_index = 0
        
        # Find all words and their positions
        word_pattern = r'\S+'
        words = list(re.finditer(word_pattern, input_str))
        
        if not words:
            # String contains only spaces or is empty
            return []
        
        for i, match in enumerate(words):
            word = match.group()
            start_pos = match.start()
            end_pos = match.end()
            
            # Calculate leading spaces
            if i == 0:
                # First word: leading spaces from start of string
                leading_spaces = start_pos
            else:
                # Subsequent words: spaces since end of previous word
                prev_end = words[i-1].end()
                leading_spaces = start_pos - prev_end
            
            # Calculate trailing spaces
            if i == len(words) - 1:
                # Last word: trailing spaces to end of string
                trailing_spaces = len(input_str) - end_pos
            else:
                # Not last word: no trailing spaces (they become leading spaces of next word)
                trailing_spaces = 0
            
            token = WordToken(
                word=word,
                leading_spaces=leading_spaces,
                trailing_spaces=trailing_spaces,
                original_index=word_index
            )
            tokens.append(token)
            word_index += 1
        
        return tokens
    
    def reconstruct_from_tokens(self, tokens: List[WordToken]) -> str:
        """
        Reconstruct the original string from WordToken objects.
        
        Args:
            tokens: List of WordToken objects to reconstruct from
            
        Returns:
            Reconstructed string with original spacing preserved
            
        Raises:
            TypeError: If tokens is not a list or contains non-WordToken objects
        """
        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")
        
        if not tokens:
            return ""
        
        result_parts = []
        
        for token in tokens:
            if not isinstance(token, WordToken):
                raise TypeError("All tokens must be WordToken objects")
            
            # Add leading spaces
            result_parts.append(" " * token.leading_spaces)
            
            # Add the word
            result_parts.append(token.word)
            
            # Add trailing spaces
            result_parts.append(" " * token.trailing_spaces)
        
        return "".join(result_parts)