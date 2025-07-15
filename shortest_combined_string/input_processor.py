"""
Input validation and preprocessing for the Shortest Combined String algorithm.

This module handles input validation, consecutive space detection and normalization,
and generates appropriate warnings for preprocessing operations.
"""

import re
from typing import List
from .models import PreprocessedInput


class InputProcessor:
    """
    Handles input validation and preprocessing for the algorithm.
    
    This class is responsible for:
    - Detecting consecutive spaces in input strings
    - Normalizing consecutive spaces to single spaces
    - Generating warnings for preprocessing operations
    - Validating input format and content
    """
    
    def __init__(self):
        """Initialize the InputProcessor."""
        # Regex pattern to detect two or more consecutive spaces
        self._consecutive_spaces_pattern = re.compile(r' {2,}')
    
    def validate_and_preprocess(self, s1: str, s2: str) -> PreprocessedInput:
        """
        Validate and preprocess two input strings.
        
        Args:
            s1: First input string
            s2: Second input string
            
        Returns:
            PreprocessedInput: Processed input with warnings and metadata
            
        Raises:
            TypeError: If inputs are not strings
            ValueError: If inputs are None
        """
        # Input validation
        if s1 is None or s2 is None:
            raise ValueError("Input strings cannot be None")
        
        if not isinstance(s1, str) or not isinstance(s2, str):
            raise TypeError("Both inputs must be strings")
        
        warnings = []
        has_consecutive_spaces = False
        
        # Process first string
        processed_s1, s1_warnings, s1_has_consecutive = self._process_single_string(s1, "s1")
        warnings.extend(s1_warnings)
        has_consecutive_spaces = has_consecutive_spaces or s1_has_consecutive
        
        # Process second string
        processed_s2, s2_warnings, s2_has_consecutive = self._process_single_string(s2, "s2")
        warnings.extend(s2_warnings)
        has_consecutive_spaces = has_consecutive_spaces or s2_has_consecutive
        
        return PreprocessedInput(
            s1=processed_s1,
            s2=processed_s2,
            warnings=warnings,
            has_consecutive_spaces=has_consecutive_spaces
        )
    
    def _process_single_string(self, input_str: str, string_name: str) -> tuple[str, List[str], bool]:
        """
        Process a single input string for consecutive spaces.
        
        Args:
            input_str: The string to process
            string_name: Name of the string for warning messages
            
        Returns:
            tuple: (processed_string, warnings, has_consecutive_spaces)
        """
        warnings = []
        has_consecutive_spaces = False
        
        # Check for consecutive spaces
        consecutive_matches = list(self._consecutive_spaces_pattern.finditer(input_str))
        
        if consecutive_matches:
            has_consecutive_spaces = True
            
            # Generate detailed warning about consecutive spaces found
            space_locations = []
            for match in consecutive_matches:
                start_pos = match.start()
                space_count = len(match.group())
                space_locations.append(f"position {start_pos} ({space_count} spaces)")
            
            warning_msg = (
                f"Consecutive spaces detected in {string_name} at {', '.join(space_locations)}. "
                f"Automatically normalized to single spaces."
            )
            warnings.append(warning_msg)
            
            # Normalize consecutive spaces to single spaces
            processed_str = self._consecutive_spaces_pattern.sub(' ', input_str)
        else:
            processed_str = input_str
        
        return processed_str, warnings, has_consecutive_spaces
    
    def has_consecutive_spaces(self, input_str: str) -> bool:
        """
        Check if a string contains consecutive spaces.
        
        Args:
            input_str: String to check
            
        Returns:
            bool: True if consecutive spaces are found
        """
        if not isinstance(input_str, str):
            return False
        
        return bool(self._consecutive_spaces_pattern.search(input_str))
    
    def normalize_spaces(self, input_str: str) -> str:
        """
        Normalize consecutive spaces in a string to single spaces.
        
        Args:
            input_str: String to normalize
            
        Returns:
            str: String with consecutive spaces normalized
        """
        if not isinstance(input_str, str):
            raise TypeError("Input must be a string")
        
        return self._consecutive_spaces_pattern.sub(' ', input_str)