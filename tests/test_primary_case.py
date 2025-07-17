"""
Comprehensive test suite for the primary test case of the ShortestCombinedString algorithm.

This test suite focuses specifically on the primary test case:
s1="this is a red vase", s2="his son freddy love vase"

Tests verify:
1. Output length ≤ 26 characters
2. Subsequence preservation and word integrity
3. Performance confirming O(n*m) time complexity
"""

import pytest
import time
from typing import List, Tuple
from shortest_combined_string.shortest_combined_string import ShortestCombinedString
from shortest_combined_string.subsequence_verifier import SubsequenceVerifier


class TestPrimaryCase:
    """Comprehensive test suite for the primary test case."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.algorithm = ShortestCombinedString()
        self.verifier = SubsequenceVerifier()
        
        # Primary test case strings
        self.s1 = "this is a red vase"
        self.s2 = "his son freddy love vase"
    
    def test_primary_case_length_requirement(self):
        """Test that the primary case output meets the length requirement."""
        result = self.algorithm.combine(self.s1, self.s2)
        
        # Verify result is valid
        assert result.is_valid, f"Result should be valid but got: {result.validation_errors}"
        
        # Verify length requirement (≤ 26 characters)
        assert len(result.combined_string) <= 26, \
            f"Output length should be ≤ 26 but got {len(result.combined_string)}: '{result.combined_string}'"
        
        # Print the actual output for reference
        print(f"Primary case output: '{result.combined_string}' (length: {len(result.combined_string)})")
    
    def test_primary_case_subsequence_preservation(self):
        """Test that the primary case output preserves both input strings as subsequences."""
        result = self.algorithm.combine(self.s1, self.s2)
        
        # Verify result is valid
        assert result.is_valid, f"Result should be valid but got: {result.validation_errors}"
        
        # Get detailed verification result
        verification_result = self.verifier.verify(self.s1, self.s2, result.combined_string)
        
        # Verify both inputs are valid subsequences
        assert verification_result.s1_match.is_valid, \
            f"First input should be a valid subsequence: {verification_result.s1_match.error_details}"
        assert verification_result.s2_match.is_valid, \
            f"Second input should be a valid subsequence: {verification_result.s2_match.error_details}"
        
        # Print detailed match information
        detailed_info = self.verifier.get_detailed_match_info(verification_result)
        print(f"\nDetailed match information:\n{detailed_info}")
    
    def test_primary_case_word_integrity(self):
        """Test that the primary case output maintains word integrity."""
        result = self.algorithm.combine(self.s1, self.s2)
        
        # Verify result is valid
        assert result.is_valid, f"Result should be valid but got: {result.validation_errors}"
        
        # Extract all words from both input strings
        s1_words = [word for word in self.s1.split() if word]
        s2_words = [word for word in self.s2.split() if word]
        all_words = set(s1_words + s2_words)
        
        # Check that all words from both inputs appear intact in the output
        for word in all_words:
            assert word in result.combined_string, f"Word '{word}' should appear intact in output"
        
        # Verify no words are broken (this is a simplified check)
        # A more thorough check would verify exact word boundaries
        output_words = result.combined_string.split()
        for word in output_words:
            # Each output word should either be from s1, s2, or a valid merge
            is_valid_word = (word in s1_words or word in s2_words or 
                            any(w1 in word and w2 in word for w1 in s1_words for w2 in s2_words))
            assert is_valid_word, f"Output word '{word}' is not a valid word from inputs"
    
    def test_primary_case_reversed_inputs(self):
        """Test the primary case with inputs in reverse order."""
        # Swap s1 and s2
        result = self.algorithm.combine(self.s2, self.s1)
        
        # Verify result is valid
        assert result.is_valid, f"Result should be valid but got: {result.validation_errors}"
        
        # Verify length requirement (≤ 26 characters)
        assert len(result.combined_string) <= 26, \
            f"Output length should be ≤ 26 but got {len(result.combined_string)}: '{result.combined_string}'"
        
        # Print the actual output for reference
        print(f"Reversed inputs output: '{result.combined_string}' (length: {len(result.combined_string)})")
    
    def test_primary_case_performance(self):
        """Test the performance of the algorithm on the primary test case."""
        # Warm-up run
        self.algorithm.combine(self.s1, self.s2)
        
        # Measure execution time
        start_time = time.time()
        result = self.algorithm.combine(self.s1, self.s2)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Primary case execution time: {execution_time:.6f} seconds")
        
        # The execution time should be reasonable for the input size
        # This is a basic performance check, not a strict assertion
        assert execution_time < 1.0, f"Execution time ({execution_time:.6f}s) is too high for the input size"
    
    def test_time_complexity(self):
        """Test that the algorithm's time complexity is O(n*m)."""
        # Generate test cases with increasing sizes
        # Use larger sizes to make timing differences more noticeable
        test_cases = self._generate_test_cases(max_size=500, step=100)
        
        # Measure execution times
        sizes = []
        times = []
        
        for size, (s1, s2) in test_cases:
            # Warm-up run
            self.algorithm.combine(s1, s2)
            
            # Measure execution time with multiple runs for more stability
            num_runs = 5
            total_time = 0
            for _ in range(num_runs):
                start_time = time.time()
                self.algorithm.combine(s1, s2)
                end_time = time.time()
                total_time += (end_time - start_time)
            
            execution_time = total_time / num_runs
            sizes.append(size)
            times.append(execution_time)
            
            print(f"Size {size}: {execution_time:.6f} seconds (avg of {num_runs} runs)")
        
        # Verify O(n*m) complexity by checking if execution time grows approximately linearly
        # with the product of input sizes (n*m)
        # This is a simplified check without using numpy/matplotlib
        
        # For very small execution times, timing noise can dominate
        # So we'll check if the largest time is significantly larger than the smallest
        # which would indicate growth with input size
        min_time = min(times)
        max_time = max(times)
        
        print(f"Min time: {min_time:.6f}, Max time: {max_time:.6f}")
        print(f"Max/Min ratio: {max_time/min_time if min_time > 0 else 'N/A'}")
        
        # Check if the largest input size takes significantly more time than the smallest
        # This is a more resilient check than requiring strictly increasing times
        assert max_time > min_time * 1.5, "Execution time should grow with input size"
        
        # Check if the growth is roughly linear or better (not exponential)
        # by comparing the largest time with the largest input size
        largest_size = max(sizes)
        smallest_size = min(sizes)
        
        # For O(n*m), time should grow no faster than quadratically with input size
        # So max_time should be less than min_time * (largest_size/smallest_size)^2
        max_allowed_ratio = (largest_size / smallest_size) ** 2
        actual_ratio = max_time / min_time if min_time > 0 else float('inf')
        
        print(f"Max allowed growth ratio: {max_allowed_ratio:.4f}")
        print(f"Actual growth ratio: {actual_ratio:.4f}")
        
        assert actual_ratio < max_allowed_ratio, \
            f"Growth ratio ({actual_ratio:.4f}) suggests worse than O(n*m) complexity"
    
    def _generate_test_cases(self, max_size: int, step: int) -> List[Tuple[int, Tuple[str, str]]]:
        """
        Generate test cases with increasing sizes.
        
        Args:
            max_size: Maximum size of test cases
            step: Step size between test cases
            
        Returns:
            List of (size, (s1, s2)) tuples
        """
        test_cases = []
        
        for size in range(step, max_size + 1, step):
            # Create strings of increasing size
            s1 = "a" * size
            s2 = "b" * size
            
            test_cases.append((size, (s1, s2)))
        
        return test_cases