"""
Performance tests for the Shortest Combined String algorithm.

This module tests the performance characteristics of the algorithm, including
time complexity validation and memory usage optimization.
"""

import pytest
import time
import random
import string
from shortest_combined_string.shortest_combined_string import ShortestCombinedString


class TestPerformance:
    """Test cases for algorithm performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.algorithm = ShortestCombinedString()
    
    def test_time_complexity_validation(self):
        """Test that the algorithm has O(n*m) time complexity."""
        # Skip this test in CI environments or when running quick tests
        # pytest.skip("Performance test skipped for quick runs")
        
        # Define input sizes to test
        sizes = [10, 20, 30, 40, 50, 60, 70, 80]
        times = []
        products = []  # n*m products
        
        # Run algorithm with increasing input sizes
        for size in sizes:
            # Generate random strings of specified size
            s1 = self._generate_random_string(size)
            s2 = self._generate_random_string(size)
            
            # Measure execution time
            start_time = time.time()
            self.algorithm.combine(s1, s2)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            products.append(size * size)  # n*m = size*size since both strings have the same length
            
            print(f"Size: {size}, Time: {execution_time:.6f} seconds")
        
        # Verify O(n*m) complexity by checking if execution time grows approximately with n*m
        # We use a simple check to ensure the growth is roughly linear
        # This is a simplified approach since we don't have numpy for correlation calculation
        
        # Calculate growth ratios between consecutive measurements
        time_ratios = []
        product_ratios = []
        
        for i in range(1, len(times)):
            time_ratios.append(times[i] / times[i-1] if times[i-1] > 0 else 1)
            product_ratios.append(products[i] / products[i-1])
        
        # Print the ratios for inspection
        print("Growth ratios (time / product):")
        for i in range(len(time_ratios)):
            print(f"Step {i+1}: Time ratio = {time_ratios[i]:.2f}, Product ratio = {product_ratios[i]:.2f}")
        
        # Check if the growth ratios are roughly similar
        # This is a very simplified check, but should catch major deviations from O(n*m)
        ratio_diffs = [abs(time_ratios[i] - product_ratios[i]) for i in range(len(time_ratios))]
        avg_diff = sum(ratio_diffs) / len(ratio_diffs)
        
        print(f"Average difference between growth ratios: {avg_diff:.2f}")
        assert avg_diff < 1.0, "Time complexity does not appear to be O(n*m)"
    
    def test_memoization_effectiveness(self):
        """Test that memoization improves performance for repeated operations."""
        # Generate test strings with repeating patterns to maximize cache hits
        s1 = "abc def ghi " * 10
        s2 = "def jkl abc " * 10
        
        # First run (cold cache)
        start_time = time.time()
        self.algorithm.combine(s1, s2)
        first_run_time = time.time() - start_time
        
        # Second run (warm cache)
        start_time = time.time()
        self.algorithm.combine(s1, s2)
        second_run_time = time.time() - start_time
        
        print(f"First run: {first_run_time:.6f} seconds")
        print(f"Second run: {second_run_time:.6f} seconds")
        print(f"Speedup: {first_run_time / second_run_time:.2f}x")
        
        # The second run should be significantly faster due to memoization
        # We allow for some variance due to system load and other factors
        assert second_run_time < first_run_time, "Memoization did not improve performance"
    
    def test_space_complexity_with_large_inputs(self):
        """Test that the algorithm handles large inputs without excessive memory usage."""
        # Generate large input strings
        s1 = self._generate_random_string(500)
        s2 = self._generate_random_string(500)
        
        # This should complete without memory errors
        try:
            result = self.algorithm.combine(s1, s2)
            assert result is not None
            print(f"Successfully processed large inputs (500 chars each)")
            print(f"Result length: {len(result.combined_string)}")
        except MemoryError:
            pytest.fail("Algorithm failed due to excessive memory usage")
    
    def test_primary_case_performance(self):
        """Test performance on the primary test case from requirements."""
        s1 = "this is a red vase"
        s2 = "his son freddy love vase"
        
        # Run multiple times to get average performance
        num_runs = 100
        total_time = 0
        
        for _ in range(num_runs):
            start_time = time.time()
            result = self.algorithm.combine(s1, s2)
            end_time = time.time()
            total_time += (end_time - start_time)
        
        avg_time = total_time / num_runs
        print(f"Primary case average execution time ({num_runs} runs): {avg_time:.6f} seconds")
        
        # Verify the result meets the length requirement
        assert len(result.combined_string) <= 26, f"Result length ({len(result.combined_string)}) exceeds requirement (26)"
    
    def test_performance_with_varying_input_ratios(self):
        """Test performance with different ratios of input string lengths."""
        base_size = 50
        ratios = [(1, 1), (1, 2), (1, 5), (1, 10)]
        
        for ratio in ratios:
            s1_size = base_size * ratio[0]
            s2_size = base_size * ratio[1]
            
            s1 = self._generate_random_string(s1_size)
            s2 = self._generate_random_string(s2_size)
            
            start_time = time.time()
            self.algorithm.combine(s1, s2)
            execution_time = time.time() - start_time
            
            print(f"Ratio {ratio[0]}:{ratio[1]} - s1:{s1_size}, s2:{s2_size}, Time: {execution_time:.6f} seconds")
    
    def _generate_random_string(self, size):
        """Generate a random string of specified size with word-like structure."""
        # Create a string with words separated by spaces
        words = []
        remaining_chars = size
        
        while remaining_chars > 0:
            # Generate a random word length between 2 and 7
            word_length = min(random.randint(2, 7), remaining_chars)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
            remaining_chars -= word_length
            
            # Add a space if there are more characters to add
            if remaining_chars > 0:
                words.append(' ')
                remaining_chars -= 1
        
        return ''.join(words)